"""Three-dimensional (r, theta, z) finite-volume solver for off-axis voids.

The copper rod, the silica shell and the laser source are axisymmetric; only the voids break the
symmetry.  A single void centred on the theta = 0 half-plane makes the problem mirror-symmetric in
theta, so only the half cylinder 0 <= theta <= pi is discretised (zero-flux faces at theta = 0 and
theta = pi).  Several voids, or a void at another azimuth, are solved on the full circle
0 <= theta < 2 pi with a periodic azimuthal link (`full_circle`).  The (r, z) grid is the same one the axisymmetric solver uses, so an axisymmetric
field on this grid reproduces the 2-D result exactly: the no-void baseline can stay 2-D and the
void signal (3-D void run minus 2-D baseline) is consistent.

Azimuthal grid: automatic.  The angular width of the void seen from the axis is resolved by at
least `void_cells` cells (uniform zone around theta = 0), then the spacing grows geometrically up to
`theta_max` towards theta = pi.  Non-uniform theta spacings are handled exactly by the finite-volume
link areas and centre distances.

Cells: p = (j * nr + i) * nth + m  with i radial, j axial, m azimuthal.  The innermost cells are
wedges that touch the axis; they need no special treatment because their radial inner face has
zero area and their azimuthal faces are handled like any other.

Linear solves: SuperLU when the system is small, otherwise conjugate gradients with a two-level
preconditioner built from the grid structure (`_StructuredPC`): exact (r, z) solves in every azimuthal
sector, exact solves along every azimuthal ring, and a coarse correction on the lowest Fourier modes
in theta.  The matrix C/dt + theta L is SPD; the preconditioner is rebuilt whenever dt changes (a few
seconds, against minutes for an ILU of comparable quality, which moreover broke down on multi-void
meshes whose cells are much finer than the diffusion length per step).  All snapshots are stored as
the (x, z) plane through the void (theta = pi | theta = 0), which is what the GUI displays.
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .solver import (
    MAT_COPPER, MAT_OXIDE, MAT_PAD, MAT_SILICA, MAT_VOID, SimConfig, SimResult, _insert_zone, _transition_faces,
    _zoned_axis, build_grid, interface_resistance,
    beam_annulus_weights,
    gaussian_annulus_weights,
)

DIRECT_MAX_UNKNOWNS = 60_000
TWO_PI = 2.0 * math.pi


def needs_3d(cfg: SimConfig) -> bool:
    """3-D is required as soon as one void is displaced from the axis (several on-axis voids stay 2-D)."""
    return any(v.r_center > 0.0 for v in cfg.voids)


def full_circle(cfg: SimConfig) -> bool:
    """Full 0..2 pi domain: more than one void, or a void away from the theta = 0 half-plane.  A single
    void at theta = 0 keeps the mirror-symmetric half cylinder."""
    voids = cfg.voids
    if len(voids) > 1:
        return True
    return bool(voids) and abs(math.remainder(voids[0].theta, TWO_PI)) > 1e-9


def max_unknowns(cfg: SimConfig) -> int:
    """Cap on the number of 3-D unknowns: four times the 2-D cell cap."""
    return cfg.numerics.max_cells * 4


def count_unknowns(cfg: SimConfig) -> tuple[int, int, int]:
    """((r, z) cells, azimuthal cells, unknowns) of the 3-D run `cfg` would make.  Cheap (only the grids are
    built).  Raises ValueError when the (r, z) grid alone exceeds `max_cells`."""
    n_cells = build_grid(cfg).n_cells
    nth = len(theta_faces(cfg)) - 1
    return n_cells, nth, n_cells * nth


def fit_resolution(cfg: SimConfig, min_void_cells: int = 3) -> SimConfig:
    """Lower `void_cells` one step at a time (never below `min_void_cells`) until the 3-D unknowns fit under
    `max_unknowns`.  Many small voids spread over the rod make the fine zones in r, z and theta cover most of
    the domain, so the resolution per void is the only knob left; a void resolved by 3 cells across still
    blocks the heat it should (the probe signal integrates over the whole arrangement).  Returns `cfg`
    unchanged when it already fits or is not a 3-D case; the caller compares the result with `max_unknowns`
    because the floor may not be enough."""
    if not needs_3d(cfg):
        return cfg
    limit = max_unknowns(cfg)
    while True:
        try:
            if count_unknowns(cfg)[2] <= limit:
                return cfg
        except ValueError:                       # (r, z) grid over the 2-D cap as well: keep coarsening
            pass
        vc = cfg.numerics.void_cells
        if vc <= min_void_cells:
            return cfg
        cfg = replace(cfg, numerics=replace(cfg.numerics, void_cells=vc - 1))


def void_angular_halfwidth(v) -> float:
    """Half of the angle subtended by a void about the axis (pi/2 if it contains the axis)."""
    if v.r_center <= v.r_half:
        return math.pi / 2
    return math.asin(v.r_half / v.r_center)


def theta_faces(cfg: SimConfig) -> np.ndarray:
    """Azimuthal faces.

    Half cylinder [0, pi] (single void at theta = 0): uniform fine zone over the void's angular extent
    (>= void_cells cells across the full width, plus a margin), then geometric growth up to theta_max.
    Full circle [0, 2 pi] (several voids / other azimuths): one fine zone per void around its centre
    (wrapped at 2 pi), graded transitions between them, theta_max elsewhere."""
    n = cfg.numerics
    if not full_circle(cfg):
        half = void_angular_halfwidth(cfg.void)
        d_fine = min(2.0 * half / n.void_cells, n.theta_max)
        margin = min(half, 5.0 * d_fine)
        th_zone = min(math.pi, half + margin)
        n_zone = max(1, int(round(th_zone / d_fine)))
        faces = np.linspace(0.0, th_zone, n_zone + 1)
        if th_zone < math.pi - 1e-12:
            faces = np.concatenate([faces, _transition_faces(th_zone, math.pi, d_fine, n.theta_max, n.stretch)])
        faces[-1] = math.pi
        return faces
    zones = []
    for v in cfg.voids:
        half = void_angular_halfwidth(v)
        d_fine = min(2.0 * half / n.void_cells, n.theta_max)
        if half >= math.pi / 2 - 1e-12:                       # the void contains the axis: seen from every theta
            zones = _insert_zone(zones, (0.0, TWO_PI, d_fine, []))
            continue
        margin = min(half, 5.0 * d_fine)
        lo, hi = v.theta - half - margin, v.theta + half + margin
        lo, hi = lo % TWO_PI, lo % TWO_PI + (hi - lo)
        if hi <= TWO_PI:
            zones = _insert_zone(zones, (lo, hi, d_fine, []))
        else:                                                # wraps past 2 pi: two pieces
            zones = _insert_zone(zones, (lo, TWO_PI, d_fine, []))
            zones = _insert_zone(zones, (0.0, hi - TWO_PI, d_fine, []))
    if not zones:
        zones = [(0.0, TWO_PI, n.theta_max, [])]
    faces = _zoned_axis(zones, TWO_PI, n.stretch)
    faces[-1] = TWO_PI
    return faces


def material_map_3d(cfg: SimConfig, grid, th_faces: np.ndarray) -> np.ndarray:
    """Material id per cell, shape (nz, nr, nth): the union of all voids (each a spheroid, or a box, around
    its own centre at (r_center, theta))."""
    th_c = 0.5 * (th_faces[1:] + th_faces[:-1])
    nth = len(th_c)
    rc, zc = grid.r_c, grid.z_c
    mat = np.full((grid.nz, grid.nr, nth), MAT_COPPER, dtype=np.int8)
    if not cfg.geometry.homogeneous_copper:
        mat[:, rc >= cfg.geometry.R_cu, :] = MAT_SILICA
    X = rc[None, :, None] * np.cos(th_c)[None, None, :]              # (1, nr, nth)
    Y = rc[None, :, None] * np.sin(th_c)[None, None, :]
    Z = zc[:, None, None]
    in_rod = rc[None, :, None] < cfg.geometry.R_cu
    for v in cfg.voids:
        xc, yc = v.r_center * math.cos(v.theta), v.r_center * math.sin(v.theta)
        rho2 = (X - xc) ** 2 + (Y - yc) ** 2                         # squared distance from the void axis
        if v.shape == "ellipse":
            inside = rho2 / v.r_half ** 2 + ((Z - v.z_center) / (0.5 * v.thickness)) ** 2 < 1.0
        else:
            inside = (rho2 < v.r_half ** 2) & (Z >= v.depth) & (Z < v.z_bottom)
        mat[np.broadcast_to(inside & in_rod, mat.shape)] = MAT_VOID
    b = cfg.bottom
    if b.enabled:                                    # layers below the rod span the whole radius
        z_ox = cfg.geometry.L + b.oxide_cells_thickness
        mat[(zc > cfg.geometry.L) & (zc < z_ox), :, :] = MAT_OXIDE
        mat[zc >= z_ox, :, :] = MAT_PAD
    return mat


def _properties(cfg: SimConfig, mat: np.ndarray):
    k = np.empty(mat.shape)
    rho_cp = np.empty(mat.shape)
    for mid, (kk, rc) in {
        MAT_COPPER: (cfg.copper.k, cfg.copper.rho_cp),
        MAT_SILICA: (cfg.silica.k, cfg.silica.rho_cp),
        MAT_VOID: (cfg.void.k, cfg.void.rho_cp),
        MAT_OXIDE: (cfg.bottom.oxide_k, cfg.bottom.oxide_rho_cp),
        MAT_PAD: (cfg.bottom.pad_k, cfg.bottom.pad_rho_cp),
    }.items():
        sel = mat == mid
        k[sel] = kk
        rho_cp[sel] = rc
    return k, rho_cp


def assemble_laplacian_3d(grid, k3: np.ndarray, th_faces: np.ndarray, r_extra_z=None, periodic: bool = False) -> sp.csr_matrix:
    """Conductance Laplacian (W/K).  Half cylinder: zero-flux at theta = 0 and theta = pi.  Full circle
    (`periodic`): the last sector is linked to the first."""
    nr, nz = grid.nr, grid.nz
    rf, zf, rc, zc = grid.r_faces, grid.z_faces, grid.r_c, grid.z_c
    dzc, drc = grid.dz_c, grid.dr_c
    dth = np.diff(th_faces)                                  # (nth,)
    nth = len(dth)
    idx = np.arange(nz * nr * nth).reshape(nz, nr, nth)

    # radial links (i, i+1): face area r_face * dtheta * dz
    A_r = rf[1:-1][None, :, None] * dth[None, None, :] * dzc[:, None, None]
    R_r = (rf[1:-1] - rc[:-1])[None, :, None] / k3[:, :-1, :] + (rc[1:] - rf[1:-1])[None, :, None] / k3[:, 1:, :]
    G_r = (A_r / R_r).ravel()
    p_r, q_r = idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel()

    # axial links (j, j+1): face area 0.5 * dtheta * (r_out^2 - r_in^2)
    A_z = 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2)[None, :, None] * dth[None, None, :]
    R_z = (zf[1:-1] - zc[:-1])[:, None, None] / k3[:-1] + (zc[1:] - zf[1:-1])[:, None, None] / k3[1:]
    if r_extra_z is not None:
        R_z = R_z + r_extra_z[:, None, None]
    G_z = (A_z / R_z).ravel()
    p_z, q_z = idx[:-1].ravel(), idx[1:].ravel()

    # azimuthal links (m, m+1): face area dr*dz; centre-to-face distances r_c*dtheta_m/2 and r_c*dtheta_{m+1}/2
    A_t = drc[None, :, None] * dzc[:, None, None]
    R_t = (0.5 * rc)[None, :, None] * (dth[:-1][None, None, :] / k3[:, :, :-1] + dth[1:][None, None, :] / k3[:, :, 1:])
    G_t = (A_t / R_t).ravel()
    p_t, q_t = idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel()
    if periodic:                                             # sector nth-1 <-> sector 0 across theta = 0 = 2 pi
        R_w = (0.5 * rc)[None, :, None] * (dth[-1] / k3[:, :, -1:] + dth[0] / k3[:, :, :1])
        G_t = np.concatenate([G_t, (A_t / R_w).ravel()])
        p_t = np.concatenate([p_t, idx[:, :, -1].ravel()])
        q_t = np.concatenate([q_t, idx[:, :, 0].ravel()])

    p = np.concatenate([p_r, p_z, p_t])
    q = np.concatenate([q_r, q_z, q_t])
    G = np.concatenate([G_r, G_z, G_t])
    N = nz * nr * nth
    rows = np.concatenate([p, q, p, q])
    cols = np.concatenate([p, q, q, p])
    data = np.concatenate([G, G, -G, -G])
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


def _submatrix(A: sp.csc_matrix, keep) -> sp.csc_matrix:
    """A with only the entries for which keep(row, col) holds (the diagonal is always kept).  Dropping
    off-diagonal links of an SPD M-matrix while keeping its diagonal leaves an SPD M-matrix."""
    Ac = A.tocoo()
    m = keep(Ac.row, Ac.col) | (Ac.row == Ac.col)
    return sp.csc_matrix((Ac.data[m], (Ac.row[m], Ac.col[m])), shape=A.shape)


class _StructuredPC:
    """Two-level SPD preconditioner for A = C/dt + theta L on the (z, r, theta) cell ordering p = ji * nth + m.

    Smoother (symmetric, multiplicative): exact solve along every azimuthal ring (only the theta links
    kept: block-diagonal with nth x nth blocks), then exact solve of the (r, z) problem of every azimuthal
    sector (the azimuthal links dropped: nth independent 2-D systems, one SuperLU factorisation), then the
    ring solve again.  Coarse level: the span of the lowest `n_modes` Fourier modes in theta on every (r, z)
    cell, Galerkin operator P^T A P factorised directly (nc * n_modes unknowns with 2-D sparsity).

    Why this and not ILU: with a long pulse the diffusion length per step sqrt(D dt) spans many cells of a
    finely resolved void, so A is a near-Laplacian over wide regions and an incomplete factorisation
    cannot represent its inverse (BiCGSTAB broke down, residuals stalled at 1e-3..1e-8).  The sector and
    ring solves are exact in the two strongly coupled directions and the Fourier coarse space carries the
    long-range, smooth-in-theta error that remains; CG then converges in 10-25 iterations independently of
    the cell/diffusion-length ratio.  Setup is a few seconds for 10^6 unknowns."""

    def __init__(self, A: sp.csc_matrix, nth: int, th_faces: np.ndarray, periodic: bool, n_modes: int = 9):
        self.A = A
        N = A.shape[0]
        nc = N // nth
        self.lu_sector = spla.splu(_submatrix(A, lambda r, c: (r % nth) == (c % nth)), permc_spec="MMD_AT_PLUS_A")
        self.lu_ring = spla.splu(_submatrix(A, lambda r, c: (r // nth) == (c // nth)), permc_spec="NATURAL")
        th_c = 0.5 * (th_faces[1:] + th_faces[:-1])
        n_modes = max(1, min(n_modes, nth))
        cols = [np.ones(nth)]
        k = 1
        while len(cols) < n_modes:
            if periodic:                                     # full circle: cos and sin pairs
                cols.append(np.cos(k * th_c))
                if len(cols) < n_modes:
                    cols.append(np.sin(k * th_c))
            else:                                            # half cylinder, zero flux at 0 and pi: cos k theta
                cols.append(np.cos(k * th_c))
            k += 1
        Phi = sp.csr_matrix(np.array(cols).T)
        self.P = sp.kron(sp.identity(nc, format="csr"), Phi, format="csr")
        self.lu_coarse = spla.splu((self.P.T @ A @ self.P).tocsc(), permc_spec="MMD_AT_PLUS_A")
        self.op = spla.LinearOperator(A.shape, self.apply)

    def _smooth(self, r: np.ndarray) -> np.ndarray:
        # ring - sector - ring: symmetric, and the cheap ring solve is the one done twice
        z = self.lu_ring.solve(r)
        z = z + self.lu_sector.solve(r - self.A @ z)
        return z + self.lu_ring.solve(r - self.A @ z)

    def apply(self, r: np.ndarray) -> np.ndarray:
        z = self._smooth(r)
        z = z + self.P @ self.lu_coarse.solve(self.P.T @ (r - self.A @ z))
        return z + self._smooth(r - self.A @ z)


SOLVER_DIRECT, SOLVER_STRUCTURED, SOLVER_ILU = "직접 LU (SuperLU)", "구조 전처리 CG", "ILU + BiCGSTAB (예비)"


class _Solver:
    """Solve of A x = b, rebuilt per dt: SuperLU when small, otherwise CG with the structured preconditioner.
    If that preconditioner cannot be built (SuperLU reports a failed allocation as a RuntimeError, or a
    MemoryError is raised), the solver falls back to the earlier ILU-preconditioned BiCGSTAB/GMRES, which
    needs less memory at setup; `kind` records what was used."""

    def __init__(self, A: sp.csc_matrix, direct: bool, nth: int = 0, th_faces=None, periodic: bool = False):
        # A = C/dt + theta * L  (theta = 0.5 Crank-Nicolson, 1.0 backward Euler after a source discontinuity)
        self.direct = direct
        self.n_iter = []
        self.A = A
        if direct:
            self.kind = SOLVER_DIRECT
            self.lu = spla.splu(A)
            return
        gc.collect()                                          # release the previous step's factors first
        try:
            self.M = _StructuredPC(A, nth, th_faces, periodic).op
            self.kind = SOLVER_STRUCTURED
        except (MemoryError, RuntimeError) as e:              # SuperLU: "SUPERLU_MALLOC fails ..."
            if isinstance(e, RuntimeError) and "MALLOC" not in str(e).upper():
                raise
            gc.collect()
            ilu = spla.spilu(A, drop_tol=1e-5, fill_factor=12)
            self.M = spla.LinearOperator(A.shape, ilu.solve)
            self.kind = SOLVER_ILU

    def solve(self, b: np.ndarray, x0: np.ndarray) -> np.ndarray:
        if self.direct:
            return self.lu.solve(b)
        cnt = [0]

        def cb(_):
            cnt[0] += 1

        scale = float(np.max(np.abs(b))) or 1.0
        if self.kind == SOLVER_STRUCTURED:
            x, info = spla.cg(self.A, b, x0=x0, M=self.M, rtol=1e-11, atol=1e-13 * scale, maxiter=200, callback=cb)
        else:
            x, info = spla.bicgstab(self.A, b, x0=x0, M=self.M, rtol=1e-11, atol=1e-13 * scale, maxiter=400, callback=cb)
            if info != 0:
                x, info = spla.gmres(self.A, b, x0=x, M=self.M, rtol=1e-11, atol=1e-13 * scale, restart=50, maxiter=40)
        if info != 0:
            raise RuntimeError("3D 선형해가 수렴하지 않았습니다 (반복 솔버). Δt 성장률을 낮추거나 격자를 조정하세요.")
        self.n_iter.append(cnt[0])
        return x


def run_simulation_3d(
    cfg: SimConfig,
    progress: Optional[Callable[[float], None]] = None,
    store_fields: bool = True,
) -> SimResult:
    t_start = time.perf_counter()
    grid = build_grid(cfg)
    th_f = theta_faces(cfg)
    dth = np.diff(th_f)
    nth = len(dth)
    periodic = full_circle(cfg)
    sym = 1.0 if periodic else 2.0                           # the half cylinder stands for both halves
    m_pi = int(np.argmin(np.abs(0.5 * (th_f[1:] + th_f[:-1]) - math.pi)))   # sector of the displayed theta = pi half-plane
    mat = material_map_3d(cfg, grid, th_f)
    k3, rho_cp = _properties(cfg, mat)
    nr, nz = grid.nr, grid.nz
    N = nz * nr * nth
    if N > max_unknowns(cfg):
        raise ValueError(
            f"3D 미지수 {N:,} ((r, z) 셀 {nr * nz:,} × 방위각 {nth}) 가 한도({max_unknowns(cfg):,})를 넘습니다. "
            "void 개수를 줄이거나 void 를 크게 하거나 더 깊은 프리셋(긴 펄스 → 거친 격자)을 고르세요."
        )

    A_z_cell = 0.5 * (grid.r_faces[1:] ** 2 - grid.r_faces[:-1] ** 2)[None, :, None] * dth[None, None, :]   # (1, nr, nth) top-face area
    V = A_z_cell * grid.dz_c[:, None, None]
    C = (rho_cp * V).ravel()                                     # J/K per cell
    Lap = assemble_laplacian_3d(grid, k3, th_f, interface_resistance(cfg, grid), periodic=periodic)
    # heat sink below the thermal pad: far face of the last cell row held at ambient (theta = 0)
    g_sink = np.zeros(N)
    if cfg.bottom.enabled and cfg.bottom.sink == "isothermal":
        g_sink[-nr * nth:] = (k3[-1] * A_z_cell[0] / (0.5 * grid.dz_c[-1])).ravel()
        Lap = (Lap + sp.diags(g_sink)).tocsr()
    Cdiag = sp.diags(C)
    times = cfg.time_grid()
    dts = np.diff(times)

    # laser source on the top faces of copper cells: axisymmetric beam, per-sector fraction dtheta/(2 pi)
    la = cfg.laser
    q_abs = la.I0 * (1.0 - la.reflectivity)
    W_ann = beam_annulus_weights(grid.r_faces, la, cfg.geometry.R_cu)                             # (nr,)
    src3 = np.zeros((nz, nr, nth))
    src3[0] = q_abs * W_ann[:, None] * dth[None, :] / (2.0 * math.pi)
    src3[0][mat[0] != MAT_COPPER] = 0.0
    src = src3.ravel()
    src_total = src.sum()

    # surface reconstruction and probe weights (same definitions as the 2-D solver)
    dz0 = grid.dz_c[0]
    face_corr = (src3[0] / A_z_cell[0]) * (0.5 * dz0) / k3[0]                                    # (nr, nth)
    if la.probe_w <= 0:
        pw = np.zeros((nr, nth))
        pw[0, :] = 1.0
    else:
        pw = gaussian_annulus_weights(grid.r_faces, la.probe_w, cfg.geometry.R_cu)[:, None] * dth[None, :]
    pw = pw / pw.sum()

    n_steps = len(times) - 1
    pulse = la.f(times)
    f_mean = la.f_mean(times[:-1], times[1:])
    T0 = cfg.numerics.T0
    T = np.zeros(N)                                              # temperature rise
    T_surface = np.empty((n_steps + 1, 2 * nr))                  # plane through the void: x<0 (theta=pi) | x>0 (theta=0)
    T_probe = np.empty(n_steps + 1)
    T_axis = np.empty((n_steps + 1, nz))
    E_in = np.zeros(n_steps + 1)
    E_stored = np.zeros(n_steps + 1)
    E_out = np.zeros(n_steps + 1)
    snap_idx = set(np.unique(np.linspace(0, n_steps, cfg.numerics.n_snapshots).round().astype(int))) if store_fields else set()
    snapshots = []

    def plane(T3):
        """(nz, 2 nr) cut through the theta = 0 half-plane: left half theta = pi, right half theta = 0."""
        return np.concatenate([T3[:, ::-1, m_pi], T3[:, :, 0]], axis=1)

    def record(n):
        T3 = T.reshape(nz, nr, nth)
        Ts = T3[0] + face_corr * pulse[n]                        # (nr, nth) surface rise
        T_surface[n] = np.concatenate([Ts[::-1, m_pi], Ts[:, 0]]) + T0
        T_probe[n] = float((pw * Ts).sum()) + T0
        T_axis[n] = T3[:, 0, 0] + T0
        E_stored[n] = sym * float(C @ T)
        if n in snap_idx:
            snapshots.append((times[n], plane(T3) + T0))

    record(0)
    direct = N <= DIRECT_MAX_UNKNOWNS
    thetas = cfg.theta_schedule(times)
    solver, B, key_cur, n_factor = None, None, None, 0
    kinds, iters = set(), []                                     # solver kinds used and CG/BiCGSTAB iteration counts
    report_every = max(1, n_steps // 100)
    for n in range(n_steps):
        dt, theta = dts[n], thetas[n]
        if key_cur is None or abs(dt - key_cur[0]) > 1e-12 * key_cur[0] or theta != key_cur[1]:
            A = (Cdiag * (1.0 / dt) + theta * Lap).tocsc()
            B = (Cdiag * (1.0 / dt) - (1.0 - theta) * Lap).tocsr()
            if solver is not None:
                kinds.add(solver.kind)
                iters.extend(solver.n_iter)
                solver = None                                    # free the old factors before building the new ones
            solver = _Solver(A, direct, nth=nth, th_faces=th_f, periodic=periodic)
            key_cur = (dt, theta)
            n_factor += 1
        rhs = B @ T + src * f_mean[n]
        T_old = T
        T = solver.solve(rhs, T)
        E_in[n + 1] = E_in[n] + sym * f_mean[n] * src_total * dt
        E_out[n + 1] = E_out[n] + sym * dt * float(g_sink @ (theta * T + (1.0 - theta) * T_old))
        record(n + 1)
        if progress is not None and (n % report_every == 0 or n == n_steps - 1):
            progress((n + 1) / n_steps)

    diag = cfg.diagnostics()
    if solver is not None:
        kinds.add(solver.kind)
        iters.extend(solver.n_iter)
    diag.update(
        n_cells=N, nr=nr, nz=nz, n_theta=nth, dtheta_min=float(dth.min()), dtheta_max=float(dth.max()),
        absorbed_energy=float(E_in[-1]), n_factorisations=n_factor,
        absorbed_fraction_in_rod=float(W_ann.sum() / la.beam_area),
        solver=" / ".join(sorted(kinds)) if kinds else SOLVER_DIRECT, full_circle=periodic,
        mean_iterations=float(np.mean(iters)) if iters else 0.0,
    )
    mat_plane = np.concatenate([mat[:, ::-1, m_pi], mat[:, :, 0]], axis=1)
    return SimResult(
        config=cfg, grid=grid, material=mat_plane, times=times, T_surface=T_surface, T_probe=T_probe,
        T_axis=T_axis, snapshots=snapshots, E_in=E_in, E_stored=E_stored, pulse=pulse,
        wall_time=time.perf_counter() - t_start, diagnostics=diag, is_3d=True, n_theta=nth, E_out=E_out,
    )
