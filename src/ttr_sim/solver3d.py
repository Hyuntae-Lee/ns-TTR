"""Three-dimensional (r, theta, z) finite-volume solver for an off-axis void.

The copper rod, the silica shell and the laser source are axisymmetric; only the void breaks the
symmetry.  With the void centred on the theta = 0 half-plane the problem is mirror-symmetric in
theta, so only the half cylinder 0 <= theta <= pi is discretised (zero-flux faces at theta = 0 and
theta = pi).  The (r, z) grid is the same one the axisymmetric solver uses, so an axisymmetric
field on this grid reproduces the 2-D result exactly: the no-void baseline can stay 2-D and the
void signal (3-D void run minus 2-D baseline) is consistent.

Azimuthal grid: automatic.  The angular width of the void seen from the axis is resolved by at
least `void_cells` cells (uniform zone around theta = 0), then the spacing grows geometrically up to
`theta_max` towards theta = pi.  Non-uniform theta spacings are handled exactly by the finite-volume
link areas and centre distances.

Cells: p = (j * nr + i) * nth + m  with i radial, j axial, m azimuthal.  The innermost cells are
wedges that touch the axis; they need no special treatment because their radial inner face has
zero area and their azimuthal faces are handled like any other.

Linear solves: SuperLU when the system is small, otherwise ILU-preconditioned BiCGSTAB (the matrix
C/dt + L/2 is SPD and well conditioned during the pulse; the preconditioner is rebuilt whenever
dt changes).  All snapshots are stored as the (x, z) plane through the void (theta = pi | theta = 0),
which is what the GUI displays.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .solver import (
    MAT_COPPER, MAT_SILICA, MAT_VOID, SimConfig, SimResult, _transition_faces, build_grid,
    gaussian_annulus_weights,
)

DIRECT_MAX_UNKNOWNS = 60_000


def needs_3d(cfg: SimConfig) -> bool:
    """3-D is required only for a void displaced from the axis."""
    return cfg.void.enabled and cfg.void.r_center > 0.0


def void_angular_halfwidth(cfg: SimConfig) -> float:
    """Half of the angle subtended by the void about the axis (pi/2 if it contains the axis)."""
    v = cfg.void
    if v.r_center <= v.r_half:
        return math.pi / 2
    return math.asin(v.r_half / v.r_center)


def theta_faces(cfg: SimConfig) -> np.ndarray:
    """Azimuthal faces on [0, pi]: uniform fine zone over the void's angular extent (>= void_cells cells
    across the full width, plus a margin), then geometric growth up to theta_max."""
    n = cfg.numerics
    half = void_angular_halfwidth(cfg)
    d_fine = min(2.0 * half / n.void_cells, n.theta_max)
    margin = min(half, 5.0 * d_fine)
    th_zone = min(math.pi, half + margin)
    n_zone = max(1, int(round(th_zone / d_fine)))
    faces = np.linspace(0.0, th_zone, n_zone + 1)
    if th_zone < math.pi - 1e-12:
        faces = np.concatenate([faces, _transition_faces(th_zone, math.pi, d_fine, n.theta_max, n.stretch)])
    faces[-1] = math.pi
    return faces


def material_map_3d(cfg: SimConfig, grid, th_faces: np.ndarray) -> np.ndarray:
    """Material id per cell, shape (nz, nr, nth), for the half cylinder 0 <= theta <= pi."""
    th_c = 0.5 * (th_faces[1:] + th_faces[:-1])
    nth = len(th_c)
    rc, zc = grid.r_c, grid.z_c
    mat = np.full((grid.nz, grid.nr, nth), MAT_COPPER, dtype=np.int8)
    if not cfg.geometry.homogeneous_copper:
        mat[:, rc >= cfg.geometry.R_cu, :] = MAT_SILICA
    v = cfg.void
    if v.enabled:
        X = rc[None, :, None] * np.cos(th_c)[None, None, :]          # (1, nr, nth)
        Y = rc[None, :, None] * np.sin(th_c)[None, None, :]
        Z = zc[:, None, None]
        rho2 = (X - v.r_center) ** 2 + Y ** 2                        # squared distance from the void axis
        if v.shape == "ellipse":
            inside = rho2 / v.r_half ** 2 + ((Z - v.z_center) / (0.5 * v.thickness)) ** 2 < 1.0
        else:
            inside = (rho2 < v.r_half ** 2) & (Z >= v.depth) & (Z < v.z_bottom)
        inside = inside & (rc[None, :, None] < cfg.geometry.R_cu)
        mat[np.broadcast_to(inside, mat.shape)] = MAT_VOID
    return mat


def _properties(cfg: SimConfig, mat: np.ndarray):
    k = np.empty(mat.shape)
    rho_cp = np.empty(mat.shape)
    for mid, (kk, rc) in {
        MAT_COPPER: (cfg.copper.k, cfg.copper.rho_cp),
        MAT_SILICA: (cfg.silica.k, cfg.silica.rho_cp),
        MAT_VOID: (cfg.void.k, cfg.void.rho_cp),
    }.items():
        sel = mat == mid
        k[sel] = kk
        rho_cp[sel] = rc
    return k, rho_cp


def assemble_laplacian_3d(grid, k3: np.ndarray, th_faces: np.ndarray) -> sp.csr_matrix:
    """Conductance Laplacian (W/K) on the half cylinder; zero-flux at theta = 0 and theta = pi."""
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
    G_z = (A_z / R_z).ravel()
    p_z, q_z = idx[:-1].ravel(), idx[1:].ravel()

    # azimuthal links (m, m+1): face area dr*dz; centre-to-face distances r_c*dtheta_m/2 and r_c*dtheta_{m+1}/2
    A_t = drc[None, :, None] * dzc[:, None, None]
    R_t = (0.5 * rc)[None, :, None] * (dth[:-1][None, None, :] / k3[:, :, :-1] + dth[1:][None, None, :] / k3[:, :, 1:])
    G_t = (A_t / R_t).ravel()
    p_t, q_t = idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel()

    p = np.concatenate([p_r, p_z, p_t])
    q = np.concatenate([q_r, q_z, q_t])
    G = np.concatenate([G_r, G_z, G_t])
    N = nz * nr * nth
    rows = np.concatenate([p, q, p, q])
    cols = np.concatenate([p, q, q, p])
    data = np.concatenate([G, G, -G, -G])
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


class _Solver:
    """Direct (SuperLU) or ILU-preconditioned BiCGSTAB solve of A x = b, rebuilt per dt."""

    def __init__(self, A: sp.csc_matrix, direct: bool):
        self.direct = direct
        self.n_iter = []
        if direct:
            self.lu = spla.splu(A)
        else:
            self.A = A
            ilu = spla.spilu(A, drop_tol=1e-5, fill_factor=12)
            self.M = spla.LinearOperator(A.shape, ilu.solve)

    def solve(self, b: np.ndarray, x0: np.ndarray) -> np.ndarray:
        if self.direct:
            return self.lu.solve(b)
        cnt = [0]

        def cb(_):
            cnt[0] += 1

        scale = float(np.max(np.abs(b))) or 1.0
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
    mat = material_map_3d(cfg, grid, th_f)
    k3, rho_cp = _properties(cfg, mat)
    nr, nz = grid.nr, grid.nz
    N = nz * nr * nth
    if N > cfg.numerics.max_cells * 4:
        raise ValueError(f"3D 미지수 {N:,} 가 한도({cfg.numerics.max_cells * 4:,})를 넘습니다. void 크기/깊이 또는 프리셋을 조정하세요.")

    A_z_cell = 0.5 * (grid.r_faces[1:] ** 2 - grid.r_faces[:-1] ** 2)[None, :, None] * dth[None, None, :]   # (1, nr, nth) top-face area
    V = A_z_cell * grid.dz_c[:, None, None]
    C = (rho_cp * V).ravel()                                     # J/K per cell (half cylinder)
    Lap = assemble_laplacian_3d(grid, k3, th_f)
    Cdiag = sp.diags(C)
    times = cfg.time_grid()
    dts = np.diff(times)

    # laser source on the top faces of copper cells: axisymmetric Gaussian, per-sector fraction dtheta/(2 pi)
    la = cfg.laser
    q_abs = la.I0 * (1.0 - la.reflectivity)
    W_ann = gaussian_annulus_weights(grid.r_faces, la.w, cfg.geometry.R_cu)                       # (nr,)
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
    snap_idx = set(np.unique(np.linspace(0, n_steps, cfg.numerics.n_snapshots).round().astype(int))) if store_fields else set()
    snapshots = []

    def plane(T3):
        """(nz, 2 nr) cut through the void: left half theta = pi, right half theta = 0."""
        return np.concatenate([T3[:, ::-1, -1], T3[:, :, 0]], axis=1)

    def record(n):
        T3 = T.reshape(nz, nr, nth)
        Ts = T3[0] + face_corr * pulse[n]                        # (nr, nth) surface rise
        T_surface[n] = np.concatenate([Ts[::-1, -1], Ts[:, 0]]) + T0
        T_probe[n] = float((pw * Ts).sum()) + T0
        T_axis[n] = T3[:, 0, 0] + T0
        E_stored[n] = 2.0 * float(C @ T)                        # both halves
        if n in snap_idx:
            snapshots.append((times[n], plane(T3) + T0))

    record(0)
    direct = N <= DIRECT_MAX_UNKNOWNS
    solver, B, dt_cur, n_factor = None, None, None, 0
    report_every = max(1, n_steps // 100)
    for n in range(n_steps):
        dt = dts[n]
        if dt_cur is None or abs(dt - dt_cur) > 1e-12 * dt_cur:
            A = (Cdiag * (1.0 / dt) + 0.5 * Lap).tocsc()
            B = (Cdiag * (1.0 / dt) - 0.5 * Lap).tocsr()
            solver = _Solver(A, direct)
            dt_cur = dt
            n_factor += 1
        rhs = B @ T + src * f_mean[n]
        T = solver.solve(rhs, T)
        E_in[n + 1] = E_in[n] + 2.0 * f_mean[n] * src_total * dt
        record(n + 1)
        if progress is not None and (n % report_every == 0 or n == n_steps - 1):
            progress((n + 1) / n_steps)

    diag = cfg.diagnostics()
    iters = solver.n_iter if solver is not None else []
    diag.update(
        n_cells=N, nr=nr, nz=nz, n_theta=nth, dtheta_min=float(dth.min()), dtheta_max=float(dth.max()),
        absorbed_energy=float(E_in[-1]), n_factorisations=n_factor,
        absorbed_fraction_in_rod=float(W_ann.sum() / (0.5 * math.pi * la.w ** 2)),
        solver="직접 LU (SuperLU)" if direct else "ILU + BiCGSTAB",
        mean_iterations=float(np.mean(iters)) if iters else 0.0,
    )
    mat_plane = np.concatenate([mat[:, ::-1, -1], mat[:, :, 0]], axis=1)
    return SimResult(
        config=cfg, grid=grid, material=mat_plane, times=times, T_surface=T_surface, T_probe=T_probe,
        T_axis=T_axis, snapshots=snapshots, E_in=E_in, E_stored=E_stored, pulse=pulse,
        wall_time=time.perf_counter() - t_start, diagnostics=diag, is_3d=True, n_theta=nth,
    )
