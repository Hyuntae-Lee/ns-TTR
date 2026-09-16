"""Three-dimensional (r, theta, z) finite-volume solver for an off-axis void.

The copper rod, the silica shell and the laser source are axisymmetric; only the void breaks the
symmetry.  With the void centred on the theta = 0 half-plane the problem is mirror-symmetric in
theta, so only the half cylinder 0 <= theta <= pi is discretised (zero-flux faces at theta = 0 and
theta = pi).  The (r, z) grid is the same one the axisymmetric solver uses, so an axisymmetric
field on this grid reproduces the 2-D result exactly: the no-void baseline can stay 2-D and the
void signal (3-D void run minus 2-D baseline) is consistent.

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
    MAT_COPPER, MAT_SILICA, MAT_VOID, SimConfig, SimResult, build_grid, gaussian_annulus_weights,
)

DIRECT_MAX_UNKNOWNS = 60_000


def needs_3d(cfg: SimConfig) -> bool:
    """3-D is required only for a void displaced from the axis."""
    return cfg.void.enabled and cfg.void.r_center > 0.0


def material_map_3d(cfg: SimConfig, grid, nth: int) -> np.ndarray:
    """Material id per cell, shape (nz, nr, nth), for the half cylinder 0 <= theta <= pi."""
    th_c = (np.arange(nth) + 0.5) * math.pi / nth
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
            # spheroid: horizontal semi-axes r_half, vertical semi-axis thickness/2
            inside = rho2 / v.r_half ** 2 + ((Z - v.z_center) / (0.5 * v.thickness)) ** 2 < 1.0
        else:
            # short cylinder (disk) of radius r_half and height thickness
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


def assemble_laplacian_3d(grid, k3: np.ndarray, nth: int) -> sp.csr_matrix:
    """Conductance Laplacian (W/K) on the half cylinder; zero-flux at theta = 0 and theta = pi."""
    nr, nz = grid.nr, grid.nz
    rf, zf, rc, zc = grid.r_faces, grid.z_faces, grid.r_c, grid.z_c
    dzc, drc = grid.dz_c, grid.dr_c
    dth = math.pi / nth
    idx = np.arange(nz * nr * nth).reshape(nz, nr, nth)

    # radial links (i, i+1)
    A_r = (rf[1:-1] * dth)[None, :, None] * dzc[:, None, None]
    R_r = (rf[1:-1] - rc[:-1])[None, :, None] / k3[:, :-1, :] + (rc[1:] - rf[1:-1])[None, :, None] / k3[:, 1:, :]
    G_r = (A_r / R_r).ravel()
    p_r, q_r = idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel()

    # axial links (j, j+1)
    A_z = (0.5 * dth * (rf[1:] ** 2 - rf[:-1] ** 2))[None, :, None]
    R_z = (zf[1:-1] - zc[:-1])[:, None, None] / k3[:-1] + (zc[1:] - zf[1:-1])[:, None, None] / k3[1:]
    G_z = (A_z / R_z).ravel()
    p_z, q_z = idx[:-1].ravel(), idx[1:].ravel()

    # azimuthal links (m, m+1): face area dr*dz, centre distance r_c*dth (half in each cell)
    A_t = drc[None, :, None] * dzc[:, None, None]
    R_t = (0.5 * rc * dth)[None, :, None] * (1.0 / k3[:, :, :-1] + 1.0 / k3[:, :, 1:])
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
    nth = max(2, int(cfg.numerics.n_theta))
    grid = build_grid(cfg)
    mat = material_map_3d(cfg, grid, nth)
    k3, rho_cp = _properties(cfg, mat)
    nr, nz = grid.nr, grid.nz
    N = nz * nr * nth
    if N > cfg.numerics.max_cells * 4:
        raise ValueError(f"3D 미지수 {N:,} 가 한도({cfg.numerics.max_cells * 4:,})를 넘습니다. Δz, nθ 또는 void 크기를 조정하세요.")

    dth = math.pi / nth
    V = (0.5 * dth * (grid.r_faces[1:] ** 2 - grid.r_faces[:-1] ** 2))[None, :, None] * grid.dz_c[:, None, None]
    C = (rho_cp * V).ravel()                                     # J/K per cell (half cylinder)
    Lap = assemble_laplacian_3d(grid, k3, nth)
    Cdiag = sp.diags(C)
    times = cfg.time_grid()
    dts = np.diff(times)

    # laser source on the top faces of copper cells: axisymmetric Gaussian, per-sector fraction dth/(2 pi)
    la = cfg.laser
    q_abs = la.I0 * (1.0 - la.reflectivity)
    Wsrc = gaussian_annulus_weights(grid.r_faces, la.w, cfg.geometry.R_cu) * dth / (2.0 * math.pi)   # (nr,)
    src3 = np.zeros((nz, nr, nth))
    src3[0, :, :] = q_abs * Wsrc[:, None]
    src3[0][mat[0] != MAT_COPPER] = 0.0
    src = src3.ravel()
    src_total = src.sum()

    # surface reconstruction and probe weights (same definitions as the 2-D solver)
    dz0 = grid.dz_c[0]
    face_corr = (src3[0] / (0.5 * dth * (grid.r_faces[1:] ** 2 - grid.r_faces[:-1] ** 2))[:, None]) * (0.5 * dz0) / k3[0]   # (nr, nth)
    if la.probe_w <= 0:
        pw = np.zeros((nr, nth))
        pw[0, :] = 1.0
    else:
        pw = np.broadcast_to(gaussian_annulus_weights(grid.r_faces, la.probe_w, cfg.geometry.R_cu)[:, None], (nr, nth)).copy()
    pw /= pw.sum()

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
        n_cells=N, nr=nr, nz=nz, n_theta=nth, absorbed_energy=float(E_in[-1]), n_factorisations=n_factor,
        absorbed_fraction_in_rod=float(gaussian_annulus_weights(grid.r_faces, la.w, cfg.geometry.R_cu).sum() / (0.5 * math.pi * la.w ** 2)),
        solver="직접 LU (SuperLU)" if direct else "ILU + BiCGSTAB",
        mean_iterations=float(np.mean(iters)) if iters else 0.0,
    )
    # material plane for the display (same layout as the snapshots)
    mat_plane = np.concatenate([mat[:, ::-1, -1], mat[:, :, 0]], axis=1)
    return SimResult(
        config=cfg, grid=grid, material=mat_plane, times=times, T_surface=T_surface, T_probe=T_probe,
        T_axis=T_axis, snapshots=snapshots, E_in=E_in, E_stored=E_stored, pulse=pulse,
        wall_time=time.perf_counter() - t_start, diagnostics=diag, is_3d=True, n_theta=nth,
    )
