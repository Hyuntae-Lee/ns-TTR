"""Axisymmetric (r, z) finite-volume heat diffusion solver with Crank-Nicolson time stepping.

Discretisation
--------------
* Cell-centred finite volumes on a structured, non-uniform (r, z) grid.  The grid is
  uniform (dr, dz from the depth preset) in the region of interest near the surface / void
  and geometrically stretched further away so the full 500 um rod and the surrounding
  silica are included at low cost.
* Face conductances use series resistances (harmonic averaging), which gives exact flux and
  temperature continuity across copper/silica/void interfaces without any division by k of
  the void material.  A k_void of 0 would simply give a zero conductance, so the k_void
  "floor" is purely a modelling choice here, not a numerical necessity.
* Laser heating enters as a surface flux on the z = 0 faces of copper cells
  (-k dT/dz = I0 (1-R) f(t) g(r)), integrated exactly over each annular face.
* All other boundaries are adiabatic (axis by symmetry, back face z = L, outer silica).
  Therefore the discrete scheme is exactly conservative and the energy balance is an
  end-to-end check of assembly + linear solve.
* The linear system is factorised once (SuperLU) and reused for every step.

The material map is produced from a function of cell-centre coordinates, and the assembly
works from generic link lists, so a 3-D Cartesian variant can reuse the same structure.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import Material, COPPER, FUSED_SILICA, AIR, COPPER_ABSORPTION_DEPTH

MAT_COPPER, MAT_SILICA, MAT_VOID = 0, 1, 2
MAT_NAMES = {MAT_COPPER: "Copper", MAT_SILICA: "Fused silica", MAT_VOID: "Void"}


# ----------------------------------------------------------------------------- configuration
@dataclass
class Geometry:
    R_cu: float = 40e-6          # copper cylinder radius, m
    L: float = 500e-6            # copper cylinder length, m
    homogeneous_copper: bool = False   # validation mode: replace silica by copper (semi-infinite analogue)


@dataclass
class VoidSpec:
    enabled: bool = True
    depth: float = 10e-6         # z of void top face, m
    thickness: float = 5e-6      # axial extent, m
    r_center: float = 0.0        # radial centre (0 = on-axis disk, >0 = ring void), m
    r_half: float = 10e-6        # radial half-extent, m
    k: float = 1.0               # "floor" conductivity used for the void, W/(m K)
    rho_cp: float = AIR.rho_cp   # volumetric heat capacity of the void filling
    shape: str = "ellipse"       # "ellipse": elliptical (r, z) cross-section inscribed in the box below; "box": rectangle

    @property
    def z_center(self) -> float:
        return self.depth + 0.5 * self.thickness

    @property
    def r_inner(self) -> float:
        return max(0.0, self.r_center - self.r_half)

    @property
    def r_outer(self) -> float:
        return self.r_center + self.r_half

    @property
    def z_bottom(self) -> float:
        return self.depth + self.thickness


@dataclass
class Laser:
    tau_p: float                 # pulse width (square: duration, gaussian: FWHM), s
    profile: str = "square"      # "square" | "gaussian"
    energy: float = 1e-9         # incident pulse energy, J
    w: float = 10e-6             # 1/e^2 spot radius, m
    reflectivity: float = 0.6
    probe_w: float = 5e-6        # probe 1/e^2 radius for the thermoreflectance signal (<=0: centre cell), m

    @property
    def I0(self) -> float:
        """Peak incident intensity (W/m^2); defined so that the incident fluence integrates to `energy`."""
        return 2.0 * self.energy / (math.pi * self.w ** 2 * self.tau_p)

    @property
    def t_center(self) -> float:
        """Centre of the Gaussian pulse (0 for square)."""
        return 1.5 * self.tau_p if self.profile == "gaussian" else 0.0

    @property
    def sigma(self) -> float:
        return self.tau_p / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def f(self, t):
        """Temporal profile normalised so that the integral of f dt = tau_p (same fluence for both shapes)."""
        t = np.asarray(t, dtype=float)
        if self.profile == "square":
            return ((t >= 0.0) & (t < self.tau_p)).astype(float)
        s = self.sigma
        return self.tau_p / (s * math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * ((t - self.t_center) / s) ** 2)

    def f_mean(self, t0, t1):
        """Exact average of f over [t0, t1] (the Crank-Nicolson source uses the step-averaged flux,
        so a square pulse edge that falls inside a step deposits exactly the right energy)."""
        t0 = np.asarray(t0, dtype=float)
        t1 = np.asarray(t1, dtype=float)
        if self.profile == "square":
            overlap = np.clip(np.minimum(t1, self.tau_p) - np.maximum(t0, 0.0), 0.0, None)
            return overlap / (t1 - t0)
        from scipy.special import erf
        s = self.sigma
        F = lambda t: 0.5 * self.tau_p * erf((t - self.t_center) / (s * math.sqrt(2.0)))  # noqa: E731
        return (F(t1) - F(t0)) / (t1 - t0)

    def fprime(self, t):
        """d f / dt (Gaussian only; used by the analytic convolution)."""
        t = np.asarray(t, dtype=float)
        s = self.sigma
        return -(t - self.t_center) / s ** 2 * self.f(t)

    @property
    def pulse_end(self) -> float:
        """Time after which essentially no more energy is deposited."""
        return self.tau_p if self.profile == "square" else self.t_center + 3.0 * self.sigma


@dataclass
class Numerics:
    dz: float                    # axial spacing in the fine regions (surface layer, void zone), m
    dr: Optional[float] = None   # radial spacing in the fine region, m (None -> automatic)
    fo: float = 0.5              # Fourier number D_cu*dt/dz^2 used to pick the initial dt
    t_end_factor: float = 5.0    # observation window multiplier (see t_end_mode)
    t_end_mode: str = "tau_p"    # "tau_p": factor*tau_p | "void": factor*d_void^2/D | "absolute": t_end_abs
    t_end_abs: float = 1e-6      # absolute observation window, s (t_end_mode == "absolute")
    dt_growth: float = 1.05      # per-step growth of dt after the pulse (1.0 = constant dt); applied in blocks
    dt_block: int = 8            # steps per constant-dt block (one LU factorisation per block)
    stretch: float = 1.15        # geometric growth ratio of the grid outside the fine regions
    T0: float = 293.15           # initial / ambient temperature, K
    n_snapshots: int = 12        # number of stored full-field snapshots
    max_cells: int = 600_000
    max_steps: int = 400_000


@dataclass
class SimConfig:
    geometry: Geometry
    void: VoidSpec
    laser: Laser
    numerics: Numerics
    copper: Material = COPPER
    silica: Material = FUSED_SILICA

    # -- derived quantities used by both the solver and the GUI
    @property
    def dr(self) -> float:
        n, g, v = self.numerics, self.geometry, self.void
        if n.dr is not None:
            return n.dr
        dr = min(n.dz, self.laser.w / 5.0, g.R_cu / 10.0)
        if v.enabled and v.r_half > 0:
            dr = min(dr, max(v.r_half / 3.0, n.dz / 4.0))
        return dr

    @property
    def dt(self) -> float:
        """Initial (pulse-phase) time step."""
        return self.numerics.fo * self.numerics.dz ** 2 / self.copper.alpha

    @property
    def t_end(self) -> float:
        la, n = self.laser, self.numerics
        extra = la.t_center - 0.5 * la.tau_p if la.profile == "gaussian" else 0.0
        t_pulse_based = n.t_end_factor * la.tau_p + extra
        if n.t_end_mode == "void" and self.void.depth > 0:
            # uses the void depth even when the void is disabled, so the baseline run of a pair
            # covers exactly the same window as the void run
            tau_void = self.void.depth ** 2 / self.copper.alpha
            return max(n.t_end_factor * tau_void, 2.0 * la.tau_p + extra)
        if n.t_end_mode == "absolute":
            return max(float(n.t_end_abs), la.pulse_end * 1.05)
        return t_pulse_based

    def time_grid(self) -> np.ndarray:
        """Step boundaries t_0=0 < ... < t_N = t_end.

        Constant dt (= self.dt) while the laser is on; afterwards dt is multiplied by
        dt_growth**dt_block after every block of dt_block steps (Crank-Nicolson is unconditionally
        stable, and per-step growth g keeps dt ~ (g-1)*t, i.e. a fixed relative resolution in time).
        """
        n = self.numerics
        dt0, t_end, g, B = self.dt, self.t_end, max(1.0, float(n.dt_growth)), max(1, int(n.dt_block))
        pulse_end = self.laser.pulse_end
        times = [0.0]
        t, dt = 0.0, dt0
        while t < t_end:
            if t >= pulse_end and g > 1.0:
                for _ in range(B):
                    if t >= t_end:
                        break
                    step = min(dt, t_end - t)
                    t += step
                    times.append(t)
                dt *= g ** B
            else:
                step = min(dt, t_end - t)
                t += step
                times.append(t)
            if len(times) > n.max_steps:
                raise ValueError(
                    f"시간 스텝 수가 한도({n.max_steps:,})를 넘습니다. 관측 시간창을 줄이거나 Δt 성장률/격자 간격을 키우세요."
                )
        times = np.array(times)
        # avoid a vanishingly small final step: merge it into the previous one
        if len(times) > 2 and (times[-1] - times[-2]) < 1e-3 * (times[-2] - times[-3]):
            times = np.delete(times, -2)
        times[-1] = t_end
        return times

    @property
    def n_steps(self) -> int:
        return len(self.time_grid()) - 1

    @property
    def dt_max(self) -> float:
        return float(np.max(np.diff(self.time_grid())))

    def penetration_length(self, t: Optional[float] = None) -> float:
        """Thermal penetration length sqrt(D_cu * t)."""
        return math.sqrt(self.copper.alpha * (self.t_end if t is None else t))

    def diagnostics(self) -> dict:
        """Numerical/physical sanity indicators shown in the GUI."""
        g, la, n = self.geometry, self.laser, self.numerics
        L_diff = self.penetration_length()
        L_diff_si = math.sqrt(self.silica.alpha * self.t_end)
        flux_ratio = n.dz / COPPER_ABSORPTION_DEPTH
        warnings = []
        if L_diff > 0.8 * g.L:
            warnings.append(
                f"열 침투 깊이 sqrt(D*t_end) = {L_diff * 1e6:.0f} μm 가 로드 길이의 80%({0.8 * g.L * 1e6:.0f} μm)를 넘습니다. "
                "후면 단열 경계조건이 결과에 영향을 줍니다."
            )
        if flux_ratio < 7:
            warnings.append(
                f"Δz/δ_abs = {flux_ratio:.1f} < 7: 표면 flux 근사(Beer-Lambert 부피열원 → 표면 경계조건)의 유효 범위를 벗어납니다."
            )
        if self.void.enabled and self.void.z_bottom > g.L:
            warnings.append("void 하단이 로드 후면(z=L)을 넘습니다. void가 잘립니다.")
        if self.void.enabled and self.void.r_outer > g.R_cu:
            warnings.append("void 반경 범위가 구리 반경(40 μm)을 넘습니다. void가 구리 내부로 잘립니다.")
        if self.void.enabled and self.void.thickness < 2 * n.dz:
            warnings.append(f"void 두께({self.void.thickness * 1e6:.2f} μm)가 격자 Δz의 2배보다 작아 해상이 부족합니다.")
        interface_reached = L_diff > (g.R_cu - la.w)
        try:
            tg = self.time_grid()
            n_steps, dt_max = len(tg) - 1, float(np.max(np.diff(tg)))
        except ValueError as e:
            warnings.append(str(e))
            n_steps, dt_max = n.max_steps, float("nan")
        return dict(
            dz=n.dz, dr=self.dr, dt=self.dt, dt_max=dt_max, dt_growth=n.dt_growth, fo=n.fo, tau_p=la.tau_p,
            t_end=self.t_end, t_end_mode=n.t_end_mode,
            n_steps=n_steps, L_diff=L_diff, L_diff_silica=L_diff_si, flux_ratio=flux_ratio,
            interface_reached=interface_reached, warnings=warnings, I0=la.I0,
            q_abs_peak=la.I0 * (1 - la.reflectivity),
            fluence_peak=la.I0 * la.tau_p,
        )


# ----------------------------------------------------------------------------- grid
@dataclass
class Grid:
    r_faces: np.ndarray
    z_faces: np.ndarray

    @property
    def r_c(self):
        return 0.5 * (self.r_faces[1:] + self.r_faces[:-1])

    @property
    def z_c(self):
        return 0.5 * (self.z_faces[1:] + self.z_faces[:-1])

    @property
    def nr(self):
        return len(self.r_faces) - 1

    @property
    def nz(self):
        return len(self.z_faces) - 1

    @property
    def n_cells(self):
        return self.nr * self.nz

    @property
    def dz_c(self):
        return np.diff(self.z_faces)

    @property
    def dr_c(self):
        return np.diff(self.r_faces)

    @property
    def face_area_z(self):
        """Area of a z-face for each radial cell (annulus)."""
        rf = self.r_faces
        return math.pi * (rf[1:] ** 2 - rf[:-1] ** 2)

    @property
    def volume(self):
        return self.face_area_z[None, :] * self.dz_c[:, None]


def _uniform_faces(a: float, b: float, d: float) -> np.ndarray:
    """Faces in (a, b] with spacing ~ d (b included, a excluded)."""
    if b - a <= 1e-15:
        return np.empty(0)
    n = max(1, int(round((b - a) / d)))
    return np.linspace(a, b, n + 1)[1:]


def _stretched_faces(a: float, b: float, d0: float, ratio: float) -> np.ndarray:
    """Geometrically growing spacings from a to b, first spacing ~ d0 (b included, a excluded)."""
    Ltot = b - a
    if Ltot <= 1e-15:
        return np.empty(0)
    if Ltot <= 1.5 * d0:
        return np.array([b])
    n = max(1, int(math.ceil(math.log(1.0 + Ltot * (ratio - 1.0) / d0) / math.log(ratio))))
    spac = d0 * ratio ** np.arange(n)
    spac *= Ltot / spac.sum()
    faces = a + np.cumsum(spac)
    faces[-1] = b
    return faces


def _transition_faces(a: float, b: float, d0: float, d1: float, ratio: float) -> np.ndarray:
    """Spacings growing geometrically from d0 to d1, then uniform d1, fitted to [a, b] (b included)."""
    Ltot = b - a
    if Ltot <= 1e-15:
        return np.empty(0)
    if d0 >= d1:
        return _uniform_faces(a, b, d1)
    spac, s = [], d0
    while sum(spac) < Ltot:
        spac.append(min(s, d1))
        s *= ratio
    spac = np.array(spac)
    spac *= Ltot / spac.sum()
    faces = a + np.cumsum(spac)
    faces[-1] = b
    return faces


def _segmented(start: float, breakpoints, end: float, d: float, d_start: Optional[float] = None,
               ratio: float = 1.15) -> np.ndarray:
    """Uniform faces from start to end with faces forced onto the (sorted) breakpoints.

    If `d_start` < d is given, the first segment uses spacing d_start and the second segment
    grows geometrically from d_start to d (surface refinement for a laser spot finer than d).
    """
    pts = [start] + sorted(p for p in breakpoints if start + 0.25 * d < p < end - 0.25 * d) + [end]
    faces = [np.array([start])]
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        if d_start is not None and d_start < d and i == 0:
            faces.append(_uniform_faces(a, b, d_start))
        elif d_start is not None and d_start < d and i == 1:
            faces.append(_transition_faces(a, b, d_start, d, ratio))
        else:
            faces.append(_uniform_faces(a, b, d))
    return np.concatenate(faces)


def _graded_faces(a: float, b: float, d_start: float, d_end: float, ratio: float) -> np.ndarray:
    """Faces from a to b whose spacing grows geometrically away from both ends (d_start at a, d_end at b)
    and meets in the middle: a coarse 'bridge' between two fine zones (b included, a excluded)."""
    Ltot = b - a
    if Ltot <= 1e-15:
        return np.empty(0)
    left, right, sl, sr = [], [], d_start, d_end
    while sum(left) + sum(right) < Ltot:
        if sl <= sr:
            left.append(sl)
            sl *= ratio
        else:
            right.append(sr)
            sr *= ratio
    spac = np.array(left + right[::-1])
    spac *= Ltot / spac.sum()
    faces = a + np.cumsum(spac)
    faces[-1] = b
    return faces


def build_grid(cfg: SimConfig) -> Grid:
    """Axial grid: a uniform surface layer (dz) resolving the pulse deposition, a uniform zone around the
    void (dz), a geometrically graded bridge between them and a stretched tail down to z = L.  The
    temperature field smooths as sqrt(D t) away from the surface, so cells growing roughly in
    proportion to depth lose no accuracy while keeping the cell count small even for a short pulse
    (fine dz) combined with a deep void."""
    g, v, la, n = cfg.geometry, cfg.void, cfg.laser, cfg.numerics
    dz, dr = n.dz, cfg.dr

    # ---- surface layer: >= 10 pulse penetration lengths, >= 5 cells, plus the 2w spot layer if refined
    dz_surf = dr if dr < dz else None           # radial spacing finer than dz -> refine the first ~2w in z too
    z_s = max(10.0 * math.sqrt(cfg.copper.alpha * la.tau_p), 5.0 * dz)
    if dz_surf is not None:
        z_s = max(z_s, 2.0 * la.w)
    z_s = min(z_s, g.L)
    bps = [min(2.0 * la.w, z_s)] if dz_surf is not None else []

    # ---- void zone: uniform dz from (depth - margin) to (bottom + margin), faces on the void boundaries
    if v.enabled and v.depth < g.L:
        margin = min(v.thickness, 5.0 * dz)
        zv0, zv1 = max(0.0, v.depth - margin), min(g.L, v.z_bottom + margin)
    else:
        zv0, zv1 = None, None

    if zv0 is not None and zv0 <= z_s + 2.0 * dz:
        # shallow void: one uniform region from the surface to below the void
        z_uni = max(z_s, zv1)
        if g.L - z_uni < 2 * dz:
            z_uni = g.L
        z_faces = _segmented(0.0, bps + [v.depth, v.z_bottom], z_uni, dz, d_start=dz_surf, ratio=n.stretch)
    else:
        if zv0 is None and g.L - z_s < 2 * dz:
            z_s = g.L
        z_faces = _segmented(0.0, bps, z_s, dz, d_start=dz_surf, ratio=n.stretch)
        if zv0 is not None:
            # graded bridge, then the uniform void zone
            z_faces = np.concatenate([z_faces, _graded_faces(z_s, zv0, dz, dz, n.stretch)])
            z_faces = np.concatenate([z_faces, _segmented(zv0, [v.depth, v.z_bottom], zv1, dz)[1:]])
            z_uni = zv1
        else:
            z_uni = z_s
    if z_uni < g.L:
        z_faces = np.concatenate([z_faces, _stretched_faces(z_uni, g.L, dz, n.stretch)])

    # ---- radial: fine uniform region (spot + void), stretched to R_cu, then silica shell
    r_fine = 2.0 * la.w
    if v.enabled and v.r_outer < g.R_cu * (1 - 1e-9):
        # resolve the void's radial edge; a void reaching the copper boundary needs no extra radial refinement
        r_fine = max(r_fine, v.r_outer + 5.0 * dr)
    r_fine = min(r_fine, g.R_cu)
    if g.R_cu - r_fine < 2 * dr:
        r_fine = g.R_cu
    bps = [v.r_inner, v.r_outer] if v.enabled else []
    r_faces = _segmented(0.0, bps, r_fine, dr)
    if r_fine < g.R_cu:
        r_faces = np.concatenate([r_faces, _stretched_faces(r_fine, g.R_cu, dr, n.stretch)])
    outer = cfg.copper if g.homogeneous_copper else cfg.silica
    L_out = math.sqrt(outer.alpha * cfg.t_end)
    r_max = g.R_cu + max(2.0 * g.R_cu, 4.0 * L_out)
    d_last = r_faces[-1] - r_faces[-2]
    r_faces = np.concatenate([r_faces, _stretched_faces(g.R_cu, r_max, d_last, n.stretch)])

    grid = Grid(r_faces=r_faces, z_faces=z_faces)
    if grid.n_cells > n.max_cells:
        raise ValueError(
            f"격자 셀 수 {grid.n_cells:,} 가 한도({n.max_cells:,})를 넘습니다. 프리셋/void 크기를 조정하세요."
        )
    return grid


def material_map(cfg: SimConfig, grid: Grid) -> np.ndarray:
    """Integer material id per cell, shape (nz, nr)."""
    rc, zc = grid.r_c[None, :], grid.z_c[:, None]
    mat = np.full((grid.nz, grid.nr), MAT_COPPER, dtype=np.int8)
    if not cfg.geometry.homogeneous_copper:
        mat[np.broadcast_to(rc >= cfg.geometry.R_cu, mat.shape)] = MAT_SILICA
    v = cfg.void
    if v.enabled:
        in_box = (zc >= v.depth) & (zc < v.z_bottom) & (rc >= v.r_inner) & (rc < v.r_outer)
        if v.shape == "ellipse":
            # Elliptical cross-section in the (r, z) plane: an oblate/prolate spheroid when on the axis,
            # a torus with elliptical section when r_center > 0.  Semi-axes: r_half and thickness/2.
            ell = ((rc - v.r_center) / v.r_half) ** 2 + ((zc - v.z_center) / (0.5 * v.thickness)) ** 2 < 1.0
            inside = in_box & ell
        else:
            inside = in_box
        inside = inside & (rc < cfg.geometry.R_cu)
        mat[np.broadcast_to(inside, mat.shape)] = MAT_VOID
    return mat


def _property_arrays(cfg: SimConfig, mat: np.ndarray):
    k = np.empty(mat.shape)
    rho_cp = np.empty(mat.shape)
    table = {
        MAT_COPPER: (cfg.copper.k, cfg.copper.rho_cp),
        MAT_SILICA: (cfg.silica.k, cfg.silica.rho_cp),
        MAT_VOID: (cfg.void.k, cfg.void.rho_cp),
    }
    for mid, (kk, rc) in table.items():
        sel = mat == mid
        k[sel] = kk
        rho_cp[sel] = rc
    return k, rho_cp


def assemble_laplacian(grid: Grid, k: np.ndarray) -> sp.csr_matrix:
    """Conductance Laplacian L (W/K) such that heat flow out of cell p = (L T)_p."""
    nr, nz = grid.nr, grid.nz
    rf, zf, rc, zc = grid.r_faces, grid.z_faces, grid.r_c, grid.z_c
    dzc = grid.dz_c
    idx = np.arange(nr * nz).reshape(nz, nr)

    # radial links (i, i+1)
    A_r = 2.0 * math.pi * rf[1:-1][None, :] * dzc[:, None]
    R_r = (rf[1:-1] - rc[:-1])[None, :] / k[:, :-1] + (rc[1:] - rf[1:-1])[None, :] / k[:, 1:]
    G_r = A_r / R_r
    p_r, q_r = idx[:, :-1].ravel(), idx[:, 1:].ravel()

    # axial links (j, j+1)
    A_z = np.broadcast_to(grid.face_area_z[None, :], (nz - 1, nr))
    R_z = (zf[1:-1] - zc[:-1])[:, None] / k[:-1, :] + (zc[1:] - zf[1:-1])[:, None] / k[1:, :]
    G_z = A_z / R_z
    p_z, q_z = idx[:-1, :].ravel(), idx[1:, :].ravel()

    p = np.concatenate([p_r, p_z])
    q = np.concatenate([q_r, q_z])
    G = np.concatenate([G_r.ravel(), G_z.ravel()])
    N = nr * nz
    rows = np.concatenate([p, q, p, q])
    cols = np.concatenate([p, q, q, p])
    data = np.concatenate([G, G, -G, -G])
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


def gaussian_annulus_weights(r_faces: np.ndarray, w: float, r_cut: float) -> np.ndarray:
    """Integral of exp(-2 r^2/w^2) 2 pi r dr over each annular face, clipped at r_cut (m^2)."""
    ra = np.minimum(r_faces[:-1], r_cut)
    rb = np.minimum(r_faces[1:], r_cut)
    return 0.5 * math.pi * w ** 2 * (np.exp(-2.0 * ra ** 2 / w ** 2) - np.exp(-2.0 * rb ** 2 / w ** 2))


def probe_weights(cfg: SimConfig, grid: Grid) -> np.ndarray:
    """Normalised radial weights for the probe-averaged surface temperature."""
    if cfg.laser.probe_w <= 0:
        wts = np.zeros(grid.nr)
        wts[0] = 1.0
        return wts
    wts = gaussian_annulus_weights(grid.r_faces, cfg.laser.probe_w, cfg.geometry.R_cu)
    return wts / wts.sum()


# ----------------------------------------------------------------------------- result
@dataclass
class SimResult:
    config: SimConfig
    grid: Grid
    material: np.ndarray            # (nz, nr)
    times: np.ndarray               # (nt,)
    T_surface: np.ndarray           # (nt, nr) temperature of the z=0 cell row
    T_probe: np.ndarray             # (nt,) probe-weighted surface temperature
    T_axis: np.ndarray              # (nt, nz) temperature along r~0
    snapshots: list                 # [(t, T(nz, nr))]
    E_in: np.ndarray                # (nt,) cumulative absorbed energy, J
    E_stored: np.ndarray            # (nt,) stored thermal energy, J
    pulse: np.ndarray               # (nt,) f(t)
    wall_time: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def energy_error(self) -> float:
        """Relative energy-balance error at the end of the run."""
        return float((self.E_stored[-1] - self.E_in[-1]) / self.E_in[-1])

    @property
    def energy_error_max(self) -> float:
        E = np.where(self.E_in > 0, self.E_in, np.nan)
        return float(np.nanmax(np.abs(self.E_stored - self.E_in) / E))

    @property
    def dT_probe(self) -> np.ndarray:
        return self.T_probe - self.config.numerics.T0

    @property
    def n_cells(self) -> int:
        return self.grid.n_cells

    @property
    def n_steps(self) -> int:
        return len(self.times) - 1


# ----------------------------------------------------------------------------- solver
def run_simulation(
    cfg: SimConfig,
    progress: Optional[Callable[[float], None]] = None,
    store_fields: bool = True,
) -> SimResult:
    t_start = time.perf_counter()
    grid = build_grid(cfg)
    mat = material_map(cfg, grid)
    k, rho_cp = _property_arrays(cfg, mat)
    nr, nz = grid.nr, grid.nz
    N = nr * nz

    C = (rho_cp * grid.volume).ravel()                       # J/K per cell
    Lap = assemble_laplacian(grid, k)
    Cdiag = sp.diags(C)
    times = cfg.time_grid()
    dts = np.diff(times)

    lu, B, dt_cur, n_factor = None, None, None, 0

    def factorise(dt):
        A = (Cdiag * (1.0 / dt) + 0.5 * Lap).tocsc()
        return spla.splu(A), (Cdiag * (1.0 / dt) - 0.5 * Lap).tocsr()

    # surface source (W per cell at f = 1): copper cells of the j = 0 row
    la = cfg.laser
    q_abs = la.I0 * (1.0 - la.reflectivity)
    Wsrc = gaussian_annulus_weights(grid.r_faces, la.w, cfg.geometry.R_cu)
    Wsrc[mat[0, :] != MAT_COPPER] = 0.0          # transparent silica surface receives nothing
    src = np.zeros(N)
    src[:nr] = q_abs * Wsrc

    n_steps = len(times) - 1
    pulse = la.f(times)
    T0 = cfg.numerics.T0
    # The problem is linear: integrate the temperature *rise* theta = T - T0 (avoids cancellation in
    # the energy balance) and add T0 on output.
    T = np.zeros(N)

    # Surface (face) temperature reconstruction: the cell centre sits at dz0/2 below the face, so
    # T_face = T_c + q_face * (dz0/2) / k for the heated copper cells (second-order, exact for a
    # linear profile).  This is the quantity a thermoreflectance probe sees.
    dz0 = grid.dz_c[0]
    q_face_unit = src[:nr] / grid.face_area_z          # W/m^2 at f = 1
    face_corr = q_face_unit * (0.5 * dz0) / k[0, :]    # K at f = 1

    pw = probe_weights(cfg, grid)
    T_surface = np.empty((n_steps + 1, nr))
    T_probe = np.empty(n_steps + 1)
    T_axis = np.empty((n_steps + 1, nz))
    E_in = np.zeros(n_steps + 1)
    E_stored = np.zeros(n_steps + 1)
    if store_fields:
        snap_idx = set(np.unique(np.linspace(0, n_steps, cfg.numerics.n_snapshots).round().astype(int)))
    else:
        snap_idx = set()
    snapshots = []

    def record(n):
        T2 = T.reshape(nz, nr)
        Ts = T2[0] + face_corr * pulse[n] + T0
        T_surface[n] = Ts
        T_probe[n] = pw @ Ts
        T_axis[n] = T2[:, 0] + T0
        E_stored[n] = float(C @ T)
        if n in snap_idx:
            snapshots.append((times[n], T2 + T0))

    record(0)
    report_every = max(1, n_steps // 100)
    src_total = src.sum()
    f_mean = la.f_mean(times[:-1], times[1:])       # exact step-averaged pulse profile
    for n in range(n_steps):
        dt = dts[n]
        if dt_cur is None or abs(dt - dt_cur) > 1e-12 * dt_cur:
            lu, B = factorise(dt)                   # one LU per constant-dt block
            dt_cur = dt
            n_factor += 1
        f_half = f_mean[n]
        rhs = B @ T + src * f_half
        T = lu.solve(rhs)
        E_in[n + 1] = E_in[n] + f_half * src_total * dt
        record(n + 1)
        if progress is not None and (n % report_every == 0 or n == n_steps - 1):
            progress((n + 1) / n_steps)

    diag = cfg.diagnostics()
    diag.update(
        n_cells=N, nr=nr, nz=nz, absorbed_energy=float(E_in[-1]), n_factorisations=n_factor,
        absorbed_fraction_in_rod=float(Wsrc.sum() / (0.5 * math.pi * la.w ** 2)),
    )
    return SimResult(
        config=cfg, grid=grid, material=mat, times=times, T_surface=T_surface, T_probe=T_probe,
        T_axis=T_axis, snapshots=snapshots, E_in=E_in, E_stored=E_stored, pulse=pulse,
        wall_time=time.perf_counter() - t_start, diagnostics=diag,
    )


def baseline_config(cfg: SimConfig) -> SimConfig:
    """Same configuration without the void (reference for the void signal)."""
    return replace(cfg, void=replace(cfg.void, enabled=False))


def refined_config(cfg: SimConfig, factor: float) -> SimConfig:
    """Spatially refined configuration (dz, dr divided by `factor`; dt follows via the fixed Fo)."""
    n = cfg.numerics
    return replace(cfg, numerics=replace(n, dz=n.dz / factor, dr=cfg.dr / factor))


def void_signal(base: SimResult, void: SimResult) -> dict:
    """Metrics of the thermoreflectance void signature (probe temperature difference)."""
    t = void.times
    if len(base.times) != len(t) or not np.allclose(base.times, t):
        Tb = np.interp(t, base.times, base.T_probe)
    else:
        Tb = base.T_probe
    dT = void.T_probe - Tb
    rise_b = Tb - base.config.numerics.T0
    i_peak = int(np.argmax(np.abs(dT)))
    # Relative contrast where the baseline rise is still resolvable (>= 1e-3 of its peak; deep voids
    # under short pulses respond when the surface has cooled to ~1e-4 of the peak, so the threshold
    # must be well below the classic 1%).
    valid = rise_b > 1e-3 * rise_b.max()
    contrast = np.zeros_like(dT)
    contrast[valid] = dT[valid] / rise_b[valid]
    i_c = int(np.argmax(np.abs(contrast)))
    contrast_at_peak = float(dT[i_peak] / rise_b[i_peak]) if rise_b[i_peak] > 0 else float("nan")
    return dict(
        dT=dT, contrast=contrast, Tb=Tb,
        peak_dT=float(dT[i_peak]), t_peak=float(t[i_peak]), contrast_at_peak=contrast_at_peak,
        peak_contrast=float(contrast[i_c]), t_peak_contrast=float(t[i_c]),
        peak_rise_base=float(rise_b.max()),
        peak_rise_void=float((void.T_probe - void.config.numerics.T0).max()),
    )
