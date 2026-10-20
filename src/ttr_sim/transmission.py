"""CW transmission mode (docs/cw-transmission-spec.md, step 2): configuration builder and step-response run.

Geometry / conventions
----------------------
* z = 0 is the PUMP face (heat source), z = L the opposite (far) face; both faces are adiabatic (air), the
  bottom stack of the reflection mode is disabled.
* r <= R_cu copper, R_cu < r <= R_out silica.  Outer boundary option `outer`:
    "far"               : large adiabatic shell (automatic radius, 4 silica diffusion lengths of the window); valid
                          until the heat reaches it, which `StepResponse.warnings` reports (spec §2.1).
    "neighbor_sink" / "neighbor_adiabatic" : spec step 5, not implemented yet (raise).
* Pump: CW, switched on at t = 0, incident power `power`, flat-top spot of equivalent circular radius
  a = sqrt(6 um * 5 um) ~ 5.5 um (12 x 10 um ellipse), absorbed fraction 1 - reflectivity.
* Probes: Gaussian weights, 1/e^2 radius 4 um on both faces.

Grid / time stepping (all through the existing `SimConfig` machinery, nothing CW-specific in the solver)
* `Laser.tau_p` is not a pulse width in CW mode; it only sets the surface-layer refinement 10 sqrt(alpha tau_p)
  of the grid.  The builder uses tau_p = a^2 / alpha_cu (radial equilibration time of the spot), i.e. a surface
  layer of 10 spot radii, which covers the spreading-resistance region next to the pump spot.
* `Numerics.far_layer` adds a uniform zone against the far face so the far probe sits on a resolved face.
* The time step grows geometrically from the start (the source has no end), `dt_growth` per step.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import COPPER, FUSED_SILICA, Material
from .solver import (
    MAT_COPPER, BottomStack, Geometry, Laser, Numerics, SimConfig, SimResult, VoidSpec, _property_arrays,
    assemble_laplacian, beam_annulus_weights, build_grid, far_face_reconstructor, interface_resistance, material_map,
    probe_weights, run_simulation, sink_conductances,
)

PUMP_SPOT_RADIUS = math.sqrt(6e-6 * 5e-6)   # equivalent circular radius of the flat 12 x 10 um elliptical spot, m
PROBE_RADIUS = 4e-6                         # 1/e^2 radius of the 8 um probe spots, m
DEFAULT_WINDOW = 20e-3                      # observation window, s (~10 L^2/alpha for the 500 um copper via)
DEFAULT_POWER = 10e-3                       # incident pump power, W
DEFAULT_REFLECTIVITY = 0.6                  # Cu reflectivity at 532 nm after CMP (spec §6, unconfirmed)
OUTER_OPTIONS = ("far", "neighbor_sink", "neighbor_adiabatic")


def null_void() -> VoidSpec:
    """A disabled void with zero extent: the grid then gets no void refinement zones at all."""
    return VoidSpec(enabled=False, thickness=0.0, r_half=0.0)


def cw_config(
    power: float = DEFAULT_POWER,
    reflectivity: float = DEFAULT_REFLECTIVITY,
    L: float = 500e-6,
    R_cu: float = 40e-6,
    pump_radius: float = PUMP_SPOT_RADIUS,
    probe_w: float = PROBE_RADIUS,
    t_end: float = DEFAULT_WINDOW,
    void: Optional[VoidSpec] = None,
    extra_voids: tuple = (),
    outer: str = "far",
    R_out: Optional[float] = None,
    homogeneous_copper: bool = False,
    dz: float = 2.5e-6,
    far_layer: Optional[float] = None,
    dt_growth: float = 1.05,
    stretch: float = 1.10,
    copper: Material = COPPER,
    silica: Material = FUSED_SILICA,
    c_tr: float = -1.5e-4,
    **numerics,
) -> SimConfig:
    """SimConfig for the CW transmission measurement (pump at z = 0, probes on both faces, far face adiabatic).

    `void` (and `extra_voids`) follow the reflection-mode `VoidSpec` with depths measured from the pump face;
    None gives the void-free baseline.  `R_out` overrides the automatic outer radius (R_out = R_cu: no silica at
    all, an adiabatic rod side, used by the 1-D analytic checks).  `**numerics` are forwarded to `Numerics`.
    """
    if outer not in OUTER_OPTIONS:
        raise ValueError(f"outer must be one of {OUTER_OPTIONS}, got {outer!r}")
    if outer != "far":
        raise NotImplementedError(f"outer boundary {outer!r} is spec step 5 (neighbour-via boundaries); only 'far' exists yet")
    if power <= 0:
        raise ValueError("power must be positive")
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    laser = Laser(
        tau_p=pump_radius ** 2 / copper.alpha,     # grid surface layer = 10 spot radii (see module docstring)
        profile="square", mode="cw", power=power,
        w=pump_radius, beam="flat", reflectivity=reflectivity,
        probe_w=probe_w, probe_w_far=probe_w, c_tr=c_tr,
    )
    num = Numerics(
        dz=dz, t_end_mode="absolute", t_end_abs=t_end,
        far_layer=(8.0 * dz if far_layer is None else far_layer),
        dt_growth=dt_growth, stretch=stretch, **numerics,
    )
    return SimConfig(
        geometry=Geometry(R_cu=R_cu, L=L, homogeneous_copper=homogeneous_copper, R_out=R_out),
        void=(null_void() if void is None else void),
        extra_voids=tuple(extra_voids),
        laser=laser, numerics=num, copper=copper, silica=silica,
        bottom=BottomStack(enabled=False),
    )


@dataclass
class StepResponse:
    """Step response of the CW transmission configuration (temperature rises in K, times in s)."""
    t: np.ndarray                 # (nt,)
    dT_pump: np.ndarray           # (nt,) probe-weighted rise of the pump face z = 0
    dT_far: np.ndarray            # (nt,) probe-weighted rise of the far face z = L
    P_abs: float                  # absorbed pump power, W
    P_in: float                   # incident pump power, W
    energy_error: float           # (stored - absorbed) / absorbed at the end of the window
    energy_error_max: float       # same, worst step
    E_in: np.ndarray              # (nt,) cumulative absorbed energy, J
    E_stored: np.ndarray          # (nt,) stored thermal energy, J
    result: SimResult
    warnings: list = field(default_factory=list)

    @property
    def c_tr(self) -> float:
        return self.result.config.laser.c_tr

    @property
    def dRR_pump(self) -> np.ndarray:
        """Thermoreflectance signal dR/R of the pump-face probe."""
        return self.c_tr * self.dT_pump

    @property
    def dRR_far(self) -> np.ndarray:
        return self.c_tr * self.dT_far

    @property
    def dT_diff(self) -> np.ndarray:
        """Pump-face minus far-face rise (the two-face temperature difference of spec §3)."""
        return self.dT_pump - self.dT_far

    @property
    def diagnostics(self) -> dict:
        return self.result.diagnostics


def outer_radius(cfg: SimConfig) -> float:
    """Outer radius of the computational domain actually used (automatic shell or `Geometry.R_out`)."""
    return float(build_grid(cfg).r_faces[-1])


def rod_area(cfg: SimConfig) -> float:
    return math.pi * cfg.geometry.R_cu ** 2


def run_step_response(cfg: SimConfig, progress: Optional[Callable[[float], None]] = None,
                      store_fields: bool = True) -> StepResponse:
    """Run the CW step response and return both probe signals with the absorbed power and the energy balance."""
    if not cfg.laser.is_cw:
        raise ValueError("run_step_response needs a CW laser (Laser.mode == 'cw'); use cw_config()")
    res = run_simulation(cfg, progress=progress, store_fields=store_fields)
    la = cfg.laser
    P_abs = la.power * (1.0 - la.reflectivity) * res.diagnostics["absorbed_fraction_in_rod"]
    warnings = list(res.diagnostics.get("warnings", []))
    # spec §2.1: the "far" boundary is only valid while the heat has not reached it
    outer_mat = cfg.copper if cfg.geometry.homogeneous_copper else cfg.silica
    R_max = float(res.grid.r_faces[-1])
    shell = R_max - cfg.geometry.R_cu
    if shell > 0:
        L_out = math.sqrt(outer_mat.alpha * cfg.t_end)
        if L_out > 0.8 * shell:
            warnings.append(
                f"외곽 침투 깊이 sqrt(α_shell·t_end) = {L_out * 1e6:.0f} μm 가 껍질 두께의 80%({0.8 * shell * 1e6:.0f} μm)를 넘습니다. "
                "외곽 단열 경계('far')가 후반부 결과에 영향을 줍니다."
            )
    return StepResponse(
        t=res.times, dT_pump=res.dT_probe, dT_far=res.dT_probe_far, P_abs=P_abs, P_in=la.power,
        energy_error=res.energy_error, energy_error_max=res.energy_error_max,
        E_in=res.E_in, E_stored=res.E_stored, result=res, warnings=warnings,
    )


# ----------------------------------------------------------------------------- 1-D reference (Carslaw & Jaeger)
def slab_step_response(z, t, q: float, L: float, mat: Material = COPPER, n_terms: int = 200) -> np.ndarray:
    """Temperature rise of a slab 0 <= z <= L, initially at 0, with constant flux q (W/m^2) entering at z = 0
    from t = 0 and the face z = L insulated (Carslaw & Jaeger, Conduction of Heat in Solids, §3.8):

        T = (q L / k) [ alpha t / L^2 + (3 (L - z)^2 - L^2) / (6 L^2)
                        - (2 / pi^2) sum_{n>=1} (-1)^n / n^2 exp(-n^2 pi^2 alpha t / L^2) cos(n pi (L - z) / L) ]

    Broadcasts over z and t.  Negative t gives 0."""
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    tau = mat.alpha * np.clip(t, 0.0, None) / L ** 2
    x = (L - z) / L
    n = np.arange(1, n_terms + 1, dtype=float)
    shape = np.broadcast(tau, x).shape
    tau_b = np.broadcast_to(tau, shape)[..., None]
    x_b = np.broadcast_to(x, shape)[..., None]
    series = np.sum((-1.0) ** n / n ** 2 * np.exp(-(n * math.pi) ** 2 * tau_b) * np.cos(n * math.pi * x_b), axis=-1)
    theta = tau + (3.0 * x ** 2 - 1.0) / 6.0 - 2.0 / math.pi ** 2 * series
    return np.where(t > 0.0, q * L / mat.k * theta, 0.0)


def slab_far_face_step(t, q: float, L: float, mat: Material = COPPER, n_terms: int = 200) -> np.ndarray:
    """Far-face (z = L) rise of `slab_step_response`: its linear asymptote is (q / (rho c L)) (t - L^2 / (6 alpha)),
    i.e. the time intercept L^2 / (6 alpha) of spec §1 (the CW-step counterpart of Parker's half-rise time)."""
    return slab_step_response(L, t, q, L, mat, n_terms)


def slab_intercept(L: float, mat: Material = COPPER) -> float:
    """Time intercept L^2 / (6 alpha) of the far-face linear asymptote, s."""
    return L ** 2 / (6.0 * mat.alpha)


# ----------------------------------------------------------------------------- frequency-domain direct solve (spec §2.4)
@dataclass
class FrequencyResponse:
    """Complex temperature amplitudes per unit absorbed-power amplitude [K/W] at the two probes:
    (Lap + i omega C) T^ = src, same matrices as the time integrator; dT(t) = Re(H e^{i omega t}) for
    P_abs(t) = Re(e^{i omega t}), so arg H <= 0 is the lag."""
    freqs: np.ndarray             # Hz
    H_pump: np.ndarray            # complex, K/W
    H_far: np.ndarray             # complex, K/W
    P_abs: float                  # absorbed power of the configuration, W (for lock-in amplitudes)
    n_cells: int
    wall_time: float

    @property
    def amplitude_far(self) -> np.ndarray:
        return np.abs(self.H_far)

    @property
    def phase_far_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.H_far))


def frequency_response(cfg: SimConfig, freqs, progress: Optional[Callable[[float], None]] = None) -> FrequencyResponse:
    """Direct harmonic solution on the CW grid: one complex sparse LU per frequency.  The pump face is
    reconstructed with the (complex) flux correction like the time solver, the far face with the same
    far-face reconstructor; both are probe-weighted.  Exact for the discretised system, so it cross-checks
    the time integrator (step response -> FFT) and gives lock-in amplitude / phase without any transient."""
    t_start = time.perf_counter()
    grid = build_grid(cfg)
    mat = material_map(cfg, grid)
    k, rho_cp = _property_arrays(cfg, mat)
    nr, nz = grid.nr, grid.nz
    N = nr * nz
    C = (rho_cp * grid.volume).ravel()
    r_extra = interface_resistance(cfg, grid)
    Lap = assemble_laplacian(grid, k, r_extra)
    g_sink = sink_conductances(cfg, grid, k)
    if g_sink.any():
        Lap = (Lap + sp.diags(g_sink)).tocsr()
    la = cfg.laser
    Wsrc = beam_annulus_weights(grid.r_faces, la, cfg.geometry.R_cu)
    Wsrc[mat[0, :] != MAT_COPPER] = 0.0
    src = np.zeros(N)
    src[:nr] = la.I0 * (1.0 - la.reflectivity) * Wsrc
    P_abs = float(src.sum())
    face_corr = (src[:nr] / grid.face_area_z) * (0.5 * grid.dz_c[0]) / k[0, :]      # K at unit f (complex f: same)
    far_face = far_face_reconstructor(cfg, grid, k, r_extra, g_sink)
    pw = probe_weights(cfg, grid)
    pw_far = probe_weights(cfg, grid, la.probe_w_far_eff)
    freqs = np.atleast_1d(np.asarray(freqs, float))
    H_pump = np.empty(len(freqs), complex)
    H_far = np.empty(len(freqs), complex)
    Cd = sp.diags(C.astype(complex))
    for i, f in enumerate(freqs):
        w = 2.0 * math.pi * f
        A = (Lap.astype(complex) + 1j * w * Cd).tocsc()
        T = spla.splu(A).solve(src.astype(complex))
        T2 = T.reshape(nz, nr)
        H_pump[i] = (pw @ (T2[0] + face_corr)) / P_abs
        H_far[i] = (pw_far @ far_face(T2)) / P_abs
        if progress is not None:
            progress((i + 1) / len(freqs))
    return FrequencyResponse(freqs=freqs, H_pump=H_pump, H_far=H_far, P_abs=P_abs, n_cells=N,
                             wall_time=time.perf_counter() - t_start)
