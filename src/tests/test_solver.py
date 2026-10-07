"""Verification tests for the ns-TTR simulator core (run with `pytest`)."""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttr_sim import BottomStack, DEPTH_PRESETS, KVOID_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, build_grid, run_simulation  # noqa: E402
from ttr_sim.analytic import center_step_response, center_step_response_flat, surface_step_response  # noqa: E402
from ttr_sim.materials import COPPER, FUSED_SILICA  # noqa: E402
from ttr_sim.solver import MAT_SILICA, MAT_VOID, baseline_config, material_map, void_signal  # noqa: E402
from ttr_sim.validation import analytic_comparison, grid_convergence, kvoid_sensitivity, run_pair  # noqa: E402


def make_cfg(preset_idx=2, profile="square", k_void=1.0, homogeneous=False, **num):
    p = DEPTH_PRESETS[preset_idx]
    d = p.d_rep
    return SimConfig(
        geometry=Geometry(homogeneous_copper=homogeneous),
        void=VoidSpec(enabled=True, depth=d, thickness=0.5 * d, r_half=2 * d, k=k_void),
        laser=Laser(tau_p=p.tau_p, profile=profile, energy=10e-9, w=10e-6, probe_w=5e-6),
        numerics=Numerics(dz=p.dz, **num),
    )


# ----------------------------------------------------------------------------- presets (spec §3 table)
@pytest.mark.parametrize("idx,tau_p,dz_um", [
    (0, 17e-9, 0.14), (1, 69e-9, 0.28), (2, 431e-9, 0.71), (3, 1.7e-6, 1.4), (4, 6.9e-6, 2.8),
    (5, 27.6e-6, 5.7), (6, 172e-6, 14.1), (7, 690e-6, 28.3), (8, 2.76e-3, 56.6),
])
def test_presets_match_spec_table(idx, tau_p, dz_um):
    p = DEPTH_PRESETS[idx]
    assert p.tau_p == pytest.approx(tau_p, rel=0.02)
    assert p.dz * 1e6 == pytest.approx(dz_um, rel=0.02)


def test_kvoid_presets():
    assert [p.k for p in KVOID_PRESETS] == [0.026, 0.1, 1.0, 10.0]
    assert KVOID_PRESETS[-1].warning  # the 10 W/mK option must carry a warning


def test_material_diffusivity_ratio():
    assert COPPER.alpha / FUSED_SILICA.alpha == pytest.approx(136, rel=0.05)


# ----------------------------------------------------------------------------- grid
def test_grid_faces_align_with_interfaces():
    cfg = make_cfg(2)
    g = build_grid(cfg)
    assert np.any(np.isclose(g.r_faces, cfg.geometry.R_cu))
    assert np.any(np.isclose(g.z_faces, cfg.void.depth))
    assert np.any(np.isclose(g.z_faces, cfg.void.z_bottom))
    assert np.any(np.isclose(g.r_faces, cfg.void.r_outer))
    assert g.z_faces[-1] == pytest.approx(cfg.z_total) and np.any(np.isclose(g.z_faces, cfg.geometry.L))
    assert np.all(np.diff(g.r_faces) > 0) and np.all(np.diff(g.z_faces) > 0)
    mat = material_map(cfg, g)
    assert (mat == MAT_VOID).any() and (mat == MAT_SILICA).any()
    # default void shape is an on-axis spheroid (semi-axes r_half, r_half, thickness/2)
    vol_void = (g.volume * (mat == MAT_VOID)).sum()
    vol_spheroid = 4.0 / 3.0 * math.pi * cfg.void.r_half ** 2 * (0.5 * cfg.void.thickness)
    assert vol_void == pytest.approx(vol_spheroid, rel=0.15)      # staircase approximation on a coarse grid
    # the box variant reproduces the cylinder exactly because its faces are grid-aligned
    cfg_box = replace(cfg, void=replace(cfg.void, shape="box"))
    mat_box = material_map(cfg_box, g)
    vol_box = (g.volume * (mat_box == MAT_VOID)).sum()
    assert vol_box == pytest.approx(math.pi * cfg.void.r_outer ** 2 * cfg.void.thickness, rel=1e-6)
    assert vol_void < vol_box


def test_dt_from_fourier_number():
    cfg = make_cfg(2, fo=0.5, dt_growth=1.0)
    assert COPPER.alpha * cfg.dt / cfg.numerics.dz ** 2 == pytest.approx(0.5)
    assert cfg.n_steps == pytest.approx(1000, abs=10)  # 5 tau_p / (tau_p/200), constant dt


def test_time_grid_growth():
    cfg = make_cfg(2, fo=0.5, dt_growth=1.05)
    t = cfg.time_grid()
    dts = np.diff(t)
    assert t[0] == 0.0 and t[-1] == pytest.approx(cfg.t_end)
    assert np.all(dts > 0)
    n_pulse = int(np.sum(t[:-1] < cfg.laser.tau_p))
    assert n_pulse == pytest.approx(200, abs=2)                      # constant dt during the pulse
    assert np.allclose(dts[:n_pulse], cfg.dt)
    assert dts.max() > 10 * cfg.dt and len(t) < 400                   # grows afterwards -> far fewer steps
    assert dts.max() <= 0.06 * cfg.t_end                              # dt ~ 5% of elapsed time


def test_void_based_window_and_short_pulse_deep_void():
    """Short pulse (preset 0) + deep void: feasible only with the graded grid and growing dt."""
    p = DEPTH_PRESETS[0]
    cfg = SimConfig(
        geometry=Geometry(),
        void=VoidSpec(enabled=True, depth=150e-6, thickness=40e-6, r_half=40e-6, k=1.0, shape="box"),
        laser=Laser(tau_p=p.tau_p, energy=100e-9, w=10e-6, probe_w=5e-6),
        numerics=Numerics(dz=p.dz, t_end_mode="void", t_end_factor=3.0, dt_growth=1.05, n_snapshots=6),
    )
    tau_void = cfg.void.depth ** 2 / COPPER.alpha
    assert cfg.t_end == pytest.approx(3.0 * tau_void)                 # ~0.6 ms window from a 17 ns pulse
    g = build_grid(cfg)
    assert g.n_cells < 300_000 and cfg.n_steps < 2000
    assert np.any(np.isclose(g.z_faces, cfg.void.depth)) and np.any(np.isclose(g.z_faces, cfg.void.z_bottom))
    base, void, sig = run_pair(cfg)
    assert abs(base.energy_error) < 1e-9 and abs(void.energy_error) < 1e-9
    assert sig["peak_dT"] > 0 and sig["t_peak"] > 0.3 * tau_void    # signature arrives on the d^2/D time scale


# ----------------------------------------------------------------------------- 3-D (off-axis void)
def _cfg3d(r_center_um, void_cells=4, shape="ellipse"):
    p = DEPTH_PRESETS[4]   # 6.9 us pulse, dz 2.8 um -> small (r, z) grid
    return SimConfig(
        geometry=Geometry(),
        void=VoidSpec(enabled=True, depth=20e-6, thickness=10e-6, r_center=r_center_um * 1e-6, r_half=8e-6, k=0.026, shape=shape),
        laser=Laser(tau_p=p.tau_p, energy=100e-9, w=10e-6, probe_w=5e-6),
        numerics=Numerics(dz=p.dz, n_snapshots=4, void_cells=void_cells),
    )


def test_grid_zones_resolve_void_automatically():
    """Surface layer from tau_p, void zone from the void size: a thin deep void gets >= void_cells cells."""
    p = DEPTH_PRESETS[7]   # 690 us pulse, dz 28 um
    cfg = SimConfig(Geometry(), VoidSpec(enabled=True, depth=200e-6, thickness=2e-6, r_half=3e-6, k=0.026),
                    Laser(tau_p=p.tau_p), Numerics(dz=p.dz, void_cells=8))
    g = build_grid(cfg)
    assert cfg.dz_void == pytest.approx(2e-6 / 8) and cfg.dr_void == pytest.approx(3e-6 / 8)
    in_void_z = (g.z_c > cfg.void.depth) & (g.z_c < cfg.void.z_bottom)
    in_void_r = g.r_c < cfg.void.r_half
    assert in_void_z.sum() >= 8 and in_void_r.sum() >= 8
    assert np.any(np.isclose(g.z_faces, cfg.void.depth)) and np.any(np.isclose(g.z_faces, cfg.void.z_bottom))
    assert np.any(np.isclose(g.r_faces, cfg.void.r_half)) and np.any(np.isclose(g.r_faces, cfg.geometry.R_cu))
    assert np.all(np.diff(g.z_faces) > 0) and np.all(np.diff(g.r_faces) > 0)
    assert g.n_cells < 60_000                                     # zoning keeps the grid small
    assert g.z_faces[-1] == pytest.approx(cfg.z_total) and np.any(np.isclose(g.z_faces, cfg.geometry.L))


def test_3d_reproduces_2d_for_on_axis_void():
    """Forced 3-D run of an axisymmetric case must equal the 2-D solver (same (r, z) grid, direct solves)."""
    from ttr_sim.solver3d import run_simulation_3d, theta_faces
    cfg = _cfg3d(0.0)
    th = theta_faces(cfg)
    assert th[0] == 0.0 and th[-1] == pytest.approx(math.pi) and np.all(np.diff(th) > 0)
    r2 = run_simulation(cfg)
    r3 = run_simulation_3d(cfg)
    assert r3.is_3d and r3.n_theta == len(th) - 1
    assert np.allclose(r3.T_probe, r2.T_probe, rtol=1e-7, atol=1e-9)
    assert np.allclose(r3.T_axis, r2.T_axis, rtol=1e-7, atol=1e-9)
    assert abs(r3.energy_error) < 1e-9
    # the displayed plane is the mirrored 2-D field
    _, plane = r3.snapshots[-1]
    _, T2 = r2.snapshots[-1]
    assert plane.shape == (r2.grid.nz, 2 * r2.grid.nr)
    assert np.allclose(plane[:, r2.grid.nr:], T2, rtol=1e-7, atol=1e-9)


def test_off_axis_void_dispatch_and_signal():
    from ttr_sim.validation import run_pair
    base, v0, s0 = run_pair(_cfg3d(0.0))                # on axis -> 2-D
    base3, v1, s1 = run_pair(_cfg3d(15.0))              # displaced -> 3-D
    assert not v0.is_3d and v1.is_3d
    assert np.allclose(base.T_probe, base3.T_probe)     # baseline stays 2-D and identical
    assert abs(v1.energy_error) < 1e-9
    assert 0 < s1["peak_dT"] < s0["peak_dT"]            # a displaced void gives a weaker but positive signal
    # symmetry of the displayed plane: void appears only on the x > 0 side
    from ttr_sim.solver import MAT_VOID
    nr = v1.grid.nr
    assert (v1.material[:, nr:] == MAT_VOID).any() and not (v1.material[:, :nr] == MAT_VOID).any()


def test_3d_void_volume():
    """Staircase spheroid volume on the 3-D grid is close to 4/3 pi a^2 c (both halves)."""
    from ttr_sim.solver3d import material_map_3d, theta_faces
    cfg = _cfg3d(15.0, void_cells=8)
    g = build_grid(cfg)
    th = theta_faces(cfg)
    mat = material_map_3d(cfg, g, th)
    dth = np.diff(th)
    V = 0.5 * (g.r_faces[1:] ** 2 - g.r_faces[:-1] ** 2)[None, :, None] * dth[None, None, :] * g.dz_c[:, None, None]
    vol = 2.0 * float((np.broadcast_to(V, mat.shape) * (mat == MAT_VOID)).sum())
    v = cfg.void
    assert vol == pytest.approx(4.0 / 3.0 * math.pi * v.r_half ** 2 * 0.5 * v.thickness, rel=0.2)


@pytest.mark.parametrize("profile", ["square", "gaussian"])
def test_analytic_with_growing_dt(profile):
    cfg = make_cfg(2, profile=profile, homogeneous=True, dt_growth=1.05)
    base = run_simulation(baseline_config(cfg))
    cmp = analytic_comparison(base)
    assert cmp["rms_rel"] < 0.015 and abs(cmp["peak_rel"]) < 0.02


# ----------------------------------------------------------------------------- energy conservation
@pytest.mark.parametrize("profile", ["square", "gaussian"])
@pytest.mark.parametrize("k_void", [0.026, 10.0])
def test_energy_conservation(profile, k_void):
    res = run_simulation(make_cfg(2, profile=profile, k_void=k_void))
    assert abs(res.energy_error) < 1e-9
    assert res.energy_error_max < 1e-9
    # absorbed energy = E (1-R) * fraction of the Gaussian inside the rod.  abs=0: pytest.approx's default
    # absolute tolerance (1e-12) would otherwise hide a 2e-4 relative deficit on nJ-scale energies.
    la = res.config.laser
    frac = 1 - math.exp(-2 * res.config.geometry.R_cu ** 2 / la.w ** 2)
    assert res.E_in[-1] == pytest.approx(la.energy * (1 - la.reflectivity) * frac, rel=1e-9, abs=0)


# ----------------------------------------------------------------------------- analytic solution
def test_quadrature_matches_closed_form_on_axis():
    tau = np.array([1e-9, 1e-7, 1e-5, 1e-3])
    q = surface_step_response([0.0], tau, 1e9, 10e-6, COPPER)[:, 0]
    assert np.allclose(q, center_step_response(tau, 1e9, 10e-6, COPPER), rtol=1e-6)


def test_step_response_limits():
    q0, w = 1e9, 10e-6
    t_short = 1e-12                      # 1-D limit: 2 q0 sqrt(alpha t / pi) / k
    assert center_step_response(t_short, q0, w, COPPER) == pytest.approx(
        2 * q0 * math.sqrt(COPPER.alpha * t_short / math.pi) / COPPER.k, rel=1e-3)
    t_long = 1e3                         # Lax steady state: q0 w sqrt(pi/8)/k
    assert center_step_response(t_long, q0, w, COPPER) == pytest.approx(q0 * w * math.sqrt(math.pi / 8) / COPPER.k, rel=1e-3)


@pytest.mark.parametrize("preset_idx,profile", [(0, "square"), (2, "square"), (2, "gaussian"), (4, "gaussian")])
def test_baseline_matches_analytic(preset_idx, profile):
    cfg = make_cfg(preset_idx, profile=profile, homogeneous=True)
    base = run_simulation(baseline_config(cfg))
    cmp = analytic_comparison(base)
    assert cmp["rms_rel"] < 0.01
    assert abs(cmp["peak_rel"]) < 0.02


def test_silica_shell_reduces_late_time_cooling():
    """With silica around the rod, heat is confined -> the surface stays hotter than the semi-infinite copper case."""
    cfg = make_cfg(5, homogeneous=False)   # 40 um preset: heat reaches r = 40 um within the window
    base = run_simulation(baseline_config(cfg))
    cmp = analytic_comparison(base)
    assert cmp["numeric"][-1] > cmp["analytic"][-1]
    assert cmp["notes"]


# ----------------------------------------------------------------------------- void physics
def test_void_raises_surface_temperature_and_is_sensitive_to_depth():
    cfg = make_cfg(2)
    base, void, sig = run_pair(cfg)
    assert sig["peak_dT"] > 0                       # insulating void traps heat -> hotter surface
    assert sig["t_peak"] > 0.5 * cfg.laser.tau_p    # signature appears after the heat has diffused to the void
    deeper = replace(cfg, void=replace(cfg.void, depth=3 * cfg.void.depth))
    sig2 = void_signal(base, run_simulation(deeper))
    assert 0 < sig2["peak_dT"] < sig["peak_dT"]


def test_kvoid_sensitivity_is_monotonic_and_small_below_1():
    cfg = make_cfg(2)
    base = run_simulation(baseline_config(cfg))
    rows = kvoid_sensitivity(cfg, base)
    peaks = [r["peak_dT"] for r in rows]
    assert peaks == sorted(peaks, reverse=True)     # smaller k_void -> stronger signal
    assert abs(rows[1]["d_peak_dT_vs_min"]) < 0.01  # 0.1 vs 0.026: negligible
    assert abs(rows[2]["d_peak_dT_vs_min"]) < 0.05  # 1.0 vs 0.026: small
    assert rows[3]["d_peak_dT_vs_min"] < -0.05       # 10 vs 0.026: clearly different
    assert all(abs(r["energy_error"]) < 1e-9 for r in rows)


def test_grid_convergence():
    cfg = make_cfg(3)
    base, void, sig = run_pair(cfg)
    calls = []
    conv = grid_convergence(cfg, base, void, sig, factor=2.0, include_coarse=True, include_dt_half=True,
                            progress=lambda frac, label: calls.append((frac, label)))
    fracs = [c[0] for c in calls]
    assert fracs == sorted(fracs) and 0 < fracs[0] and fracs[-1] == pytest.approx(1.0)   # nested progress is monotonic
    assert all(" · " in c[1] for c in calls)                                              # outer and inner labels
    cur, dth = conv["comparisons"]["current"], conv["comparisons"]["dt_half"]
    assert abs(cur["d_peak_rise"]) < 0.01
    assert abs(cur["d_peak_dT"]) < 0.05
    assert abs(dth["d_peak_dT"]) < abs(conv["comparisons"]["coarse"]["d_peak_dT"]) + 1e-12
    assert conv["order"] is None or conv["order"] > 1.0

# ----------------------------------------------------------------------------- bottom stack (oxide + thermal pad + heat sink)
def _cfg_deep(stack):
    p = DEPTH_PRESETS[8]   # 2.76 ms pulse: heat reaches the back face within the window
    return SimConfig(
        geometry=Geometry(),
        void=VoidSpec(enabled=True, depth=300e-6, thickness=50e-6, r_half=40e-6, k=0.026),
        laser=Laser(tau_p=p.tau_p, energy=1e-6, w=10e-6, probe_w=5e-6),
        numerics=Numerics(dz=p.dz, n_snapshots=4), bottom=stack,
    )


def test_bottom_stack_grid_and_materials():
    from ttr_sim.solver import MAT_OXIDE, MAT_PAD
    cfg = _cfg_deep(BottomStack(oxide_thickness=1e-6))          # thick oxide: resolved as its own cells
    assert cfg.bottom.oxide_resolved and cfg.bottom.oxide_resistance == 0.0
    g = build_grid(cfg)
    mat = material_map(cfg, g)
    L, b = cfg.geometry.L, cfg.bottom
    assert g.z_faces[-1] == pytest.approx(L + b.oxide_thickness + b.pad_thickness)
    assert np.any(np.isclose(g.z_faces, L)) and np.any(np.isclose(g.z_faces, L + b.oxide_thickness))
    ox, pad = (mat[:, 0] == MAT_OXIDE), (mat[:, 0] == MAT_PAD)
    assert ox.sum() >= 4 and pad.sum() >= 8
    assert (g.dz_c[ox].sum()) == pytest.approx(b.oxide_thickness) and (g.dz_c[pad].sum()) == pytest.approx(b.pad_thickness)
    assert (mat[ox][:, -1] == MAT_OXIDE).all() and (mat[pad][:, -1] == MAT_PAD).all()     # layers span the whole radius
    assert build_grid(_cfg_deep(BottomStack(enabled=False))).z_faces[-1] == pytest.approx(L)


def test_native_oxide_is_an_interface_resistance():
    """Default (native, 5 nm) oxide: no oxide cells, resistance t/k on the face at z = L, negligible effect."""
    from ttr_sim.solver import MAT_OXIDE, interface_resistance
    cfg = _cfg_deep(BottomStack())
    b = cfg.bottom
    assert not b.oxide_resolved and b.oxide_resistance == pytest.approx(5e-9 / 4.5)
    g = build_grid(cfg)
    assert g.z_faces[-1] == pytest.approx(cfg.geometry.L + b.pad_thickness) and not (material_map(cfg, g) == MAT_OXIDE).any()
    extra = interface_resistance(cfg, g)
    assert (extra > 0).sum() == 1 and g.z_faces[1:-1][extra > 0][0] == pytest.approx(cfg.geometry.L)
    with_ox = run_simulation(cfg)
    no_ox = run_simulation(_cfg_deep(BottomStack(oxide_thickness=0.0)))
    assert abs(with_ox.energy_error) < 1e-9
    assert np.allclose(with_ox.dT_probe, no_ox.dT_probe, rtol=1e-3, atol=1e-9)      # nm oxide is thermally negligible


def test_interface_and_resolved_oxide_agree():
    """Just below / above the switch-over thickness the two treatments must give the same answer."""
    thin = run_simulation(_cfg_deep(BottomStack(oxide_thickness=0.19e-6, oxide_k=0.5)))     # interface resistance
    thick = run_simulation(_cfg_deep(BottomStack(oxide_thickness=0.21e-6, oxide_k=0.5)))    # resolved cells
    assert not thin.config.bottom.oxide_resolved and thick.config.bottom.oxide_resolved
    assert np.allclose(thin.dT_probe, thick.dT_probe, rtol=2e-2, atol=1e-9)
    assert thin.E_lost[-1] == pytest.approx(thick.E_lost[-1], rel=0.05)


def test_bottom_stack_energy_balance_and_cooling():
    sink = run_simulation(_cfg_deep(BottomStack(sink="isothermal")))
    adia = run_simulation(_cfg_deep(BottomStack(sink="adiabatic")))
    assert abs(sink.energy_error) < 1e-9 and sink.energy_error_max < 1e-9       # stored + lost = absorbed
    assert abs(adia.energy_error) < 1e-9
    assert sink.E_lost[-1] > 0.05 * sink.E_in[-1] and adia.E_lost[-1] == 0.0   # the sink removes heat
    assert np.all(np.diff(sink.E_lost) >= -1e-30)
    assert sink.dT_probe[-1] < adia.dT_probe[-1]                                  # and cools the surface at late times


# ----------------------------------------------------------------------------- flat-top pump beam
def test_flat_top_beam_energy_and_uniform_heating():
    """Flat-top beam covering the whole rod: all of E (1-R) lands on the copper, the surface heats uniformly,
    and during the pulse the surface follows the 1-D uniform-flux solution 2 q sqrt(alpha t / pi) / k
    (the silica shell is a poor conductor, so the rod behaves almost one-dimensionally)."""
    cfg = baseline_config(make_cfg(2))
    R = cfg.geometry.R_cu
    cfg = replace(cfg, laser=replace(cfg.laser, beam="flat", w=R))
    res = run_simulation(cfg)
    la = cfg.laser
    assert abs(res.energy_error) < 1e-9
    assert res.E_in[-1] == pytest.approx(la.energy * (1 - la.reflectivity), rel=1e-9, abs=0)
    i_end = int(np.searchsorted(res.times, la.tau_p)) - 1               # last step inside the pulse
    Ts = res.T_surface[i_end] - cfg.numerics.T0
    in_rod = res.grid.r_c < 0.8 * R                                     # away from the copper/silica edge
    assert Ts[in_rod].max() / Ts[in_rod].min() < 1.02                   # flat across the rod
    q = la.I0 * (1 - la.reflectivity)
    one_d = 2.0 * q * math.sqrt(COPPER.alpha * res.times[i_end] / math.pi) / COPPER.k
    assert Ts[0] == pytest.approx(one_d, rel=0.03)


def test_flat_top_beam_partial_spot_and_analytic_flag():
    """A flat-top spot smaller than the rod deposits its whole energy inside r <= w; no analytic comparison."""
    cfg = baseline_config(make_cfg(2))
    cfg = replace(cfg, laser=replace(cfg.laser, beam="flat", w=12.3e-6))
    res = run_simulation(cfg)
    la = cfg.laser
    assert res.E_in[-1] == pytest.approx(la.energy * (1 - la.reflectivity), rel=1e-9, abs=0)
    assert la.I0 == pytest.approx(la.energy / (math.pi * la.w ** 2 * la.tau_p))
    i_end = int(np.searchsorted(res.times, la.tau_p)) - 1
    Ts = res.T_surface[i_end] - cfg.numerics.T0
    assert Ts[0] > 5 * Ts[np.searchsorted(res.grid.r_c, 30e-6)]         # hot inside the spot, cool well outside
    centre = center_step_response_flat(res.times[i_end], la.I0 * (1 - la.reflectivity), la.w, COPPER)
    assert Ts[0] == pytest.approx(float(centre), rel=0.05)              # semi-infinite disc-source centre value
    assert analytic_comparison(res)["available"] is False


# ----------------------------------------------------------------------------- several voids / full-circle 3-D
def test_full_circle_single_void_matches_half_cylinder():
    """A void at theta = 1.3 rad (full-circle solver) is the same physical case as the void at theta = 0
    (half-cylinder solver); only the azimuthal grid differs, so the signals must agree closely."""
    from ttr_sim.solver3d import full_circle, run_simulation_3d, theta_faces
    c0 = _cfg3d(15.0)
    c1 = replace(c0, void=replace(c0.void, theta=1.3))
    assert not full_circle(c0) and full_circle(c1)
    th = theta_faces(c1)
    assert th[0] == 0.0 and th[-1] == pytest.approx(2 * math.pi) and np.all(np.diff(th) > 0)
    base = run_simulation(baseline_config(c0))
    r0, r1 = run_simulation_3d(c0), run_simulation_3d(c1)
    s0, s1 = void_signal(base, r0), void_signal(base, r1)
    assert abs(r1.energy_error) < 1e-9
    assert s1["peak_dT"] == pytest.approx(s0["peak_dT"], rel=0.03)
    assert s1["t_peak"] == pytest.approx(s0["t_peak"], rel=0.1)


def test_stacked_on_axis_voids_stay_2d():
    """Two on-axis voids are axisymmetric: 2-D solve, exact energy balance, more signal than either alone."""
    from ttr_sim.solver3d import needs_3d
    p = DEPTH_PRESETS[3]
    va = VoidSpec(enabled=True, depth=10e-6, thickness=5e-6, r_half=10e-6, k=0.026)
    vb = replace(va, depth=30e-6)
    mk = lambda void, extra=(): SimConfig(geometry=Geometry(), void=void, extra_voids=tuple(extra),  # noqa: E731
                                          laser=Laser(tau_p=p.tau_p, energy=20e-9, w=10e-6, probe_w=5e-6),
                                          numerics=Numerics(dz=p.dz, n_snapshots=4))
    cab = mk(va, [vb])
    assert not needs_3d(cab) and len(cab.voids) == 2 and cab.void_depth_min == va.depth
    assert baseline_config(cab).voids == [] and baseline_config(cab).extra_voids == ()
    _, ra, sa = run_pair(mk(va))
    _, rb, sb = run_pair(mk(vb))
    _, rab, sab = run_pair(cab)
    assert not rab.is_3d and abs(rab.energy_error) < 1e-9
    assert (rab.material == MAT_VOID).sum() > max((ra.material == MAT_VOID).sum(), (rb.material == MAT_VOID).sum())
    assert sab["peak_dT"] > max(sa["peak_dT"], sb["peak_dT"])


def test_random_voids_generator():
    from ttr_sim import random_voids
    g = Geometry()
    vs = random_voids(50, 5e-6, 4e-6, g, seed=3)
    assert len(vs) == 50
    assert all(v.enabled and v.r_half == 5e-6 and v.thickness == 4e-6 for v in vs)
    assert all(0.0 <= v.r_center <= g.R_cu - v.r_half for v in vs)             # void inside the rod radially
    assert all(0.0 <= v.depth and v.z_bottom <= g.L + 1e-12 for v in vs)       # ... and axially
    assert all(0.0 <= v.theta < 2 * math.pi for v in vs)
    assert min(v.theta for v in vs) < 1.0 and max(v.theta for v in vs) > 5.0  # spread around the circle
    assert [v.depth for v in random_voids(50, 5e-6, 4e-6, g, seed=3)] == [v.depth for v in vs]   # deterministic
    assert [v.depth for v in random_voids(50, 5e-6, 4e-6, g, seed=4)] != [v.depth for v in vs]


def test_random_arrangement_runs_on_the_full_circle():
    from ttr_sim import random_voids
    from ttr_sim.solver3d import full_circle, needs_3d
    from ttr_sim.validation import run_pair
    c = _cfg3d(0.0)
    vs = random_voids(2, 8e-6, 10e-6, c.geometry, seed=7, template=c.void)
    c = replace(c, void=vs[0], extra_voids=tuple(vs[1:]))
    assert needs_3d(c) and full_circle(c)
    base, r, sig = run_pair(c)
    assert r.is_3d and r.diagnostics["full_circle"] and abs(r.energy_error) < 1e-9
    assert sig["peak_dT"] > 0


def test_fit_resolution_lowers_void_cells_until_the_unknowns_fit():
    """Many small voids over the rod: the mesh is coarsened per void (void_cells 8 -> >= 3) instead of failing
    inside the solver; a single void that already fits is left alone."""
    from ttr_sim import random_voids
    from ttr_sim.solver3d import count_unknowns, fit_resolution, max_unknowns
    c1 = _cfg3d(15.0, void_cells=8)
    assert fit_resolution(c1) is c1 and count_unknowns(c1)[2] <= max_unknowns(c1)
    p = DEPTH_PRESETS[2]
    vs = random_voids(10, 5e-6, 5e-6, Geometry(), seed=1, template=VoidSpec(k=0.026))
    c = SimConfig(Geometry(), vs[0], Laser(tau_p=p.tau_p), Numerics(dz=p.dz, void_cells=8), extra_voids=tuple(vs[1:]))
    assert count_unknowns(c)[2] > max_unknowns(c)
    f = fit_resolution(c)
    assert 3 <= f.numerics.void_cells < 8 and count_unknowns(f)[2] <= max_unknowns(f)
    assert f.voids == c.voids and f.laser == c.laser
    # hopeless case (tiny voids under a short pulse): stops at the floor, the caller sees it is still too big
    tiny = random_voids(10, 1e-6, 1e-6, Geometry(), seed=1, template=VoidSpec(k=0.026))
    c2 = SimConfig(Geometry(), tiny[0], Laser(tau_p=p.tau_p), Numerics(dz=p.dz, void_cells=8), extra_voids=tuple(tiny[1:]))
    f2 = fit_resolution(c2)
    assert f2.numerics.void_cells == 3 and count_unknowns(f2)[2] > max_unknowns(f2)
    with pytest.raises(ValueError, match="한도"):
        run_pair(c2)


def test_3d_solver_falls_back_to_ilu_when_the_structured_preconditioner_cannot_be_built(monkeypatch):
    """A failed SuperLU allocation inside the structured preconditioner must not abort the run: the solver
    switches to the ILU-preconditioned BiCGSTAB and still converges."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from ttr_sim import solver3d
    from ttr_sim.solver3d import _Solver, SOLVER_ILU, SOLVER_STRUCTURED, theta_faces
    cfg = _cfg3d(15.0)
    g = build_grid(cfg)
    th = theta_faces(cfg)
    mat = solver3d.material_map_3d(cfg, g, th)
    k3, rho_cp = solver3d._properties(cfg, mat)
    Lap = solver3d.assemble_laplacian_3d(g, k3, th, solver3d.interface_resistance(cfg, g))
    dth = np.diff(th)
    V = 0.5 * (g.r_faces[1:] ** 2 - g.r_faces[:-1] ** 2)[None, :, None] * dth[None, None, :] * g.dz_c[:, None, None]
    A = (sp.diags((rho_cp * V).ravel() / cfg.dt) + 0.5 * Lap).tocsc()
    b = np.random.default_rng(0).random(A.shape[0])
    ok = _Solver(A, False, nth=len(dth), th_faces=th, periodic=False)
    assert ok.kind == SOLVER_STRUCTURED
    x_ok = ok.solve(b, np.zeros_like(b))

    def boom(*a, **k):
        raise RuntimeError("SUPERLU_MALLOC fails for buf in intCalloc()")
    monkeypatch.setattr(solver3d._StructuredPC, "__init__", boom)
    fb = _Solver(A, False, nth=len(dth), th_faces=th, periodic=False)
    assert fb.kind == SOLVER_ILU
    x_fb = fb.solve(b, np.zeros_like(b))
    assert np.linalg.norm(A @ x_fb - b) < 1e-8 * np.linalg.norm(b) and np.allclose(x_fb, x_ok, rtol=1e-7, atol=1e-9)
    monkeypatch.setattr(solver3d._StructuredPC, "__init__", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("other")))
    with pytest.raises(RuntimeError, match="other"):                      # unrelated errors are not swallowed
        _Solver(A, False, nth=len(dth), th_faces=th, periodic=False)
