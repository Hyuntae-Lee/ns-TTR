"""Verification tests for the ns-TTR simulator core (run with `pytest`)."""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttr_sim import DEPTH_PRESETS, KVOID_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, build_grid, run_simulation  # noqa: E402
from ttr_sim.analytic import center_step_response, surface_step_response  # noqa: E402
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
    assert g.z_faces[-1] == pytest.approx(cfg.geometry.L)
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
    assert g.z_faces[-1] == pytest.approx(cfg.geometry.L)


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
    # absorbed energy = E (1-R) * fraction of the Gaussian inside the rod
    la = res.config.laser
    frac = 1 - math.exp(-2 * res.config.geometry.R_cu ** 2 / la.w ** 2)
    assert res.E_in[-1] == pytest.approx(la.energy * (1 - la.reflectivity) * frac, rel=1e-9)


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
