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
    cfg = make_cfg(2, fo=0.5)
    assert COPPER.alpha * cfg.dt / cfg.numerics.dz ** 2 == pytest.approx(0.5)
    assert cfg.n_steps == pytest.approx(1000, abs=10)  # 5 tau_p / (tau_p/200)


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
