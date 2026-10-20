"""CW transmission mode (docs/cw-transmission-spec.md, step 2): source semantics, configuration builder,
1-D Carslaw-Jaeger slab check of the far-face step response, linear-asymptote intercept L^2/(6 alpha),
energy conservation (run with `pytest`)."""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttr_sim import BottomStack, DEPTH_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, run_simulation  # noqa: E402
from ttr_sim.materials import COPPER, FUSED_SILICA  # noqa: E402
from ttr_sim.solver import build_grid  # noqa: E402
from ttr_sim.transmission import (  # noqa: E402
    PROBE_RADIUS, PUMP_SPOT_RADIUS, cw_config, run_step_response, slab_far_face_step, slab_intercept, slab_step_response,
)

L_ROD, R_CU = 500e-6, 40e-6


def adiabatic_rod(t_end, **kw):
    """Homogeneous copper rod with no shell at all (R_out = R_cu): the 1-D slab of the analytic solution."""
    return cw_config(t_end=t_end, homogeneous_copper=True, R_out=R_CU, L=L_ROD, R_cu=R_CU, **kw)


def far_face_reference(resp, cfg):
    """1-D slab far-face rise for the absorbed power spread over the rod cross-section."""
    q = resp.P_abs / (math.pi * cfg.geometry.R_cu ** 2)
    return slab_far_face_step(resp.t, q, cfg.geometry.L, cfg.copper), q


# ----------------------------------------------------------------------------- laser source
def test_cw_laser_is_a_unit_step_and_pulse_mode_is_unchanged():
    cw = Laser(tau_p=1e-7, mode="cw", power=10e-3, w=5e-6, beam="flat")
    assert cw.is_cw and cw.pulse_end == 0.0 and cw.discontinuities == (0.0,) and cw.t_center == 0.0
    assert cw.I0 == pytest.approx(10e-3 / (math.pi * 25e-12))
    np.testing.assert_array_equal(cw.f(np.array([-1e-9, 0.0, 1e-9, 1.0])), [0.0, 1.0, 1.0, 1.0])
    # step-averaged source: exact for a step straddling t = 0, 1 afterwards
    assert cw.f_mean(-1e-9, 3e-9) == pytest.approx(0.75)
    assert cw.f_mean(np.array([0.0, 5e-3]), np.array([1e-9, 6e-3])) == pytest.approx([1.0, 1.0])
    # the pulse path keeps its defaults and semantics
    sq = Laser(tau_p=1e-7, energy=1e-9)
    assert sq.mode == "pulse" and not sq.is_cw and sq.discontinuities == (0.0, 1e-7) and sq.pulse_end == 1e-7
    assert sq.I0 == pytest.approx(1e-9 / (0.5 * math.pi * (10e-6) ** 2 * 1e-7))
    assert Laser(tau_p=1e-7, profile="gaussian").discontinuities == ()
    assert sq.probe_w_far is None and sq.probe_w_far_eff == sq.probe_w


# ----------------------------------------------------------------------------- configuration builder
def test_cw_config_follows_spec_section_6():
    cfg = cw_config()
    la, n, g = cfg.laser, cfg.numerics, cfg.geometry
    assert la.is_cw and la.beam == "flat" and la.power == 10e-3 and la.reflectivity == 0.6
    assert la.w == pytest.approx(math.sqrt(6e-6 * 5e-6)) and PUMP_SPOT_RADIUS == pytest.approx(5.477e-6, rel=1e-3)
    assert la.probe_w == PROBE_RADIUS == 4e-6 and la.probe_w_far_eff == 4e-6
    assert g.L == 500e-6 and g.R_cu == 40e-6 and not cfg.bottom.enabled
    assert n.t_end_mode == "absolute" and cfg.t_end == 20e-3 and n.far_layer > 0
    assert cfg.voids == [] and cfg.z_total == g.L          # no void, no bottom stack: the domain ends at z = L
    grid = build_grid(cfg)
    assert grid.z_faces[-1] == pytest.approx(g.L)
    assert grid.r_faces[-1] > g.R_cu + 2.0 * g.R_cu          # "far" shell
    assert np.any(np.isclose(grid.r_faces, g.R_cu))          # face on the copper / silica interface
    # the far-face zone is resolved at dz
    assert np.max(np.diff(grid.z_faces)[-4:]) == pytest.approx(n.dz, rel=1e-6)
    with pytest.raises(ValueError):
        cw_config(outer="nowhere")
    with pytest.raises(NotImplementedError):
        cw_config(outer="neighbor_sink")
    with pytest.raises(ValueError):
        run_step_response(SimConfig(geometry=Geometry(), void=VoidSpec(enabled=False), laser=Laser(tau_p=1e-7), numerics=Numerics(dz=1e-6)))


def test_adiabatic_rod_has_no_shell():
    cfg = adiabatic_rod(1e-3)
    grid = build_grid(cfg)
    assert grid.r_faces[-1] == pytest.approx(R_CU)
    assert cfg.diagnostics()["warnings"] == []          # no back-face warning in CW mode


# ----------------------------------------------------------------------------- 1-D analytic reference
def test_slab_solution_limits():
    t = np.array([0.0, 1e-4, 1e-3, 1e-2])
    q, L = 1e6, L_ROD
    far = slab_far_face_step(t, q, L)
    assert far[0] == 0.0 and np.all(np.diff(far) > 0)
    # long-time asymptote (q / (rho c L)) (t - L^2 / (6 alpha))
    asym = q / (COPPER.rho_cp * L) * (t[-1] - slab_intercept(L))
    assert far[-1] == pytest.approx(asym, rel=1e-9)
    # the mean temperature follows the absorbed energy exactly (insulated slab)
    z = np.linspace(0.0, L, 2001)
    for ti in t[1:]:
        mean = np.trapezoid(slab_step_response(z, ti, q, L), z) / L
        assert mean == pytest.approx(q * ti / (COPPER.rho_cp * L), rel=1e-6)
    # early time: the pump face behaves as a semi-infinite body, 2 q sqrt(alpha t / pi) / k
    t_early = 1e-6
    semi = 2.0 * q * math.sqrt(COPPER.alpha * t_early / math.pi) / COPPER.k
    assert slab_step_response(0.0, t_early, q, L, n_terms=4000) == pytest.approx(semi, rel=1e-3)


# ----------------------------------------------------------------------------- step response vs Carslaw-Jaeger
@pytest.fixture(scope="module")
def short_run():
    cfg = adiabatic_rod(1e-3)
    return cfg, run_step_response(cfg)


@pytest.fixture(scope="module")
def long_run():
    cfg = adiabatic_rod(3e-3)
    return cfg, run_step_response(cfg)


def test_far_face_matches_slab_solution_after_arrival(short_run):
    cfg, resp = short_run
    assert resp.t[-1] == pytest.approx(1e-3)
    ana, q = far_face_reference(resp, cfg)
    assert resp.P_abs == pytest.approx(0.4 * 10e-3)
    # "heat arrival": the far face has reached 1 % of its end-of-window rise
    i0 = int(np.searchsorted(ana, 0.01 * ana[-1]))
    num, ref = resp.dT_far[i0:], ana[i0:]
    rms = math.sqrt(np.mean((num - ref) ** 2)) / math.sqrt(np.mean(ref ** 2))
    assert rms < 0.02, f"relative RMS {rms:.2%}"
    assert np.max(np.abs(num - ref) / ref) < 0.03
    # the pump face is always hotter than the far face; both rise monotonically
    assert np.all(resp.dT_pump[1:] > resp.dT_far[1:]) and np.all(np.diff(resp.dT_far) >= 0)
    assert np.allclose(resp.dT_pump - resp.dT_far, resp.dT_diff)


def test_linear_asymptote_intercept_is_L2_over_6alpha(long_run):
    cfg, resp = long_run
    L, alpha = cfg.geometry.L, cfg.copper.alpha
    ana, q = far_face_reference(resp, cfg)
    # fit the last 40 % of the window: the slowest mode exp(-pi^2 alpha t / L^2) is < 1e-3 there
    sel = resp.t >= 0.6 * resp.t[-1]
    assert math.exp(-math.pi ** 2 * alpha * resp.t[sel][0] / L ** 2) < 1e-3
    slope, offset = np.polyfit(resp.t[sel], resp.dT_far[sel], 1)
    t_int = -offset / slope
    t_exp = slab_intercept(L, cfg.copper)
    assert t_exp == pytest.approx(0.361e-3, rel=0.01)                  # ~0.36 ms for copper (spec §1)
    assert abs(t_int / t_exp - 1.0) < 0.03, f"intercept {t_int * 1e3:.4f} ms vs {t_exp * 1e3:.4f} ms"
    assert slope == pytest.approx(q / (cfg.copper.rho_cp * L), rel=0.01)   # dT/dt = P_abs / (rho c V)
    # and the full curve still agrees with the series solution
    i0 = int(np.searchsorted(ana, 0.01 * ana[-1]))
    rms = math.sqrt(np.mean((resp.dT_far[i0:] - ana[i0:]) ** 2)) / math.sqrt(np.mean(ana[i0:] ** 2))
    assert rms < 0.02


def test_energy_conservation(short_run, long_run):
    for cfg, resp in (short_run, long_run):
        assert abs(resp.energy_error) < 1e-6 and resp.energy_error_max < 1e-6
        assert resp.E_in[-1] == pytest.approx(resp.P_abs * resp.t[-1], rel=1e-9)
        assert np.allclose(resp.E_stored, resp.E_in, rtol=1e-6, atol=1e-6 * resp.E_in[-1])


def test_default_configuration_with_silica_shell_runs_and_conserves_energy():
    cfg = cw_config()
    resp = run_step_response(cfg, store_fields=False)
    assert resp.t[-1] == pytest.approx(20e-3) and resp.warnings == []
    assert abs(resp.energy_error) < 1e-6 and resp.energy_error_max < 1e-6
    assert resp.P_abs == pytest.approx(4e-3) and resp.P_in == 10e-3
    assert resp.dT_pump[-1] > resp.dT_far[-1] > 0.0
    assert np.all(resp.dT_pump[1:] > resp.dT_far[1:])
    # the silica shell takes heat: both faces end up cooler than in the adiabatic rod (far face by tens of percent)
    adia = run_step_response(adiabatic_rod(20e-3), store_fields=False)
    assert resp.dT_far[-1] < 0.9 * adia.dT_far[-1] and resp.dT_pump[-1] < adia.dT_pump[-1]
    # the thermoreflectance signals follow c_tr
    assert np.allclose(resp.dRR_far, cfg.laser.c_tr * resp.dT_far)
    # a shell too thin for the window is reported
    thin = run_step_response(cw_config(t_end=20e-3, R_out=R_CU + 50e-6), store_fields=False)
    assert any("far" in w for w in thin.warnings)


# ----------------------------------------------------------------------------- far-face reconstruction in reflection mode
def test_far_face_is_recorded_for_pulse_runs_too():
    p = DEPTH_PRESETS[2]
    laser = Laser(tau_p=p.tau_p, energy=10e-9, w=10e-6, probe_w=5e-6)
    base = SimConfig(geometry=Geometry(), void=VoidSpec(enabled=False), laser=laser, numerics=Numerics(dz=p.dz))
    # adiabatic back face: the reconstructed face temperature is exactly the last cell row
    adia = run_simulation(replace(base, bottom=BottomStack(enabled=False)))
    assert adia.T_far.shape == (len(adia.times), adia.grid.nr) and adia.dT_probe_far is not None
    np.testing.assert_array_equal(adia.T_far[-1], adia.snapshots[-1][1][-1])
    assert adia.dT_probe_far.max() <= adia.dT_probe.max()
    # bottom stack with a heat sink: z = L is an interior face, reconstructed from the link flux; finite and bounded
    sink = run_simulation(replace(base, bottom=BottomStack(enabled=True, sink="isothermal")))
    assert np.all(np.isfinite(sink.T_far)) and sink.dT_probe_far.min() >= -1e-12
    assert sink.dT_probe_far.max() <= sink.dT_probe.max()
