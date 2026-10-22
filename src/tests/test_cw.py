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
    with pytest.raises(ValueError):
        cw_config(outer="neighbor_sink")          # pitch missing
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


# ============================================================================= step 3: analysis methods
from ttr_sim import analysis as an  # noqa: E402
from ttr_sim.transmission import frequency_response, lockin_dataset, penetration_depth  # noqa: E402

A_ROD = math.pi * R_CU ** 2
RHO_CP = COPPER.rho_cp
FREQS = np.array([20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])


def test_analysis_methods_recover_copper_on_the_lossless_rod(long_run):
    cfg, resp = long_run
    k, a = cfg.copper.k, cfg.laser.w
    si = an.step_intercept(resp.t, resp.dT_far, L_ROD, RHO_CP, resp.P_abs, A_ROD)        # tangent at maximum slope
    assert si["t_int"] == pytest.approx(slab_intercept(L_ROD), rel=0.01)
    assert si["k_eff"] == pytest.approx(k, rel=0.01) and si["rho_cp_eff"] == pytest.approx(RHO_CP, rel=0.01)
    si_w = an.step_intercept(resp.t, resp.dT_far, L_ROD, RHO_CP, t_from=0.6 * resp.t[-1])   # explicit fit window
    assert si_w["k_eff"] == pytest.approx(k, rel=0.01)
    pk = an.parker(resp.t, resp.dT_far, L_ROD, RHO_CP)
    assert pk["t_half"] == pytest.approx(an.PARKER_CONSTANT * L_ROD ** 2 / cfg.copper.alpha, rel=0.02)
    assert pk["k_eff"] == pytest.approx(k, rel=0.02)
    # two-face difference of an adiabatic rod: quasi-steady P L / (2 k A) plus the spot spreading ~ 1 / (4 k a)
    tf = an.two_face_resistance(resp.t, resp.dT_pump, resp.dT_far, resp.P_abs, L_ROD, A_ROD, spot_radius=a)
    assert tf["R_th"] == pytest.approx(tf["R_quasi_steady"] + tf["R_spread"], rel=0.05)
    assert tf["R_through"] == pytest.approx(2.0 * tf["R_quasi_steady"]) and "k_eff" not in tf    # indicator only


def test_tlm_and_lockin_on_the_lossless_rod(long_run):
    cfg, resp = long_run
    k = cfg.copper.k
    Hp, Hf = an.tlm_response(FREQS, k, RHO_CP, L_ROD, A_ROD)
    # low-frequency limit of the far face: the whole rod heats as a lumped capacity, |H| -> 1 / (omega rho c A L)
    w = 2.0 * math.pi * FREQS[0]
    assert abs(Hf[0]) == pytest.approx(1.0 / (w * RHO_CP * A_ROD * L_ROD), rel=0.05)
    assert np.all(np.diff(np.abs(Hf)) < 0) and np.all(np.degrees(np.angle(Hf[:3])) < -90.0)
    # the step response (lossless: constant final slope, linear tail exact) gives the same H
    Hs = an.frequency_response_from_step(resp.t, resp.dT_far, FREQS, resp.P_abs)
    np.testing.assert_allclose(np.abs(Hs), np.abs(Hf), rtol=0.01)
    assert np.max(np.abs(np.degrees(np.angle(Hs / Hf)))) < 1.0
    # inversion: exact data -> exact k; step data -> k within 1 %; phase-only (default) is blind to a gain factor
    assert an.invert_tlm(FREQS, Hf, RHO_CP, L_ROD, A_ROD)["k_eff"] == pytest.approx(k, rel=1e-6)
    assert an.invert_tlm(FREQS, 0.37 * Hf, RHO_CP, L_ROD, A_ROD)["k_eff"] == pytest.approx(k, rel=1e-6)   # R, c_tr do not matter
    assert an.invert_tlm(FREQS, Hf, RHO_CP, L_ROD, A_ROD, use="amplitude")["k_eff"] == pytest.approx(k, rel=1e-6)
    assert an.invert_tlm(FREQS, Hf, RHO_CP, L_ROD, A_ROD, use="both")["k_eff"] == pytest.approx(k, rel=1e-6)
    assert an.invert_tlm(FREQS, Hs, RHO_CP, L_ROD, A_ROD)["k_eff"] == pytest.approx(k, rel=0.01)
    assert an.invert_tlm(FREQS, Hs, RHO_CP, L_ROD, A_ROD, fit_loss=True)["k_eff"] == pytest.approx(k, rel=0.01)
    with pytest.raises(ValueError):
        an.invert_tlm(FREQS, Hf, RHO_CP, L_ROD, A_ROD, use="magic")
    # lock-in reading of a square-chopped pump: fundamental 2 P / pi
    lk = an.lockin(Hf, resp.P_abs)
    np.testing.assert_allclose(lk["amplitude"], 2.0 / math.pi * resp.P_abs * np.abs(Hf))
    np.testing.assert_allclose(lk["phase_deg"], np.degrees(np.angle(Hf)))
    assert an.lockin(Hf, resp.P_abs, chop="sine")["amplitude"][0] == pytest.approx(0.5 * resp.P_abs * abs(Hf[0]))


def test_frequency_domain_solve_matches_tlm_and_step_fft(long_run):
    cfg, resp = long_run
    fr = frequency_response(cfg, FREQS)
    Hp, Hf = an.tlm_response(FREQS, cfg.copper.k, RHO_CP, L_ROD, A_ROD)
    np.testing.assert_allclose(np.abs(fr.H_far), np.abs(Hf), rtol=0.005)
    assert np.max(np.abs(np.degrees(np.angle(fr.H_far / Hf)))) < 0.5
    assert fr.P_abs == pytest.approx(resp.P_abs)
    # pump face: spreading of the small spot makes it much hotter than the 1-D line predicts, lag smaller
    assert np.all(np.abs(fr.H_pump) > np.abs(Hp)) and np.all(np.abs(fr.H_pump) > np.abs(fr.H_far))
    # cross-validation with the time integrator (spec §2.4 / §5.2)
    Hs = an.frequency_response_from_step(resp.t, resp.dT_far, FREQS, resp.P_abs)
    np.testing.assert_allclose(np.abs(Hs), np.abs(fr.H_far), rtol=0.01)
    assert np.max(np.abs(np.degrees(np.angle(Hs / fr.H_far)))) < 1.0


def test_glass_shell_model_and_analyze_on_the_real_structure():
    cfg = cw_config()
    resp = run_step_response(cfg, store_fields=False)
    fr = frequency_response(cfg, FREQS)
    shell = (R_CU, FUSED_SILICA)
    # the rod-in-glass TLM reproduces the 2-D solve (axial conduction in the glass and the spot are the rest)
    _, Hsh = an.tlm_response(FREQS, cfg.copper.k, RHO_CP, L_ROD, A_ROD, shell=shell)
    np.testing.assert_allclose(np.abs(fr.H_far), np.abs(Hsh), rtol=0.03)
    assert np.max(np.abs(np.degrees(np.angle(fr.H_far / Hsh)))) < 2.0
    # the lateral admittance falls towards DC (only logarithmically: a 2-D medium has no steady state, spec §1)
    # and the glass makes the far face colder than the bare rod
    Y = an.lateral_admittance(np.array([1e-3, 1.0, 1e3]), R_CU, FUSED_SILICA)
    assert abs(Y[0]) < abs(Y[1]) < 0.3 * abs(Y[2]) and np.all(np.real(Y) > 0)
    assert abs(Y[0]) == pytest.approx(2.0 * math.pi * FUSED_SILICA.k / math.log(1.0 / abs(np.sqrt(2j * math.pi * 1e-3 / FUSED_SILICA.alpha) * R_CU)), rel=0.3)
    _, Hbare = an.tlm_response(FREQS, cfg.copper.k, RHO_CP, L_ROD, A_ROD)
    assert np.all(np.abs(Hsh) < np.abs(Hbare))
    # step FFT agrees with the direct solve where the window resolves the frequency
    Hs = an.frequency_response_from_step(resp.t, resp.dT_far, FREQS, resp.P_abs)
    mid = (FREQS >= 50.0) & (FREQS <= 500.0)
    np.testing.assert_allclose(np.abs(Hs[mid]), np.abs(fr.H_far[mid]), rtol=0.02)
    # the deliverable: the phase-based glass-shell inversion recovers copper; the lossless models are biased
    res = an.analyze(resp.t, resp.dT_pump, resp.dT_far, resp.P_abs, L_ROD, A_ROD, RHO_CP, spot_radius=cfg.laser.w,
                     freqs=FREQS, H_far=fr.H_far, shell=shell)
    kt = an.k_eff_table(res)
    assert set(res) == {"step_intercept", "parker", "two_face", "tlm_shell"} == set(an.METHOD_LABELS)
    assert kt["tlm_shell"] == pytest.approx(cfg.copper.k, rel=0.03) and res["tlm_shell"]["use"] == "phase"
    k_adia = an.invert_tlm(FREQS, fr.H_far, RHO_CP, L_ROD, A_ROD, use="both")["k_eff"]   # adiabatic-rod TLM: biased
    assert abs(k_adia / cfg.copper.k - 1.0) > 0.03
    assert kt["step_intercept"] > 1.1 * cfg.copper.k and kt["parker"] > 1.1 * cfg.copper.k   # faster apparent rise
    assert math.isnan(kt["two_face"])                             # indicator only, no k
    assert res["step_intercept"]["rho_cp_eff"] > 1.5 * RHO_CP     # the glass shares the heat
    # the same inversion on the recorded transient (step FFT) and on the lock-in data (direct solve) agree
    k_direct = an.invert_tlm(FREQS, fr.H_far, RHO_CP, L_ROD, A_ROD, shell=shell)["k_eff"]
    k_step = an.invert_tlm(FREQS, Hs, RHO_CP, L_ROD, A_ROD, shell=shell)["k_eff"]
    assert k_step == pytest.approx(k_direct, rel=0.02)
    # baseline-vs-"void" comparison bookkeeping
    cmp = an.compare(res, res)
    assert all(row[2] == 0.0 for rows in cmp.values() for row in rows.values())
    vi = an.void_indicators(res, res)
    assert set(vi) == set(res) and all(v[2] == 0.0 for v in vi.values())


def test_lockin_dataset_switches_source_by_frequency():
    cfg = cw_config(f_min=10.0)
    resp = run_step_response(cfg, store_fields=False)
    freqs = np.array([10.0, 20.0, 50.0, 100.0, 500.0, 2000.0])
    fr = frequency_response(cfg, freqs)
    assert fr.warnings == []                                      # f_min widened the shell: 3 penetration depths fit
    assert build_grid(cfg).r_faces[-1] - R_CU >= 3.0 * penetration_depth(10.0) * (1 - 1e-9)
    ds = lockin_dataset(resp, fr)
    assert ds.f_switch == pytest.approx(1.0 / resp.t[-1])
    assert list(ds.source) == ["direct"] * 2 + ["step_fft"] * 4
    np.testing.assert_array_equal(ds.H_far[:2], fr.H_far[:2])
    np.testing.assert_allclose(np.abs(ds.H_far[2:]), np.abs(fr.H_far[2:]), rtol=0.03)
    assert list(lockin_dataset(resp, fr, prefer="direct").source) == ["direct"] * 6
    assert list(lockin_dataset(resp, None, freqs).source) == ["step_fft"] * 6
    # the deliverable on the mixed dataset
    k = an.invert_tlm(ds.freqs, ds.H_far, RHO_CP, L_ROD, A_ROD, shell=(R_CU, FUSED_SILICA))["k_eff"]
    assert k == pytest.approx(COPPER.k, rel=0.03)
    # a too-thin shell for the lowest frequency is reported by the frequency solve
    thin = frequency_response(cw_config(t_end=5e-3), [1.0])
    assert any("침투" in w for w in thin.warnings)


def test_phase_noise_sensitivity_grows_with_sigma():
    _, Hf = an.tlm_response(FREQS, COPPER.k, RHO_CP, L_ROD, A_ROD, shell=(R_CU, FUSED_SILICA))
    out = an.phase_noise_sensitivity(FREQS, Hf, RHO_CP, L_ROD, A_ROD, (R_CU, FUSED_SILICA), [0.1, 0.5], n_trials=40, seed=1)
    assert out["k_noise_free"] == pytest.approx(COPPER.k, rel=1e-6)
    r01, r05 = out["rows"]
    assert 0.0 < r01["std_rel"] < r05["std_rel"] < 0.2
    assert abs(r01["bias_rel"]) < 3 * r01["std_rel"] / math.sqrt(40) + 1e-3


# ============================================================================= step 5: outer boundary options
PITCH = 130e-6      # glass gap 50 um between vias (spec §1 table)


def test_neighbour_boundary_options():
    sink = cw_config(outer="neighbor_sink", pitch=PITCH)
    adia = cw_config(outer="neighbor_adiabatic", pitch=PITCH)
    assert sink.geometry.R_out == pytest.approx(PITCH - R_CU) and sink.geometry.outer_bc == "isothermal"
    assert adia.geometry.R_out == pytest.approx(0.5 * PITCH) and adia.geometry.outer_bc == "adiabatic"
    assert build_grid(sink).r_faces[-1] == pytest.approx(PITCH - R_CU)
    for outer in ("neighbor_sink", "neighbor_adiabatic"):
        with pytest.raises(ValueError):
            cw_config(outer=outer)                       # pitch missing
        with pytest.raises(ValueError):
            cw_config(outer=outer, pitch=1.5 * R_CU)     # vias would overlap
    with pytest.raises(ValueError):
        build_grid(replace(sink, geometry=replace(sink.geometry, outer_bc="mirror")))
        run_simulation(replace(sink, geometry=replace(sink.geometry, outer_bc="mirror")))


@pytest.fixture(scope="module")
def boundary_runs():
    out = {}
    for outer in ("far", "neighbor_sink", "neighbor_adiabatic"):
        cfg = cw_config(outer=outer, pitch=PITCH, t_end=10e-3)
        out[outer] = (cfg, run_step_response(cfg, store_fields=False))
    return out


def test_side_sink_energy_balance_and_bounds(boundary_runs):
    far, sink, adia = (boundary_runs[k][1] for k in ("far", "neighbor_sink", "neighbor_adiabatic"))
    # stored + released = absorbed, every step, with the isothermal side
    assert abs(sink.energy_error) < 1e-6 and sink.energy_error_max < 1e-6
    assert sink.E_out[-1] > 0.2 * sink.E_in[-1]                       # a good part of the heat has left through the neighbours
    assert far.E_out[-1] == 0.0 and adia.E_out[-1] == 0.0
    assert np.allclose(sink.E_stored + sink.E_out, sink.E_in, rtol=1e-6, atol=1e-6 * sink.E_in[-1])
    # the neighbour options bracket the "far" result at late times: sink (lower bound) < far < adiabatic cell (upper bound)
    assert sink.dT_far[-1] < far.dT_far[-1] < adia.dT_far[-1]
    assert sink.dT_pump[-1] < far.dT_pump[-1] < adia.dT_pump[-1]
    # before the heat reaches the glass gap all three agree
    early = far.t < 20e-6
    assert np.allclose(sink.dT_far[early], far.dT_far[early], rtol=1e-3, atol=1e-9)
    assert np.allclose(adia.dT_pump[early], far.dT_pump[early], rtol=1e-3, atol=1e-9)
    assert sink.warnings == []                                        # the "far" warning is not raised for the neighbour boundaries
    # the neighbour sink gives a true steady state: the far face flattens and matches the DC of the frequency solve
    cfg_sink = boundary_runs["neighbor_sink"][0]
    long = run_step_response(replace(cfg_sink, numerics=replace(cfg_sink.numerics, t_end_abs=40e-3)), store_fields=False)
    H0 = frequency_response(cfg_sink, [1e-3]).H_far[0]
    assert abs(H0.imag) < 1e-3 * abs(H0.real)
    assert long.dT_far[-1] == pytest.approx(long.P_abs * H0.real, rel=0.01)
    assert long.dT_far[-1] - long.dT_far[-5] < 1e-3 * long.dT_far[-1]


def test_lateral_loss_biases_the_methods_in_order(boundary_runs):
    """Each method's k_eff must come out ordered like the lateral loss: the more heat leaves sideways, the more
    the (lossless-model) methods are biased.  The shell-model TLM is the least biased on 'far'."""
    freqs = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
    k = {}
    for outer, (cfg, resp) in boundary_runs.items():
        fr = frequency_response(cfg, freqs)
        res = an.analyze(resp.t, resp.dT_pump, resp.dT_far, resp.P_abs, L_ROD, A_ROD, RHO_CP, spot_radius=cfg.laser.w,
                         freqs=freqs, H_far=fr.H_far, shell=(R_CU, FUSED_SILICA))
        k[outer] = an.k_eff_table(res)
    # the deliverable (infinite-glass model) is within 3 % on the "far" boundary; the neighbour bounds bend the
    # low-frequency phase (the glass is not infinite there) and stay within the documented 10 %
    assert k["far"]["tlm_shell"] == pytest.approx(COPPER.k, rel=0.03)
    assert k["neighbor_sink"]["tlm_shell"] == pytest.approx(COPPER.k, rel=0.10)
    assert k["neighbor_adiabatic"]["tlm_shell"] == pytest.approx(COPPER.k, rel=0.25)   # closed thin annulus: no infinite glass at all
    for m in ("step_intercept", "parker"):             # faster apparent rise reads as a higher apparent alpha ...
        assert all(k[b][m] > 1.02 * COPPER.k for b in ("far", "neighbor_sink"))
        assert k["neighbor_adiabatic"][m] < COPPER.k   # ... unless a thin closed glass annulus acts as extra heat capacity
    # the boundary matters for the indicators (the spread between the two neighbour bounds is the array sensitivity)
    for m in ("step_intercept", "parker"):
        assert k["neighbor_sink"][m] != pytest.approx(k["neighbor_adiabatic"][m], rel=0.005)


# ============================================================================= step 7: 3-D spot shape / far face
from ttr_sim.solver3d import count_unknowns, needs_3d, spot_sector_weights, theta_faces  # noqa: E402
from ttr_sim.solver import beam_annulus_weights  # noqa: E402

T3D = 1e-3      # short window and a 5 um grid: the 3-D runs stay at ~30 s under direct LU
DZ3 = 5e-6


def test_spot_sector_weights_reproduce_the_circular_annulus_weights():
    cfg = cw_config(t_end=T3D, dz=DZ3, spot_offset=1e-12)                  # circular spot, forced non-axisymmetric
    assert needs_3d(cfg) and not cfg.laser.axisymmetric and cfg.laser.w_b is None
    g = build_grid(cfg)
    th = theta_faces(cfg)
    assert th[0] == 0.0 and th[-1] == pytest.approx(math.pi) and len(th) - 1 == cfg.numerics.spot_theta_cells
    W = spot_sector_weights(g, th, cfg.laser, cfg.geometry.R_cu)
    Wa = beam_annulus_weights(g.r_faces, cfg.laser, cfg.geometry.R_cu)
    assert W.sum() == pytest.approx(0.5 * Wa.sum(), rel=2e-3)        # half circle
    np.testing.assert_allclose(W.sum(axis=1), 0.5 * Wa, rtol=5e-3, atol=1e-3 * Wa.max())
    # an ellipse of the same area absorbs the same power; an offset moves the weight to theta = 0
    ce = cw_config(t_end=T3D, dz=DZ3, spot_axes=(12e-6, 10e-6))
    assert ce.laser.is_elliptical and ce.laser.beam_area == pytest.approx(math.pi * 6e-6 * 5e-6)
    We = spot_sector_weights(g, th, ce.laser, ce.geometry.R_cu)
    assert We.sum() == pytest.approx(0.5 * ce.laser.beam_area, rel=2e-3)
    nth = len(th) - 1
    assert We[:, 0].sum() > 1.1 * We[:, nth // 2].sum()                # long axis along theta = 0, short along pi/2
    assert We[:, 0].sum() == pytest.approx(We[:, -1].sum())           # mirror symmetric about the y axis
    co = cw_config(t_end=T3D, dz=DZ3, spot_offset=3e-6)
    Wo = spot_sector_weights(g, th, co.laser, co.geometry.R_cu)
    assert Wo.sum() == pytest.approx(0.5 * co.laser.beam_area, rel=2e-3) and Wo[:, 0].sum() > Wo[:, -1].sum()
    with pytest.raises(NotImplementedError):
        run_step_response(replace(co, laser=replace(co.laser, beam="gaussian")))


@pytest.fixture(scope="module")
def spot_runs():
    c2 = cw_config(t_end=T3D, dz=DZ3)
    out = {"2d": run_step_response(c2, store_fields=False)}
    out["3d_circle"] = run_step_response(cw_config(t_end=T3D, dz=DZ3, spot_offset=1e-12), store_fields=False)
    out["ellipse"] = run_step_response(cw_config(t_end=T3D, dz=DZ3, spot_axes=(12e-6, 10e-6)), store_fields=False)
    return out


def test_3d_circular_spot_reproduces_the_2d_solution(spot_runs):
    r2, r3 = spot_runs["2d"], spot_runs["3d_circle"]
    assert r3.result.is_3d and not r2.result.is_3d and abs(r3.energy_error) < 1e-6
    assert r3.P_abs == pytest.approx(r2.P_abs, rel=2e-3)
    assert np.allclose(r3.t, r2.t)
    np.testing.assert_allclose(r3.dT_pump[5:], r2.dT_pump[5:], rtol=0.01)
    far_scale = r2.dT_far[-1]
    assert np.max(np.abs(r3.dT_far - r2.dT_far)) < 0.01 * far_scale
    assert r3.result.T_far.shape == (len(r3.t), 2 * r3.result.grid.nr) and r3.result.dT_probe_far is not None
    np.testing.assert_allclose(r3.dT_far, r3.result.dT_probe_far)


def test_elliptical_spot_changes_the_pump_face_only(spot_runs):
    r2, re = spot_runs["2d"], spot_runs["ellipse"]
    assert re.result.is_3d and re.P_abs == pytest.approx(r2.P_abs, rel=2e-3)
    i = int(np.searchsorted(re.t, 1e-4))
    pump_dev = abs(re.dT_pump[i] / r2.dT_pump[i] - 1.0)
    assert 0.002 < pump_dev < 0.08                                   # a few percent on the pump face (spec section 1 table)
    far_dev = np.max(np.abs(re.dT_far - r2.dT_far)) / r2.dT_far[-1]
    assert far_dev < 0.005                                           # negligible on the far face (radial equilibration)


def test_3d_far_face_of_an_on_axis_void_matches_2d():
    v = VoidSpec(enabled=True, depth=245e-6, thickness=10e-6, r_half=20e-6, k=0.026)
    kw = dict(t_end=T3D, dz=DZ3, void=v, void_cells=4, theta_max=math.pi / 6)
    r2 = run_step_response(cw_config(**kw), store_fields=False)
    r3 = run_step_response(cw_config(spot_offset=1e-12, **kw), store_fields=False)
    assert r3.result.is_3d and abs(r3.energy_error) < 1e-6
    assert np.max(np.abs(r3.dT_far - r2.dT_far)) < 0.01 * r2.dT_far[-1]
    np.testing.assert_allclose(r3.dT_pump[5:], r2.dT_pump[5:], rtol=0.01)


def test_frequency_solve_uses_the_equivalent_circle_for_3d_spots():
    ce = cw_config(t_end=T3D, dz=DZ3, spot_axes=(12e-6, 10e-6))
    f = np.array([100.0, 1000.0])
    fe = frequency_response(ce, f)
    fc = frequency_response(cw_config(t_end=T3D, dz=DZ3), f)
    np.testing.assert_allclose(fe.H_far, fc.H_far, rtol=2e-3)       # same spot area; the surface-layer grid differs slightly
    with pytest.raises(NotImplementedError):
        frequency_response(cw_config(t_end=T3D, void=VoidSpec(enabled=True, depth=100e-6, thickness=10e-6, r_half=10e-6, r_center=20e-6)), f)


# ============================================================================= off-axis / multiple voids in CW mode
def test_off_axis_void_runs_in_3d_and_inverts_from_the_step_fft():
    v = VoidSpec(enabled=True, depth=245e-6, thickness=10e-6, r_half=10e-6, r_center=20e-6, k=0.026)
    kw = dict(t_end=T3D, dz=DZ3, void_cells=3, theta_max=math.pi / 6)
    cfg = cw_config(void=v, **kw)
    assert needs_3d(cfg)
    resp = run_step_response(cfg, store_fields=False)
    assert resp.result.is_3d and abs(resp.energy_error) < 1e-6
    assert np.all(np.isfinite(resp.dT_far)) and resp.dT_far[-1] > 0 and resp.dT_pump[-1] > resp.dT_far[-1]
    # the frequency solve is axisymmetric only: the lock-in data come from the step FFT
    freqs = np.array([2000.0, 5000.0])                   # >= 1 / window of this short run
    with pytest.raises(NotImplementedError):
        frequency_response(cfg, freqs)
    ds = lockin_dataset(resp, None, freqs)
    assert list(ds.source) == ["step_fft", "step_fft"]
    base = run_step_response(cw_config(**kw), store_fields=False)
    db = lockin_dataset(base, None, freqs)
    shell = (R_CU, FUSED_SILICA)
    k_v = an.invert_tlm(ds.freqs, ds.H_far, RHO_CP, L_ROD, A_ROD, shell=shell)["k_eff"]
    k_b = an.invert_tlm(db.freqs, db.H_far, RHO_CP, L_ROD, A_ROD, shell=shell)["k_eff"]
    assert k_b == pytest.approx(COPPER.k, rel=0.10)                 # short window, two high frequencies: coarse but sane
    assert 0.90 < k_v / k_b < 1.0                                   # the void lowers the apparent k by a few percent
    # two voids: one on the axis and one off it -> full circle 3-D; both faces still finite and consistent
    v2 = VoidSpec(enabled=True, depth=100e-6, thickness=10e-6, r_half=10e-6, r_center=0.0, k=0.026)
    cfg2 = cw_config(void=v2, extra_voids=(v,), **kw)
    assert needs_3d(cfg2) and len(cfg2.voids) == 2
    r2 = run_step_response(cfg2, store_fields=False)
    assert r2.result.is_3d and r2.result.diagnostics["full_circle"] and abs(r2.energy_error) < 1e-6
    assert r2.dT_far[-1] < resp.dT_far[-1] * 1.001 and r2.dT_pump[-1] > resp.dT_pump[-1] * 0.999
