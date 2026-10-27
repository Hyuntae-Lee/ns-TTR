"""CW void-detection study, stage 1 (docs/cw-void-detection-study-plan.md §7): single on-axis void, 2-D.

Series A: sphere-size ladder at mid-rod (centre depth 250 um) -> reference void (the diameter whose Δk/k is
2 x the 3σ_k detection limit, interpolated) -> series B (depth of the reference sphere) and series C (shape at
equal volume: sphere / disc 1:4 / thin disc 1:10 / rod 4:1).  Every void case runs together with the void-free
via on the SAME grid (solver.baseline_config) so that the void signal is free of discretisation offsets.

Outputs (results/cw_void_detect):
    cases/<id>.json   every number of plan §4 for one case      cases/<id>.npz   curves (void and reference)
    cases.csv         one row per case                           run.log          timing
    interim.md        stage-1 interim report (plan §9)           figs/*.png
Finished cases are not recomputed (cache by id).  Run from src:   .venv\\Scripts\\python.exe studies\\cw_void_detect.py
Nothing in the simulator or the GUIs is modified by this script.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(SRC))

from ttr_sim import VoidSpec  # noqa: E402
from ttr_sim import analysis as an  # noqa: E402
from ttr_sim.materials import COPPER, FUSED_SILICA  # noqa: E402
from ttr_sim.solver import baseline_config  # noqa: E402
from ttr_sim.solver3d import needs_3d  # noqa: E402
from ttr_sim.transmission import cw_config, frequency_response, lockin_dataset, run_step_response  # noqa: E402

OUT = SRC / "results" / "cw_void_detect"
CASES_DIR = OUT / "cases"
UM = 1e-6
R_CU, L = 40 * UM, 500 * UM
A = math.pi * R_CU ** 2
SHELL = (R_CU, FUSED_SILICA)
T_END = 20e-3                     # observation window = t*
DZ = 2.5e-6
F_MIN = 50.0
FREQS = np.logspace(math.log10(50), math.log10(2000), 9)
K_VOID = 0.026                    # air
Z_REF_UM = 250.0                  # centre depth of series A and C
LADDER_UM = [5, 8, 12, 16, 20, 25, 30, 40]        # plan §7; extended automatically until the target Δk/k is bracketed
LADDER_EXT_UM = [50, 60, 70]
DEPTHS_UM = [25, 100, 250, 400, 475]
SHAPES = {"sphere": 1.0, "disc_1_4": 0.25, "disc_1_10": 0.10, "rod_4_1": 4.0}     # aspect = thickness / diameter
SHAPE_LABEL = {"sphere": "구 (1:1)", "disc_1_4": "원판 1:4", "disc_1_10": "얇은 원판 1:10", "rod_4_1": "막대 4:1",
               "disc_1_10_r38": "얇은 원판 1:10 (r_half 38 µm, 부피 ½)"}
R_HALF_MAX = R_CU - 2 * UM        # keep one copper cell ring between void and glass
# detection-criterion constants (plan §6; assumptions, stated in the report)
SIGMA_PHI_DEG = 0.5               # lock-in phase noise
SIGMA_T_MK = 6.7                  # ΔT noise per transient (ΔR/R 1e-6, c_tr 1.5e-4)
AMP_DEV_PCT = 1.0                 # via-to-via absorption spread (unfounded assumption, "R-dependent")
N_SIGMA = 3.0
FACES = {"pump": "펌프면", "far": "반대면", "diff": "양면 차"}


# ----------------------------------------------------------------------------- geometry helpers
def ellipsoid(d_eq_um: float, aspect: float) -> tuple:
    """(r_half, thickness) [um] of the on-axis spheroid with the volume of a sphere of diameter d_eq and
    aspect = thickness / (2 r_half).  V = 4/3 pi r_half^2 (t/2) = 4/3 pi a r_half^3 = pi d^3 / 6."""
    r_half = d_eq_um / (2.0 * aspect ** (1.0 / 3.0))
    return r_half, 2.0 * aspect * r_half


def volume_um3(r_half_um: float, t_um: float) -> float:
    return 4.0 / 3.0 * math.pi * r_half_um ** 2 * (0.5 * t_um)


def make_case(cid: str, series: str, label: str, d_eq_um: float, z_c_um: float, aspect: float, shape: str) -> dict:
    r_half, t = ellipsoid(d_eq_um, aspect)
    note = ""
    if r_half > R_HALF_MAX / UM:                      # does not fit inside the rod: same volume, widest disc that fits
        r_half = R_HALF_MAX / UM
        t = volume_um3(d_eq_um / 2.0, d_eq_um) / (4.0 / 3.0 * math.pi * r_half ** 2) * 2.0
        aspect = t / (2.0 * r_half)
        note = f"r_half 가 R_cu 를 넘어 r_half = {r_half:.0f} µm 로 제한, 같은 부피 → 실제 축비 1:{1 / aspect:.1f}"
    z_min, z_max = 0.5 * t + DZ / UM, L / UM - 0.5 * t - DZ / UM
    z_used = min(max(z_c_um, z_min), z_max)
    if z_used != z_c_um:
        note = (note + "; " if note else "") + f"중심 깊이를 {z_c_um:g} → {z_used:.1f} µm 로 옮김 (면에 닿지 않게)"
    return dict(id=cid, series=series, label=label, d_eq_um=d_eq_um, z_c_um=z_used, z_c_requested_um=z_c_um, aspect=aspect,
                shape=shape, r_half_um=r_half, t_um=t, volume_um3=volume_um3(r_half, t), note=note)


def void_spec(c: dict) -> VoidSpec:
    return VoidSpec(enabled=True, depth=(c["z_c_um"] - 0.5 * c["t_um"]) * UM, thickness=c["t_um"] * UM,
                    r_half=c["r_half_um"] * UM, k=K_VOID, shape="ellipse")


# ----------------------------------------------------------------------------- one case
def _lockin_amp_mK(ds) -> dict:
    return {f: 2.0 / math.pi * ds.P_abs * np.abs(getattr(ds, f"H_{f}")) * 1e3 for f in ("pump", "far")}


def _phase_deg(ds) -> dict:
    return {f: np.degrees(np.unwrap(np.angle(getattr(ds, f"H_{f}")))) for f in ("pump", "far")}


def time_signal(t, y_void, y_base, t_star) -> dict:
    """Void signal y_void - y_base: largest absolute change [mK] (and when), largest relative change [%] where the
    reference signal is above the ΔT noise (and when), both at t*."""
    d = y_void - y_base
    i_abs = int(np.argmax(np.abs(d)))
    mask = np.abs(y_base) >= SIGMA_T_MK * 1e-3
    rel = np.where(mask, d / np.where(mask, y_base, 1.0) * 100.0, 0.0)
    i_rel = int(np.argmax(np.abs(rel)))
    return dict(max_abs_mK=float(d[i_abs] * 1e3), t_max_abs_ms=float(t[i_abs] * 1e3),
                max_rel_pct=float(rel[i_rel]), t_max_rel_ms=float(t[i_rel] * 1e3),
                rel_at_tstar_pct=float(np.interp(t_star, t, d) / np.interp(t_star, t, y_base) * 100.0),
                abs_at_tstar_mK=float(np.interp(t_star, t, d) * 1e3),
                base_at_tstar_K=float(np.interp(t_star, t, y_base)))


def run_pair(cfg):
    """Void run and the same-grid void-free run: (resp, ds, fr) for both.  2-D: mixed lock-in path (direct below
    1/window, step FFT above).  3-D (not used in stage 1): step FFT only for both."""
    out = []
    for c in (cfg, baseline_config(cfg)):
        resp = run_step_response(c, store_fields=False)
        fr = None if needs_3d(c) else frequency_response(c, FREQS)
        out.append((resp, lockin_dataset(resp, fr, FREQS), fr))
    return out


def run_case(c: dict) -> dict:
    jpath = CASES_DIR / f"{c['id']}.json"
    if jpath.exists():
        return json.loads(jpath.read_text(encoding="utf-8"))
    cfg = cw_config(void=void_spec(c), f_min=F_MIN, t_end=T_END, dz=DZ)
    t0 = time.perf_counter()
    (resp, ds, fr), (bresp, bds, bfr) = run_pair(cfg)
    wall = time.perf_counter() - t0
    t = resp.t
    assert np.array_equal(t, bresp.t), "void and reference runs must share the time grid"
    res = dict(c)
    res.update(solver=("3-D" if needs_3d(cfg) else "2-D"), n_cells=int(resp.result.n_cells), n_steps=int(resp.result.n_steps),
               void_cells=int(resp.result.diagnostics.get("void_cells", 0)), wall_s=wall, P_abs_mW=resp.P_abs * 1e3,
               energy_error=float(resp.energy_error), energy_error_base=float(bresp.energy_error),
               R_out_um=float(resp.result.grid.r_faces[-1] / UM), lockin_source=list(map(str, ds.source)),
               warnings="; ".join(resp.warnings))
    # 1. k_eff, mixed path (the deliverable) and the two single paths (cross-check)
    kk = {}
    for name, Hv, Hb in (("mixed", ds.H_far, bds.H_far),
                         ("step_fft", lockin_dataset(resp, None, FREQS).H_far, lockin_dataset(bresp, None, FREQS).H_far),
                         ("direct", fr.H_far if fr is not None else None, bfr.H_far if bfr is not None else None)):
        if Hv is None:
            continue
        kv = an.invert_tlm(FREQS, Hv, COPPER.rho_cp, L, A, shell=SHELL)["k_eff"]
        kb = an.invert_tlm(FREQS, Hb, COPPER.rho_cp, L, A, shell=SHELL)["k_eff"]
        kk[name] = dict(k_void=kv, k_base=kb, dk_pct=(kv / kb - 1.0) * 100.0)
    res["k_eff"] = kk
    res["dk_pct"] = kk["mixed"]["dk_pct"]
    # 2. time-domain void signals
    res["time"] = {f: time_signal(t, getattr(resp, f"dT_{f}"), getattr(bresp, f"dT_{f}"), T_END) for f in FACES}
    ratio_v = resp.dT_far[-1] / resp.dT_pump[-1]
    ratio_b = bresp.dT_far[-1] / bresp.dT_pump[-1]
    res["ratio_far_pump_tstar"] = dict(base=float(ratio_b), void=float(ratio_v), rel_pct=float((ratio_v / ratio_b - 1.0) * 100.0))
    # 3. lock-in void signals
    amp_v, amp_b = _lockin_amp_mK(ds), _lockin_amp_mK(bds)
    ph_v, ph_b = _phase_deg(ds), _phase_deg(bds)
    lk = {}
    for f in ("pump", "far"):
        dphi = ph_v[f] - ph_b[f]
        damp = (amp_v[f] / amp_b[f] - 1.0) * 100.0
        ip, ia = int(np.argmax(np.abs(dphi))), int(np.argmax(np.abs(damp)))
        lk[f] = dict(dphi_deg=dphi.tolist(), damp_pct=damp.tolist(), amp_base_mK=amp_b[f].tolist(), phase_base_deg=ph_b[f].tolist(),
                     max_dphi_deg=float(dphi[ip]), f_max_dphi=float(FREQS[ip]), amp_at_f_dphi_mK=float(amp_b[f][ip]),
                     max_damp_pct=float(damp[ia]), f_max_damp=float(FREQS[ia]), amp_at_f_damp_mK=float(amp_b[f][ia]))
    res["lockin"] = lk
    # 4. void indicators
    av = an.analyze(t, resp.dT_pump, resp.dT_far, resp.P_abs, L, A, COPPER.rho_cp, spot_radius=cfg.laser.w,
                    freqs=FREQS, H_far=ds.H_far, shell=SHELL)
    ab = an.analyze(t, bresp.dT_pump, bresp.dT_far, bresp.P_abs, L, A, COPPER.rho_cp, spot_radius=cfg.laser.w,
                    freqs=FREQS, H_far=bds.H_far, shell=SHELL)
    ind = {}
    for m, (b, v, dlt, rel) in an.void_indicators(ab, av).items():
        ind[m] = dict(key=an.INDICATOR_KEYS[m], base=b, void=v, delta=dlt, rel_pct=rel * 100.0)
    ind["slope_max_K_per_s"] = float(ab["step_intercept"]["slope"])        # for the time-noise conversion of t_int / t_half
    res["indicators"] = ind
    # save
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CASES_DIR / f"{c['id']}.npz", t=t, dT_pump=resp.dT_pump, dT_far=resp.dT_far,
                        dT_pump_base=bresp.dT_pump, dT_far_base=bresp.dT_far, freqs=FREQS,
                        H_far=ds.H_far, H_pump=ds.H_pump, H_far_base=bds.H_far, H_pump_base=bds.H_pump, P_abs=resp.P_abs)
    jpath.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def reference_via() -> dict:
    """Void-free via (plan §7 last row): absolute values and σ_k of the deliverable on the same lock-in path."""
    jpath = CASES_DIR / "reference_via.json"
    if jpath.exists():
        return json.loads(jpath.read_text(encoding="utf-8"))
    cfg = cw_config(f_min=F_MIN, t_end=T_END, dz=DZ)
    t0 = time.perf_counter()
    resp = run_step_response(cfg, store_fields=False)
    fr = frequency_response(cfg, FREQS)
    ds = lockin_dataset(resp, fr, FREQS)
    wall = time.perf_counter() - t0
    paths = dict(mixed=ds.H_far, direct=fr.H_far, step_fft=lockin_dataset(resp, None, FREQS).H_far)
    k = {n: an.invert_tlm(FREQS, H, COPPER.rho_cp, L, A, shell=SHELL)["k_eff"] for n, H in paths.items()}
    sig = {}
    for n, H in paths.items():
        mc = an.phase_noise_sensitivity(FREQS, H, COPPER.rho_cp, L, A, SHELL, [SIGMA_PHI_DEG], n_trials=400, seed=1)
        sig[n] = mc["rows"][0]["std_rel"] * 100.0
    res = an.analyze(resp.t, resp.dT_pump, resp.dT_far, resp.P_abs, L, A, COPPER.rho_cp, spot_radius=cfg.laser.w,
                     freqs=FREQS, H_far=ds.H_far, shell=SHELL)
    out = dict(k_eff=k, sigma_k_pct=sig, sigma_phi_deg=SIGMA_PHI_DEG, P_abs_mW=resp.P_abs * 1e3, n_cells=int(resp.result.n_cells),
               n_steps=int(resp.result.n_steps), wall_s=wall, R_out_um=float(resp.result.grid.r_faces[-1] / UM),
               dT_pump_tstar_K=float(resp.dT_pump[-1]), dT_far_tstar_K=float(resp.dT_far[-1]),
               t_int_ms=res["step_intercept"]["t_int"] * 1e3, t_half_ms=res["parker"]["t_half"] * 1e3, R_th=res["two_face"]["R_th"],
               slope_max_K_per_s=res["step_intercept"]["slope"], energy_error=float(resp.energy_error),
               lockin_amp_far_mK=(2 / math.pi * resp.P_abs * np.abs(ds.H_far) * 1e3).tolist(),
               lockin_amp_pump_mK=(2 / math.pi * resp.P_abs * np.abs(ds.H_pump) * 1e3).tolist(),
               lockin_phase_far_deg=np.degrees(np.unwrap(np.angle(ds.H_far))).tolist(),
               lockin_source=list(map(str, ds.source)), warnings="; ".join(resp.warnings))
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CASES_DIR / "reference_via.npz", t=resp.t, dT_pump=resp.dT_pump, dT_far=resp.dT_far, freqs=FREQS,
                        H_far=ds.H_far, H_pump=ds.H_pump, P_abs=resp.P_abs)
    jpath.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- detection criteria (plan §6)
def _need_N(signal: float, noise: float) -> tuple:
    n = (N_SIGMA * noise / signal) ** 2 if signal > 0 else float("inf")
    if not math.isfinite(n):
        return n, "—"
    return n, (f"N ≥ {math.ceil(n):d}" if n < 1e4 else f"N ≥ {n:.0e}")


def best_phase(lk: dict, floor_mK: float) -> dict:
    """Largest |Δφ| among the frequencies whose reference amplitude is above the noise floor (plan §6: the phase
    is only usable where the lock-in sees a signal).  Falls back to the overall maximum when none qualifies."""
    dphi, amp = np.asarray(lk["dphi_deg"]), np.asarray(lk["amp_base_mK"])
    ok = amp >= floor_mK
    cand = np.where(ok, np.abs(dphi), -1.0)
    i = int(np.argmax(cand)) if ok.any() else int(np.argmax(np.abs(dphi)))
    return dict(dphi=float(dphi[i]), f=float(FREQS[i]), amp=float(amp[i]), floor_ok=bool(ok[i]))


def judge(r: dict, sigma_k_pct: float) -> dict:
    """Per indicator: signal, threshold, detectable?, and the σ_φ or averaging count N that detection would need."""
    out = {}
    lim_k = N_SIGMA * sigma_k_pct
    dk = abs(r["dk_pct"])
    out["k_eff"] = dict(signal=f"{r['dk_pct']:+.2f} %", ok=dk >= lim_k, need=f"σ_φ ≤ {SIGMA_PHI_DEG * dk / lim_k:.2f}°" if dk > 0 else "—",
                        thresh=f"≥ {lim_k:.2f} %")
    for f in ("pump", "far"):
        lk = r["lockin"][f]
        bp = best_phase(lk, SIGMA_T_MK)
        dphi = abs(bp["dphi"])
        out[f"phase_{f}"] = dict(signal=f"{bp['dphi']:+.2f}° @ {bp['f']:.0f} Hz ({bp['amp']:.1f} mK)",
                                 ok=(dphi >= N_SIGMA * SIGMA_PHI_DEG) and bp["floor_ok"],
                                 need=(f"σ_φ ≤ {dphi / N_SIGMA:.2f}°" + ("" if bp["floor_ok"] else "; 진폭 < σ_T")), thresh=f"≥ {N_SIGMA * SIGMA_PHI_DEG:.1f}°",
                                 dphi=bp["dphi"], f=bp["f"])
        damp = abs(lk["max_damp_pct"])
        out[f"amp_{f}"] = dict(signal=f"{lk['max_damp_pct']:+.2f} % @ {lk['f_max_damp']:.0f} Hz", ok=damp >= N_SIGMA * AMP_DEV_PCT,
                               need=f"흡수 편차 ≤ {damp / N_SIGMA:.2f} %", thresh=f"≥ {N_SIGMA * AMP_DEV_PCT:.0f} % (R 의존)")
    for f in FACES:
        ts = r["time"][f]
        s = abs(ts["max_abs_mK"])
        n, need = _need_N(s, SIGMA_T_MK)
        out[f"dT_{f}"] = dict(signal=f"{ts['max_abs_mK']:+.2f} mK @ {ts['t_max_abs_ms']:.2f} ms ({ts['max_rel_pct']:+.2f} % @ {ts['t_max_rel_ms']:.2f} ms)",
                              ok=s >= N_SIGMA * SIGMA_T_MK, need=need, N=n, thresh=f"≥ {N_SIGMA * SIGMA_T_MK:.0f} mK (R 의존)")
    # pump-face ΔT at t* and the two-face R_th (both are ΔT-type signals)
    ts = r["time"]["pump"]
    s = abs(ts["abs_at_tstar_mK"])
    n, need = _need_N(s, SIGMA_T_MK)
    out["dT_pump_tstar"] = dict(signal=f"{ts['abs_at_tstar_mK']:+.2f} mK ({ts['rel_at_tstar_pct']:+.2f} %)", ok=s >= N_SIGMA * SIGMA_T_MK,
                                need=need, N=n, thresh=f"≥ {N_SIGMA * SIGMA_T_MK:.0f} mK (R 의존)")
    ts = r["time"]["diff"]
    s = abs(ts["abs_at_tstar_mK"])
    n, need = _need_N(s, SIGMA_T_MK)
    rth = r["indicators"]["two_face"]
    out["R_th"] = dict(signal=f"{rth['rel_pct']:+.2f} % ({ts['abs_at_tstar_mK']:+.2f} mK)", ok=s >= N_SIGMA * SIGMA_T_MK,
                       need=need, N=n, thresh=f"≥ {N_SIGMA * SIGMA_T_MK:.0f} mK (R 의존)")
    # t_int, t_half: time noise = ΔT noise / maximum slope of the far-face curve
    sig_t = SIGMA_T_MK * 1e-3 / r["indicators"]["slope_max_K_per_s"]
    for m in ("step_intercept", "parker"):
        d = r["indicators"][m]
        s = abs(d["delta"])
        n, need = _need_N(s, sig_t)
        out[d["key"]] = dict(signal=f"{d['delta'] * 1e6:+.1f} µs ({d['rel_pct']:+.2f} %)", ok=s >= N_SIGMA * sig_t,
                             need=need, N=n, thresh=f"≥ {N_SIGMA * sig_t * 1e6:.1f} µs (R 의존)")
    return out


JUDGE_ORDER = [("k_eff", "k_eff (반대면)"), ("phase_far", "위상 변화, 반대면"), ("amp_far", "진폭 변화, 반대면"),
               ("t_int", "t_int (반대면)"), ("t_half", "t½ (반대면)"), ("dT_far", "ΔT 반대면 (최대)"),
               ("phase_pump", "위상 변화, 펌프면"), ("amp_pump", "진폭 변화, 펌프면"), ("dT_pump", "ΔT 펌프면 (최대)"),
               ("dT_pump_tstar", "ΔT 펌프면 (t*)"), ("R_th", "R_th(t*) (양면)"), ("dT_diff", "ΔT 양면 차 (최대)")]
CONFIG_OF = {"k_eff": "반대면", "phase_far": "반대면", "amp_far": "반대면", "t_int": "반대면", "t_half": "반대면", "dT_far": "반대면",
             "phase_pump": "펌프면", "amp_pump": "펌프면", "dT_pump": "펌프면", "dT_pump_tstar": "펌프면", "R_th": "양면", "dT_diff": "양면"}


# ----------------------------------------------------------------------------- reference-void selection
def pick_reference(rows_a: list, target_pct: float) -> dict:
    """Diameter where |Δk/k| of the sphere ladder equals `target_pct`: log-log interpolation between the two
    bracketing ladder points, rounded to 0.5 um."""
    d = np.array([r["d_eq_um"] for r in rows_a], float)
    y = np.array([abs(r["dk_pct"]) for r in rows_a], float)
    o = np.argsort(d)
    d, y = d[o], y[o]
    if y[-1] < target_pct:
        return dict(ok=False, d_um=float(d[-1]), reason=f"사다리 최대 지름 {d[-1]:g} µm 에서도 |Δk/k| = {y[-1]:.2f} % < {target_pct:.2f} %")
    j = int(np.argmax(y >= target_pct))
    if j == 0:
        return dict(ok=True, d_um=float(d[0]), reason="사다리 최소 지름에서 이미 목표 초과")
    ld = np.interp(math.log(target_pct), [math.log(y[j - 1]), math.log(y[j])], [math.log(d[j - 1]), math.log(d[j])])
    d_ref = round(2.0 * math.exp(ld)) / 2.0
    return dict(ok=True, d_um=float(d_ref), bracket=(float(d[j - 1]), float(d[j])), bracket_pct=(float(y[j - 1]), float(y[j])),
                exponent=float((math.log(y[j]) - math.log(y[j - 1])) / (math.log(d[j]) - math.log(d[j - 1]))),
                reason=f"{d[j - 1]:g} µm ({y[j - 1]:.2f} %) 과 {d[j]:g} µm ({y[j]:.2f} %) 사이 log-log 보간 → {math.exp(ld):.1f} µm, 0.5 µm 로 반올림")


# ----------------------------------------------------------------------------- main
def log(msg: str):
    print(msg, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)
    ref = reference_via()
    sigma_k = ref["sigma_k_pct"]["mixed"]
    target = 2.0 * N_SIGMA * sigma_k
    log(f"reference via: k_eff mixed={ref['k_eff']['mixed']:.2f} direct={ref['k_eff']['direct']:.2f} step_fft={ref['k_eff']['step_fft']:.2f}; "
        f"σ_k(0.5°) mixed={sigma_k:.3f} % → 3σ={N_SIGMA * sigma_k:.2f} %, target Δk/k={target:.2f} %  ({ref['wall_s']:.1f}s)")
    # series A
    rows_a = []
    ladder = list(LADDER_UM)
    ext = list(LADDER_EXT_UM)
    while True:
        for d in ladder:
            cid = f"A_sph_d{d:g}_z{Z_REF_UM:g}"
            if any(r["id"] == cid for r in rows_a):
                continue
            c = make_case(cid, "A", f"구 ⌀{d:g} µm", d, Z_REF_UM, 1.0, "sphere")
            r = run_case(c)
            rows_a.append(r)
            log(f"A d={d:5.1f} µm  Δk/k={r['dk_pct']:+.3f} %  far {r['time']['far']['max_abs_mK']:+.3f} mK  pump {r['time']['pump']['max_abs_mK']:+.3f} mK  "
                f"Δφ_far={r['lockin']['far']['max_dphi_deg']:+.3f}°  Δt_int={r['indicators']['step_intercept']['rel_pct']:+.3f} %  "
                f"cells={r['n_cells']} {r['wall_s']:.1f}s")
        sel = pick_reference(rows_a, target)
        if sel["ok"] or not ext:
            break
        ladder.append(ext.pop(0))          # extend the ladder until the target is bracketed
    log(f"reference void: {sel}")
    d_ref = sel["d_um"]
    # series B (depth) and C (shape) of the reference void
    rows_b, rows_c = [], []
    for z in DEPTHS_UM:
        cid = f"B_sph_d{d_ref:g}_z{z:g}"
        c = make_case(cid, "B", f"구 ⌀{d_ref:g} µm, 중심 깊이 {z:g} µm", d_ref, z, 1.0, "sphere")
        r = run_case(c)
        rows_b.append(r)
        log(f"B z={r['z_c_um']:5.1f} µm  Δk/k={r['dk_pct']:+.3f} %  far {r['time']['far']['max_abs_mK']:+.3f} mK  pump {r['time']['pump']['max_abs_mK']:+.3f} mK  "
            f"Δt_int={r['indicators']['step_intercept']['rel_pct']:+.3f} %  ΔR_th={r['indicators']['two_face']['rel_pct']:+.3f} %  {r['wall_s']:.1f}s")
    for shape, aspect in SHAPES.items():
        if shape == "sphere":
            rows_c.append(next(r for r in rows_b if r["z_c_requested_um"] == Z_REF_UM))
            continue
        cid = f"C_{shape}_d{d_ref:g}_z{Z_REF_UM:g}"
        c = make_case(cid, "C", SHAPE_LABEL[shape], d_ref, Z_REF_UM, aspect, shape)
        r = run_case(c)
        rows_c.append(r)
        log(f"C {shape:10s} r_half={r['r_half_um']:.1f} t={r['t_um']:.1f} µm  Δk/k={r['dk_pct']:+.3f} %  far {r['time']['far']['max_abs_mK']:+.3f} mK  "
            f"pump {r['time']['pump']['max_abs_mK']:+.3f} mK  Δt_int={r['indicators']['step_intercept']['rel_pct']:+.3f} %  {r['wall_s']:.1f}s  {r['note']}")
        if shape == "disc_1_10" and r["note"]:
            # the equal-volume 1:10 disc did not fit: add a true 1:10 disc at the widest radius (smaller volume), flagged
            r_half = R_HALF_MAX / UM
            t = 2.0 * 0.10 * r_half
            c = dict(id=f"C_disc_1_10_r38_z{Z_REF_UM:g}", series="C", label=SHAPE_LABEL["disc_1_10_r38"], d_eq_um=2.0 * (volume_um3(r_half, t) * 3 / (4 * math.pi)) ** (1 / 3),
                     z_c_um=Z_REF_UM, z_c_requested_um=Z_REF_UM, aspect=0.10, shape="disc_1_10_r38", r_half_um=r_half, t_um=t,
                     volume_um3=volume_um3(r_half, t), note=f"부피 {volume_um3(r_half, t) / volume_um3(d_ref / 2, d_ref) * 100:.0f} % (기준 부피의)")
            r = run_case(c)
            rows_c.append(r)
            log(f"C {c['shape']:10s} r_half={r['r_half_um']:.1f} t={r['t_um']:.1f} µm  Δk/k={r['dk_pct']:+.3f} %  far {r['time']['far']['max_abs_mK']:+.3f} mK  "
                f"pump {r['time']['pump']['max_abs_mK']:+.3f} mK  {r['wall_s']:.1f}s  {r['note']}")
    diag = capacity_diagnostic(d_ref)
    rows = rows_a + rows_b + [r for r in rows_c if r["series"] == "C"]
    write_csv(rows, sigma_k)
    write_interim(ref, sel, rows_a, rows_b, rows_c, diag)
    make_figures(ref, rows_a, rows_b, rows_c)
    log(f"done -> {OUT}")


def capacity_diagnostic(d_ref: float) -> list:
    """Why does Δk/k change sign near the faces?  The reference sphere at centre depth 25 and 250 um with (a) only
    the conductivity of air (heat capacity of copper kept) and (b) only the heat capacity of air (conductivity of
    copper kept), so the series-resistance and the missing-heat-capacity contributions can be read separately."""
    jpath = CASES_DIR / "diag_capacity.json"
    if jpath.exists():
        return json.loads(jpath.read_text(encoding="utf-8"))
    from ttr_sim.materials import AIR
    out = []
    for z in (25.0, 250.0):
        for variant, kv, rc in (("k 만 (ρc = Cu)", K_VOID, COPPER.rho_cp), ("ρc 만 (k = Cu)", COPPER.k, AIR.rho_cp), ("둘 다 (공기)", K_VOID, AIR.rho_cp)):
            c = make_case("diag", "D", variant, d_ref, z, 1.0, "sphere")
            v = VoidSpec(enabled=True, depth=(c["z_c_um"] - 0.5 * c["t_um"]) * UM, thickness=c["t_um"] * UM, r_half=c["r_half_um"] * UM,
                         k=kv, rho_cp=rc, shape="ellipse")
            cfg = cw_config(void=v, f_min=F_MIN, t_end=T_END, dz=DZ)
            (resp, ds, _), (bresp, bds, _) = run_pair(cfg)
            kk = an.invert_tlm(FREQS, ds.H_far, COPPER.rho_cp, L, A, shell=SHELL)["k_eff"]
            kb = an.invert_tlm(FREQS, bds.H_far, COPPER.rho_cp, L, A, shell=SHELL)["k_eff"]
            ti = an.step_intercept(resp.t, resp.dT_far, L)["t_int"]
            tb = an.step_intercept(bresp.t, bresp.dT_far, L)["t_int"]
            out.append(dict(z_c_um=c["z_c_um"], variant=variant, dk_pct=(kk / kb - 1) * 100, dt_int_pct=(ti / tb - 1) * 100,
                            far_max_mK=float((resp.dT_far - bresp.dT_far)[np.argmax(np.abs(resp.dT_far - bresp.dT_far))] * 1e3),
                            pump_tstar_mK=float((resp.dT_pump[-1] - bresp.dT_pump[-1]) * 1e3)))
            log(f"diag z={z:g} {variant}: Δk/k={out[-1]['dk_pct']:+.3f} %  Δt_int={out[-1]['dt_int_pct']:+.3f} %")
    jpath.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- outputs
def flat_row(r: dict, sigma_k: float) -> dict:
    row = dict(id=r["id"], series=r["series"], label=r["label"], shape=r["shape"], d_eq_um=r["d_eq_um"], z_c_um=r["z_c_um"],
               r_half_um=r["r_half_um"], t_um=r["t_um"], aspect=r["aspect"], volume_um3=r["volume_um3"], note=r["note"],
               solver=r["solver"], n_cells=r["n_cells"], void_cells=r["void_cells"], n_steps=r["n_steps"], wall_s=r["wall_s"],
               energy_error=r["energy_error"], R_out_um=r["R_out_um"],
               k_void=r["k_eff"]["mixed"]["k_void"], k_base=r["k_eff"]["mixed"]["k_base"], dk_pct=r["dk_pct"],
               dk_pct_direct=r["k_eff"].get("direct", {}).get("dk_pct", ""), dk_pct_step_fft=r["k_eff"]["step_fft"]["dk_pct"])
    for f in FACES:
        for k, v in r["time"][f].items():
            row[f"{f}_{k}"] = v
    row["ratio_far_pump_rel_pct"] = r["ratio_far_pump_tstar"]["rel_pct"]
    for f in ("pump", "far"):
        lk = r["lockin"][f]
        for k in ("max_dphi_deg", "f_max_dphi", "amp_at_f_dphi_mK", "max_damp_pct", "f_max_damp"):
            row[f"{f}_{k}"] = lk[k]
        for fq, dp, da in zip(FREQS, lk["dphi_deg"], lk["damp_pct"]):
            row[f"{f}_dphi_deg_{fq:.0f}Hz"] = dp
            row[f"{f}_damp_pct_{fq:.0f}Hz"] = da
    for m in ("step_intercept", "parker", "two_face", "tlm_shell"):
        d = r["indicators"][m]
        row[f"{d['key']}_base"], row[f"{d['key']}_void"], row[f"{d['key']}_rel_pct"] = d["base"], d["void"], d["rel_pct"]
    jd = judge(r, sigma_k)
    for key, _ in JUDGE_ORDER:
        row[f"detect_{key}"] = "Y" if jd[key]["ok"] else "N"
        row[f"need_{key}"] = jd[key]["need"]
    return row


def write_csv(rows, sigma_k):
    flat = [flat_row(r, sigma_k) for r in rows]
    keys = list(flat[0])
    with open(OUT / "cases.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in flat:
            w.writerow({k: r.get(k, "") for k in keys})


def _yn(ok: bool) -> str:
    return "가능" if ok else "불가"


def series_table(rows, sigma_k, cond_label, cond_fn) -> list:
    """Two tables: (1) signal values, (2) judgement with the σ_φ / N each indicator needs."""
    keys = JUDGE_ORDER
    lines = ["| " + cond_label + " | " + " | ".join(lab for _, lab in keys) + " |", "|---|" + "---|" * len(keys)]
    for r in rows:
        jd = judge(r, sigma_k)
        lines.append(f"| {cond_fn(r)} | " + " | ".join(jd[k]["signal"] for k, _ in keys) + " |")
    lines += ["", "판정 (가능/불가) 과 검출에 필요한 조건 (k_eff·위상: 필요한 σ_φ; 진폭: 허용 흡수 편차; ΔT 류: 필요한 평균 횟수 N):", "",
              "| " + cond_label + " | " + " | ".join(lab for _, lab in keys) + " |", "|---|" + "---|" * len(keys)]
    for r in rows:
        jd = judge(r, sigma_k)
        lines.append(f"| {cond_fn(r)} | " + " | ".join(f"{_yn(jd[k]['ok'])} ({jd[k]['need']})" for k, _ in keys) + " |")
    return lines


def _ratio(a, b) -> str:
    return f"{a / b:.2f}" if b != 0 else "—"


def write_interim(ref, sel, rows_a, rows_b, rows_c, diag):
    sigma_k = ref["sigma_k_pct"]["mixed"]
    lim = N_SIGMA * sigma_k
    d_ref = sel["d_um"]
    r_ref = next(r for r in rows_b if r["z_c_requested_um"] == Z_REF_UM)
    all_rows = rows_a + rows_b + [r for r in rows_c if r["series"] == "C"]
    extra = [r["d_eq_um"] for r in rows_a if r["d_eq_um"] > max(LADDER_UM)]
    lines = ["# CW 모드 void 검출 가능성 — 1단계 중간 보고 (2-D, 축 위 단일 void)", "",
             f"작성 {time.strftime('%Y-%m-%d')}. 스크립트 `studies/cw_void_detect.py`, 수치 `results/cw_void_detect/cases.csv`, "
             "케이스별 `cases/<id>.json|.npz`. 계획: `docs/cw-void-detection-study-plan.md`.", "",
             "## 0. 고정 조건", "",
             f"입사 10 mW, R = 0.6 (흡수 {ref['P_abs_mW']:.1f} mW), 플랫 스팟 등가 반경 5.48 µm, 프로브 지름 8 µm 양면, L = 500 µm, R_cu = 40 µm, "
             f"외곽 far (자동 반경 {ref['R_out_um']:.0f} µm, f_min 50 Hz), 관측창 20 ms, Δz 2.5 µm, Δt 성장률 1.05, "
             "Lock-in 50 Hz–2 kHz 로그 9점, t* = 20 ms, 스텝 절편 = 최대 기울기 접선, void 충전 k = 0.026 (공기). "
             "Lock-in 데이터 경로: 1/창(50 Hz) 미만은 주파수 영역 직접 풀이, 이상은 스텝 응답 FFT (9점 중 50 Hz 한 점만 직접). "
             "모든 Δ 는 **같은 격자의 void 없는 via** 대비 (케이스마다 기준 via 를 같은 격자로 다시 계산).", "",
             "## 1. 판정 기준 (가정은 가정)", "",
             "| 지표 | 기준 | 가정 |", "|---|---|---|",
             f"| k_eff | \\|Δk/k\\| ≥ 3 σ_k = {lim:.2f} % (σ_k = {sigma_k:.3f} %: 기준 via 의 같은 경로 lock-in 위상에 σ_φ = {SIGMA_PHI_DEG}° 가우시안 잡음, 400 회 역산) | σ_φ = 0.5° |",
             f"| Lock-in 위상 변화 | 기준 진폭이 잡음 바닥 σ_T = {SIGMA_T_MK} mK 이상인 주파수 중 \\|Δφ\\| 가 최대인 주파수에서 \\|Δφ\\| ≥ 3 σ_φ = {N_SIGMA * SIGMA_PHI_DEG:.1f}° | σ_φ = 0.5°; 진폭 바닥에 ΔT 잡음 준용 (반대면 2 kHz 는 3.4 mK 로 제외됨) |",
             f"| Lock-in 진폭 변화 | ≥ 3 × 흡수 편차 = {N_SIGMA * AMP_DEV_PCT:.0f} % | **흡수 편차 1 % 는 근거 없는 가정** (R 의존) |",
             f"| ΔT 변화 (펌프면·반대면·양면 차·R_th) | 신호 ≥ 3 σ_T = {N_SIGMA * SIGMA_T_MK:.0f} mK (1회); N 회 평균 시 N = (3·{SIGMA_T_MK} mK / 신호)² | σ_T = 6.7 mK (ΔR/R 1e-6, c_tr 1.5e-4), R 의존 |",
             f"| t_int, t½ | \\|Δt\\| ≥ 3 σ_t, σ_t = σ_T / (반대면 곡선 최대 기울기) = {SIGMA_T_MK * 1e-3 / ref['slope_max_K_per_s'] * 1e6:.1f} µs | 동일; t½ 는 미분 곡선이므로 실제 시각 잡음은 이보다 크다 (하한) |",
             "", "ΔT 류 신호의 '최대' 는 관측창 안에서 \\|void − 기준\\| 이 가장 큰 시각의 값, 괄호의 상대 변화는 기준 신호가 σ_T 이상인 구간에서의 최대값.", "",
             "## 2. 기준 via (void 없음)", "",
             "| 경로 | k_eff [W/mK] | σ_k(σ_φ = 0.5°) [%] |", "|---|---|---|"]
    for n, lab in (("mixed", "혼합 (1/창 미만 직접, 이상 스텝 FFT) — 사용"), ("direct", "주파수 영역 직접 풀이"), ("step_fft", "스텝 FFT 전체")):
        lines.append(f"| {lab} | {ref['k_eff'][n]:.1f} ({(ref['k_eff'][n] / COPPER.k - 1) * 100:+.2f} %) | {ref['sigma_k_pct'][n]:.3f} |")
    lines += ["", f"ΔT_pump(t*) = {ref['dT_pump_tstar_K']:.3f} K, ΔT_far(t*) = {ref['dT_far_tstar_K']:.3f} K, t_int = {ref['t_int_ms']:.3f} ms, "
              f"t½ = {ref['t_half_ms']:.3f} ms, R_th(t*) = {ref['R_th']:.0f} K/W, 반대면 최대 기울기 {ref['slope_max_K_per_s']:.0f} K/s. "
              f"반대면 lock-in 진폭 {ref['lockin_amp_far_mK'][0]:.0f} mK @ 50 Hz → {ref['lockin_amp_far_mK'][-1]:.1f} mK @ 2 kHz; "
              f"펌프면 {ref['lockin_amp_pump_mK'][0]:.0f} → {ref['lockin_amp_pump_mK'][-1]:.0f} mK. 참값 k = {COPPER.k:.0f} W/mK (유리 껍질 TLM 모델 편향 +1 %).", "",
              "## 3. 계열 A — 크기 사다리 (구, 축 위, 중심 깊이 250 µm)", ""]
    if extra:
        lines += [f"계획의 사다리(5–40 µm)에서는 목표 \\|Δk/k\\| = {2 * lim:.2f} % 에 못 미쳐 {', '.join(f'{d:g}' for d in extra)} µm 를 추가했다.", ""]
    lines += series_table(sorted(rows_a, key=lambda r: r["d_eq_um"]), sigma_k, "지름 [µm]", lambda r: f"{r['d_eq_um']:g}")
    lines += ["", "![](figs/A_size_ladder.png)", "", "![](figs/A_far_signal.png)", "",
              "## 4. 기준 void 선정", "",
              f"기준: Δk/k ≈ 2 × 3σ_k = {2 * lim:.2f} %. {sel['reason']}. → **기준 void = 구 ⌀{d_ref:g} µm** "
              f"(부피 {r_ref['volume_um3']:.0f} µm³, 단면 점유 {(r_ref['r_half_um'] / 40) ** 2 * 100:.0f} %, 실제 Δk/k = {r_ref['dk_pct']:+.2f} %).",
              (f"사다리의 \\|Δk/k\\| 는 지름의 약 {sel['exponent']:.1f} 제곱에 비례한다 (보간 구간)." if "exponent" in sel else ""), "",
              "## 5. 계열 B — 깊이 (기준 구)", ""]
    lines += series_table(rows_b, sigma_k, "중심 깊이 [µm]", lambda r: f"{r['z_c_um']:g}" + (f" ({r['note']})" if r["note"] else ""))
    lines += ["", "![](figs/B_depth.png)", "", "감도의 깊이 의존 (중심 깊이 250 µm 값으로 정규화; t_int 의 1-D 근사 예측은 ∝ x(1−x), x = z/L):", "",
              "| 중심 깊이 [µm] | x(1−x)/0.25 (예측) | Δt_int 비 | Δt½ 비 | ΔR_th 비 | Δk/k 비 | 반대면 Δφ 최대 비 | 펌프면 ΔT(t*) 비 | 반대면 ΔT 최대 비 |",
              "|---|---|---|---|---|---|---|---|---|"]
    rm = r_ref
    for r in rows_b:
        x = r["z_c_um"] / (L / UM)
        lines.append(f"| {r['z_c_um']:g} | {x * (1 - x) / 0.25:.2f} | {_ratio(r['indicators']['step_intercept']['delta'], rm['indicators']['step_intercept']['delta'])} | "
                     f"{_ratio(r['indicators']['parker']['delta'], rm['indicators']['parker']['delta'])} | {_ratio(r['indicators']['two_face']['delta'], rm['indicators']['two_face']['delta'])} | "
                     f"{_ratio(r['dk_pct'], rm['dk_pct'])} | {_ratio(r['lockin']['far']['max_dphi_deg'], rm['lockin']['far']['max_dphi_deg'])} | "
                     f"{_ratio(r['time']['pump']['abs_at_tstar_mK'], rm['time']['pump']['abs_at_tstar_mK'])} | "
                     f"{_ratio(r['time']['far']['max_abs_mK'], rm['time']['far']['max_abs_mK'])} |")
    lines += ["", "## 6. 계열 C — 형태 (기준 부피 동일, 중심 깊이 250 µm)", "",
              "| 형태 | r_half [µm] | 두께 [µm] | 단면 점유 [%] | 비고 |", "|---|---|---|---|---|"]
    for r in rows_c:
        lines.append(f"| {SHAPE_LABEL[r['shape']]} | {r['r_half_um']:.1f} | {r['t_um']:.1f} | {(r['r_half_um'] / 40) ** 2 * 100:.0f} | {r['note'] or '—'} |")
    lines += [""] + series_table(rows_c, sigma_k, "형태", lambda r: SHAPE_LABEL[r["shape"]])
    lines += ["", "![](figs/C_shape.png)", "", "## 7. 요약 — 프로브 구성 × 지표별 검출 한계 (축 위 단일 void, 이 단계의 범위)", ""]
    lines += summary_table(rows_a, rows_b, rows_c, sigma_k)
    ra = sorted(rows_a, key=lambda r: r["d_eq_um"])
    r50 = next((r for r in ra if r["d_eq_um"] == 50), ra[-1])
    r25 = rows_b[0]
    lines += ["", "핵심:", "",
              f"* **가장 유리한 구성·지표는 반대면 프로브의 k_eff (위상 TLM)** 다: 축 위 구는 지름 40 µm 부터, 기준 구(⌀{d_ref:g} µm)는 모든 깊이에서 3σ_k 를 넘는다. "
              "같은 반대면 데이터의 개별 위상 변화(최대 1–2°)는 3σ_φ = 1.5° 경계에 걸리고, 진폭 변화는 흡수 편차 가정에 좌우된다.",
              f"* 구형 void 의 k_eff 신호는 작다: \\|Δk/k\\| ∝ d^{sel.get('exponent', 3):.1f}, ⌀25 µm 에서 0.3 %, ⌀{r50['d_eq_um']:g} µm 에서 {abs(r50['dk_pct']):.1f} %. "
              "이유는 부록 8.3: void 는 열이 채우지 못하는 부피이므로 열용량을 빼고(겉보기 α 상승, 로드 중앙 +1.8 %), 직렬 저항은 이를 넘어서는 만큼만(−3.5 %) 보인다. "
              f"면 근처(25·475 µm)에서는 열류가 어차피 스팟에서 퍼지는 중이라 축 위 장애물의 저항이 작고 열용량 효과는 더 커서(+3.3 %) Δk/k 부호가 뒤집힌다 ({r25['dk_pct']:+.2f} %).",
              "* 반대면 지표(k_eff, t_int, t½, 반대면 ΔT·위상)는 z 와 L−z 에 대해 정확히 대칭이다 (100 ↔ 400, 25 ↔ 475): 반대면 프로브는 void 깊이를 구분하지 못한다. "
              "t_int 의 깊이 의존은 100–400 µm 구간에서 1-D 예측 x(1−x) 와 맞고(0.67 vs 0.64), 양 끝에서는 열용량 효과로 부호가 바뀐다.",
              "* 펌프면·양면 지표는 펌프면 쪽 void 에만 강하다: 중심 깊이 25 µm 의 기준 구는 펌프면 ΔT +0.46 K (1회 측정으로 충분), 100 µm 에서 +25 mK (1회), 250 µm 부터는 평균이 필요하다. "
              "R_th 의 깊이 의존은 flux ∝ 1 − z/L 예측대로 선형에 가깝다 (100: 1.7, 400: 0.38).",
              "* 형태: 같은 부피면 **넓은 원판이 구보다 5–8배** 큰 신호를 낸다 (1:4 원판 Δk/k −9 %, 반대면 ΔT −24 mK 로 1회 검출). 막대(4:1)는 구의 1/3 로 k_eff 로도 못 본다. "
              "등가 1:10 원판은 R_cu 안에 들어가지 않으므로(r_half 48 µm) r_half 38 µm 로 제한한 두 변형(같은 부피 → 1:5 / 진짜 1:10 → 부피 절반)을 돌렸다: "
              "단면 점유(90 %)가 같으면 두께가 절반이어도 Δk/k 는 −12 % 대 −14 % 로 비슷하다 — 신호는 부피보다 **막힌 단면적**이 정한다.",
              "* 시간 영역 반대면 신호(void − 기준)는 20 ms 끝에서 최대이고 포화하지 않는다: 창을 늘리면 더 커진다 (DC 극한에서 void 의 직렬 저항 × 통과 열류). "
              "펌프면 신호는 5–6 ms 에서 최대 후 줄어든다."]
    lines += ["", "## 8. 부록", "",
              "### 8.1 계산 경로 교차검증 (Δk/k [%]: 혼합 경로 vs 직접 풀이 vs 스텝 FFT)", "",
              "| 케이스 | 혼합 | 직접 | 스텝 FFT | 기준 k (혼합/직접/FFT) |", "|---|---|---|---|---|"]
    for r in all_rows:
        k = r["k_eff"]
        lines.append(f"| {r['id']} | {k['mixed']['dk_pct']:+.3f} | {k['direct']['dk_pct']:+.3f} | {k['step_fft']['dk_pct']:+.3f} | "
                     f"{k['mixed']['k_base']:.1f} / {k['direct']['k_base']:.1f} / {k['step_fft']['k_base']:.1f} |")
    lines += ["", "### 8.2 계산 시간·격자", "",
              "| 케이스 | 솔버 | 셀 | void 셀 | 스텝 | 시간 [s] (void+기준) | 에너지 오차 |", "|---|---|---|---|---|---|---|"]
    for r in all_rows:
        lines.append(f"| {r['id']} | {r['solver']} | {r['n_cells']} | {r['void_cells']} | {r['n_steps']} | {r['wall_s']:.1f} | {r['energy_error']:.1e} |")
    lines += ["", f"### 8.3 진단 — 기준 구 ⌀{d_ref:g} µm 의 열용량 효과와 저항 효과 분리", "",
              "void 의 k 만 공기로(열용량은 Cu 유지), 열용량만 공기로(k 는 Cu 유지), 둘 다 공기로 바꾼 세 경우:", "",
              "| 중심 깊이 [µm] | void 물성 | Δk/k [%] | Δt_int [%] | 반대면 ΔT 최대 [mK] | 펌프면 ΔT(t*) [mK] |", "|---|---|---|---|---|---|"]
    for d in diag:
        lines.append(f"| {d['z_c_um']:g} | {d['variant']} | {d['dk_pct']:+.3f} | {d['dt_int_pct']:+.3f} | {d['far_max_mK']:+.2f} | {d['pump_tstar_mK']:+.1f} |")
    lines += ["", "* 'k 만' 과 '둘 다' 가 같다: k ≈ 0 인 영역은 어차피 충전되지 않으므로 void 충전물의 열용량은 결과에 들어오지 않는다. 즉 공기 void 는 항상 '열용량이 빠진 부피 + 직렬 저항' 이다.",
              "* 'ρc 만' 이 열용량 결손만의 효과다 (전도는 Cu 그대로): 로드 중앙 +1.8 % (부피 분율 1.8 % 와 같음), 면 근처 +3.3 % (스팟 바로 아래의 열용량이 위상에 더 크게 작용).",
              "* 따라서 직렬 저항 몫은 나머지다: 중앙 −3.5 %, 면 근처 −2.4 % (엄밀히 가법적이지는 않음). 구의 k_eff 신호가 작은 것은 두 효과의 상쇄 때문이고, 원판처럼 단면을 넓게 막는 형태에서는 저항이 압도한다."]
    lines += ["", "### 8.4 알아둘 것", "",
              "* 기준 via 의 k_eff 가 격자마다 조금 다르다 (8.1 의 기준 k 열): void 주변의 격자 세분화 때문이며, 그래서 Δ 는 항상 같은 격자의 기준 대비다.",
              "* 혼합 경로와 스텝 FFT 경로의 Δk/k 는 거의 같다 (9점 중 50 Hz 한 점만 직접 풀이). 직접 풀이 경로와의 차이가 경로 오차의 크기다.",
              "* 2단계(3-D, 축 밖)에서는 lock-in 데이터가 스텝 FFT 경로만 있으므로 σ_k 를 §2 표의 `스텝 FFT 전체` 값으로 다시 쓴다."]
    (OUT / "interim.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_table(rows_a, rows_b, rows_c, sigma_k) -> list:
    """Detection limit per indicator: smallest ladder diameter detected (series A), depth range detected (B), shapes detected (C)."""
    lines = ["| 프로브 구성 | 지표 | 축 위 구: 검출 최소 지름 [µm] (사다리) | 기준 구의 검출 깊이 [µm] | 기준 부피에서 검출되는 형태 |", "|---|---|---|---|---|"]
    ja = {r["d_eq_um"]: judge(r, sigma_k) for r in rows_a}
    jb = {r["z_c_um"]: judge(r, sigma_k) for r in rows_b}
    jc = {r["shape"]: judge(r, sigma_k) for r in rows_c}
    for key, lab in JUDGE_ORDER:
        ds = sorted(d for d, j in ja.items() if j[key]["ok"])
        zs = sorted(z for z, j in jb.items() if j[key]["ok"])
        ss = [SHAPE_LABEL[s] for s, j in jc.items() if j[key]["ok"]]
        d_txt = (f"≥ {ds[0]:g}" + ("" if all(ja[d][key]["ok"] for d in ja if d >= ds[0]) else " (비단조)")) if ds else "불가 (사다리 내)"
        z_txt = ", ".join(f"{z:g}" for z in zs) if zs else "불가"
        lines.append(f"| {CONFIG_OF[key]} | {lab} | {d_txt} | {z_txt} | {', '.join(ss) if ss else '없음'} |")
    return lines


def _load_curves(cid: str):
    z = np.load(CASES_DIR / f"{cid}.npz")
    return {k: z[k] for k in z.files}


def make_figures(ref, rows_a, rows_b, rows_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    plt.rcParams["font.family"] = ["DejaVu Sans", "Malgun Gothic"]   # Latin / minus from DejaVu, Korean glyphs by fallback
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    sigma_k = ref["sigma_k_pct"]["mixed"]
    rows_a = sorted(rows_a, key=lambda r: r["d_eq_um"])
    d = np.array([r["d_eq_um"] for r in rows_a])

    def plain_ticks(ax, values):
        ax.set_xticks(values); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(NullFormatter())
    # A: indicators vs diameter (log-log, |Δ|) + thresholds
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.loglog(d, [abs(r["dk_pct"]) for r in rows_a], "o-", label="|Δk/k| [%] (반대면 k_eff)")
    ax.loglog(d, [abs(r["indicators"]["step_intercept"]["rel_pct"]) for r in rows_a], "s--", label="|Δt_int/t_int| [%]")
    ax.loglog(d, [abs(r["indicators"]["parker"]["rel_pct"]) for r in rows_a], "d--", label="|Δt½/t½| [%]")
    ax.loglog(d, [abs(r["indicators"]["two_face"]["rel_pct"]) for r in rows_a], "^--", label="|ΔR_th/R_th| [%] (양면)")
    ax.loglog(d, [abs(r["lockin"]["far"]["max_dphi_deg"]) for r in rows_a], "v-", label="|Δφ| 최대, 반대면 [°]")
    ax.loglog(d, [abs(r["lockin"]["pump"]["max_dphi_deg"]) for r in rows_a], "v:", label="|Δφ| 최대, 펌프면 [°]")
    ax.axhline(N_SIGMA * sigma_k, color="C0", lw=0.8, ls=":"); ax.text(d[0], N_SIGMA * sigma_k * 1.1, f"3σ_k = {N_SIGMA * sigma_k:.2f} %", fontsize=7, color="C0")
    ax.axhline(N_SIGMA * SIGMA_PHI_DEG, color="C4", lw=0.8, ls=":"); ax.text(d[0], N_SIGMA * SIGMA_PHI_DEG * 1.1, f"3σ_φ = {N_SIGMA * SIGMA_PHI_DEG:.1f}°", fontsize=7, color="C4")
    ax.set_xlabel("구 지름 [µm]"); ax.set_title("계열 A: 상대 지표 vs 지름 (중심 깊이 250 µm)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    plain_ticks(ax, d)
    ax = axes[1]
    for f, lab in FACES.items():
        ax.loglog(d, [abs(r["time"][f]["max_abs_mK"]) for r in rows_a], "o-", label=f"ΔT {lab}, 최대 [mK]")
    ax.loglog(d, [abs(r["time"]["pump"]["abs_at_tstar_mK"]) for r in rows_a], "s:", label="ΔT 펌프면 (t*) [mK]")
    ax.axhline(N_SIGMA * SIGMA_T_MK, color="k", lw=0.8, ls=":"); ax.text(d[0], N_SIGMA * SIGMA_T_MK * 1.1, f"3σ_T = {N_SIGMA * SIGMA_T_MK:.0f} mK (1회)", fontsize=7)
    ax.set_xlabel("구 지름 [µm]"); ax.set_title("계열 A: 시간 영역 void 신호 (절대값)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    plain_ticks(ax, d)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "A_size_ladder.png", dpi=150); plt.close(fig)
    # A: far / pump void signal curves and Δφ(f)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(rows_a):
        c = _load_curves(r["id"])
        col = cmap(i / max(1, len(rows_a) - 1))
        axes[0].semilogx(c["t"] * 1e3, (c["dT_far"] - c["dT_far_base"]) * 1e3, color=col, label=f"⌀{r['d_eq_um']:g}")
        axes[1].semilogx(c["t"] * 1e3, (c["dT_pump"] - c["dT_pump_base"]) * 1e3, color=col, label=f"⌀{r['d_eq_um']:g}")
        axes[2].semilogx(FREQS, r["lockin"]["far"]["dphi_deg"], "o-", color=col, label=f"⌀{r['d_eq_um']:g}")
    axes[0].set_title("반대면 ΔT_far(void) − ΔT_far(기준) [mK]"); axes[1].set_title("펌프면 ΔT_pump(void) − ΔT_pump(기준) [mK]")
    axes[2].set_title("반대면 lock-in 위상 변화 [°]"); axes[2].set_xlabel("f [Hz]")
    for ax in axes[:2]:
        ax.set_xlabel("t [ms]"); ax.set_xlim(0.01, T_END * 1e3)
    for ax in axes:
        ax.grid(alpha=0.3, which="both"); ax.axhline(0, color="k", lw=0.6)
    axes[0].legend(fontsize=7, ncol=2, title="구 지름 [µm]")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "A_far_signal.png", dpi=150); plt.close(fig)
    # B: depth
    z = np.array([r["z_c_um"] for r in rows_b])
    rm = next(r for r in rows_b if r["z_c_requested_um"] == Z_REF_UM)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    xx = np.linspace(0, 1, 101)
    ax.plot(xx * 500, xx * (1 - xx) / 0.25, "k:", label="1-D 예측 x(1−x)/0.25")
    ax.plot(z, [r["dk_pct"] / rm["dk_pct"] for r in rows_b], "o-", label="Δk/k")
    for m, lab, mk in (("step_intercept", "Δt_int", "s"), ("parker", "Δt½", "d"), ("two_face", "ΔR_th", "^")):
        ax.plot(z, [r["indicators"][m]["delta"] / rm["indicators"][m]["delta"] for r in rows_b], mk + "--", label=lab)
    ax.plot(z, [r["time"]["pump"]["abs_at_tstar_mK"] / rm["time"]["pump"]["abs_at_tstar_mK"] for r in rows_b], "v-.", label="펌프면 ΔT(t*)")
    ax.plot(z, [r["lockin"]["far"]["max_dphi_deg"] / rm["lockin"]["far"]["max_dphi_deg"] for r in rows_b], "x-", label="반대면 Δφ 최대")
    ax.set_xlabel("void 중심 깊이 [µm]"); ax.set_ylabel("감도 비 (중심 깊이 250 µm = 1)"); ax.set_title(f"계열 B: 기준 구 ⌀{rm['d_eq_um']:g} µm 의 깊이 의존")
    ax.set_yscale("symlog", linthresh=1.0); ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    ax = axes[1]
    for i, r in enumerate(rows_b):
        c = _load_curves(r["id"])
        col = plt.get_cmap("plasma")(i / max(1, len(rows_b) - 1))
        ax.semilogx(c["t"] * 1e3, (c["dT_far"] - c["dT_far_base"]) * 1e3, color=col, label=f"반대면, z = {r['z_c_um']:g}")
        ax.semilogx(c["t"] * 1e3, (c["dT_pump"] - c["dT_pump_base"]) * 1e3, color=col, ls="--", label=f"펌프면, z = {r['z_c_um']:g}")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("void − 기준 [mK] (symlog)"); ax.set_title("계열 B: 시간 영역 void 신호"); ax.grid(alpha=0.3, which="both")
    ax.set_yscale("symlog", linthresh=1.0); ax.axhline(0, color="k", lw=0.6); ax.set_xlim(0.01, T_END * 1e3); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "B_depth.png", dpi=150); plt.close(fig)
    # C: shape bars
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [SHAPE_LABEL[r["shape"]] for r in rows_c]
    x = np.arange(len(rows_c))
    series = [("|Δk/k| [%]", [abs(r["dk_pct"]) for r in rows_c]), ("|Δt_int| [%]", [abs(r["indicators"]["step_intercept"]["rel_pct"]) for r in rows_c]),
              ("|ΔR_th| [%]", [abs(r["indicators"]["two_face"]["rel_pct"]) for r in rows_c]),
              ("|Δφ| 반대면 [°]", [abs(r["lockin"]["far"]["max_dphi_deg"]) for r in rows_c]),
              ("|ΔT| 반대면 최대 [mK]", [abs(r["time"]["far"]["max_abs_mK"]) for r in rows_c]),
              ("|ΔT| 펌프면 최대 [mK]", [abs(r["time"]["pump"]["max_abs_mK"]) for r in rows_c])]
    w = 0.8 / len(series)
    for i, (lab, vals) in enumerate(series):
        axes[0].bar(x + (i - len(series) / 2 + 0.5) * w, vals, w, label=lab)
    axes[0].set_yscale("log"); axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8); axes[0].grid(axis="y", alpha=0.3, which="both")
    axes[0].set_title(f"계열 C: 같은 부피 (구 ⌀{rm['d_eq_um']:g} µm), 중심 깊이 250 µm"); axes[0].legend(fontsize=7)
    for i, r in enumerate(rows_c):
        c = _load_curves(r["id"])
        axes[1].semilogx(c["t"] * 1e3, (c["dT_far"] - c["dT_far_base"]) * 1e3, color=f"C{i}", label=f"반대면, {SHAPE_LABEL[r['shape']]}")
        axes[1].semilogx(c["t"] * 1e3, (c["dT_pump"] - c["dT_pump_base"]) * 1e3, color=f"C{i}", ls="--", label=f"펌프면, {SHAPE_LABEL[r['shape']]}")
    axes[1].set_xlabel("t [ms]"); axes[1].set_ylabel("void − 기준 [mK]"); axes[1].grid(alpha=0.3, which="both"); axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlim(0.01, T_END * 1e3); axes[1].legend(fontsize=6, ncol=2); axes[1].set_title("계열 C: 시간 영역 void 신호")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "C_shape.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
