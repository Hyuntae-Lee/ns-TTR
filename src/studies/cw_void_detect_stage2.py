"""CW void-detection study, stage 2 (docs/cw-void-detection-study-plan.md §8): single off-axis void, 3-D.

For every condition (shape x r_off x centre depth) the smallest detectable equal-volume diameter is searched for two
criteria (plan §8): k_eff (far-face / two-face probing, |Δk/k| >= 3 σ_k of the step-FFT path) and the pump-face ΔT
change (pump-face probing, |ΔT_pump - ΔT_pump,base| max >= 3 σ_T).  Every run records all plan §4 numbers as
reference data (k_eff, lock-in phase/amplitude changes, ΔT changes, ΔR/R changes = c_tr ΔT, R_th).

Search (accuracy is never traded for time; only the number of runs is reduced):
  * a bracket [d_lo (not detected), d_hi (detected)] is narrowed until d_hi - d_lo <= TOL_UM, the threshold is then
    the log-log interpolation inside that bracket;
  * the next diameter is the log-log interpolation / extrapolation of the runs so far (|signal| ~ d^p) instead of
    halving, placed so that one more run is likely to close the bracket;
  * the first diameter of a condition comes from the nearest finished condition (stage-1 on-axis cases included);
  * both criteria share every run of a condition;
  * far-face indicators are mirror-symmetric in z <-> L - z (stage 1).  One check run (sphere, r_off 20, z 475 at the
    diameter of a z 25 run) verifies it in 3-D; if it holds, the k_eff thresholds of z 400 / 475 are those of 100 / 25
    and only the pump-face criterion is searched there.  If it fails, k_eff is searched at every depth.
  * N_WORKERS runs in parallel (two concurrent 3-D runs: 114 min each against 85 min alone).

Void-free reference: solver.baseline_config of the same configuration (2-D, same r-z grid).  Checked on
2026-10-27 against the 3-D run with the void filled with copper: k_eff identical to 1e-11 (stage2_timing).

Outputs (results/cw_void_detect/stage2):  cases/<id>.json|npz, run.log, status.md (brackets, rewritten after every
run), thresholds.csv.  Finished runs are cached by id; a restart rebuilds the search state from the cache.
Run from src:   .venv\\Scripts\\python.exe studies\\cw_void_detect_stage2.py [--plan] [--workers N]
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cw_void_detect as s1  # noqa: E402  (stage 1: paths, constants, criteria helpers)

from ttr_sim import VoidSpec  # noqa: E402
from ttr_sim import analysis as an  # noqa: E402
from ttr_sim.materials import COPPER, COPPER_C_TR_PROBE  # noqa: E402
from ttr_sim.solver import baseline_config  # noqa: E402
from ttr_sim.solver3d import count_unknowns, fit_resolution, needs_3d  # noqa: E402
from ttr_sim.transmission import cw_config, lockin_dataset, run_step_response  # noqa: E402

UM, L_UM = s1.UM, s1.L / s1.UM
OUT2 = s1.OUT / "stage2"
CASES2 = OUT2 / "cases"
SEED_DIR = s1.OUT / "stage2_timing"          # the timing run (sphere 44.5 um, r_off 20, z 250) is reused as a case
SHAPES2 = {"sphere": 1.0, "disc_1_4": 0.25}
R_OFFS_UM = [10, 20, 30, 40]                 # 40 = centre on the glass wall: about half of the void stays in the copper
DEPTHS2_UM = [25, 100, 250, 400, 475]
MIRROR = {400: 100, 475: 25}
SYM_CHECK = ("sphere", 20, 25, 475)
TOL_UM = 1.0                                 # final bracket width
D_MIN_UM, D_MAX_UM = 5.0, 80.0               # search range of the equal-volume diameter (80 um = rod diameter)
MAX_RUNS = 9                                 # per condition
P_DEFAULT = {"k": 3.3, "p": 3.0}             # |signal| ~ d^p when only one run is known (stage 1: k ~ d^3.3)
SYM_TOL_PCT = 0.01                           # |Δk/k(z) - Δk/k(L - z)| allowed by the symmetry check (absolute %)
C_TR = abs(COPPER_C_TR_PROBE)


# ----------------------------------------------------------------------------- cases
def case_id(shape: str, r_off: float, z: float, d: float) -> str:
    return f"S2_{shape}_r{r_off:g}_z{z:g}_d{d:.1f}"


def make_case2(shape: str, r_off: float, z: float, d: float) -> dict:
    c = s1.make_case(case_id(shape, r_off, z, d), "S2", f"{shape} ⌀{d:.1f} µm, r_off {r_off:g} µm, 중심 깊이 {z:g} µm",
                     d, z, SHAPES2[shape], shape)
    c.update(r_off_um=float(r_off), z_req_um=float(z), copper_frac=copper_fraction(c["r_half_um"], c["t_um"], r_off))
    return c


def copper_fraction(r_half: float, t: float, r_off: float, n: int = 400_000) -> float:
    """Volume fraction of the spheroid that lies inside the copper rod (the 3-D material map keeps the rest glass)."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(n, 3))
    p = u / np.linalg.norm(u, axis=1)[:, None] * rng.random(n)[:, None] ** (1.0 / 3.0)
    x, y = r_off + p[:, 0] * r_half, p[:, 1] * r_half
    return float(np.mean(x * x + y * y <= (s1.R_CU / UM) ** 2))


def void_spec2(c: dict) -> VoidSpec:
    return VoidSpec(enabled=True, depth=(c["z_c_um"] - 0.5 * c["t_um"]) * UM, thickness=c["t_um"] * UM,
                    r_half=c["r_half_um"] * UM, r_center=c["r_off_um"] * UM, theta=0.0, k=s1.K_VOID, shape="ellipse")


def case_config(c: dict):
    return fit_resolution(cw_config(void=void_spec2(c), f_min=s1.F_MIN, t_end=s1.T_END, dz=s1.DZ))


def _seed_void(c: dict, cfg):
    """The stage-2 timing run as the void run of its case (same configuration, checked by the unknown count)."""
    if c["id"] != case_id("sphere", 20, 250, 44.5) or not (SEED_DIR / "pair_void.npz").exists():
        return None
    meta = json.loads((SEED_DIR / "pair_void.json").read_text(encoding="utf-8"))
    if count_unknowns(cfg) != (10296, 12, 123552):
        return None
    z = np.load(SEED_DIR / "pair_void.npz")
    resp = SimpleNamespace(t=z["t"], dT_pump=z["dT_pump"], dT_far=z["dT_far"], dT_diff=z["dT_pump"] - z["dT_far"], P_abs=float(z["P_abs"]),
                           energy_error=meta["Eerr"], warnings=[],
                           result=SimpleNamespace(n_cells=10296, n_steps=meta["steps"], diagnostics={}))
    ds = SimpleNamespace(H_far=z["H_far"], H_pump=z["H_pump"], P_abs=float(z["P_abs"]), source=["step_fft"] * len(s1.FREQS))
    return resp, ds, meta["wall_min"] * 60.0


def run_case2(c: dict) -> dict:
    """One void run (3-D) and its same-grid void-free reference (2-D); every number of plan §4 on the step-FFT
    lock-in path (the only one in 3-D, so the reference uses it as well)."""
    jpath = CASES2 / f"{c['id']}.json"
    if jpath.exists():
        return json.loads(jpath.read_text(encoding="utf-8"))
    cfg = case_config(c)
    seed = _seed_void(c, cfg)
    if seed is None:
        t0 = time.perf_counter()
        resp = run_step_response(cfg, store_fields=False)
        wall = time.perf_counter() - t0
        ds = lockin_dataset(resp, None, s1.FREQS)
    else:
        resp, ds, wall = seed
    bcfg = baseline_config(cfg)
    bresp = run_step_response(bcfg, store_fields=False)
    bds = lockin_dataset(bresp, None, s1.FREQS)
    t = resp.t
    assert np.array_equal(t, bresp.t), "void and reference runs must share the time grid"
    n2d, nth, nun = count_unknowns(cfg) if needs_3d(cfg) else (int(resp.result.n_cells), 1, int(resp.result.n_cells))
    res = dict(c)
    res.update(solver=("3-D" if needs_3d(cfg) else "2-D"), n_cells_rz=n2d, n_theta=nth, unknowns=nun,
               void_cells_setting=int(cfg.numerics.void_cells), n_steps=int(resp.result.n_steps), wall_s=wall,
               seeded_from_timing_run=seed is not None, P_abs_mW=resp.P_abs * 1e3,
               energy_error=float(resp.energy_error), energy_error_base=float(bresp.energy_error),
               lockin_source="step_fft", warnings="; ".join(resp.warnings))
    kv = an.invert_tlm(s1.FREQS, ds.H_far, COPPER.rho_cp, s1.L, s1.A, shell=s1.SHELL)["k_eff"]
    kb = an.invert_tlm(s1.FREQS, bds.H_far, COPPER.rho_cp, s1.L, s1.A, shell=s1.SHELL)["k_eff"]
    res["k_eff"] = {"step_fft": dict(k_void=kv, k_base=kb, dk_pct=(kv / kb - 1.0) * 100.0)}
    res["dk_pct"] = res["k_eff"]["step_fft"]["dk_pct"]
    res["time"] = {f: s1.time_signal(t, getattr(resp, f"dT_{f}"), getattr(bresp, f"dT_{f}"), s1.T_END) for f in s1.FACES}
    # ΔR/R change of the probe (c_tr ΔT; |c_tr| = 1.5e-4 /K), in units of 1e-6, same instants as the ΔT changes
    res["dRR_1e6"] = {f: dict(max_abs=C_TR * res["time"][f]["max_abs_mK"] * 1e-3 * 1e6,
                              at_tstar=C_TR * res["time"][f]["abs_at_tstar_mK"] * 1e-3 * 1e6) for f in ("pump", "far")}
    ratio_v, ratio_b = resp.dT_far[-1] / resp.dT_pump[-1], bresp.dT_far[-1] / bresp.dT_pump[-1]
    res["ratio_far_pump_tstar"] = dict(base=float(ratio_b), void=float(ratio_v), rel_pct=float((ratio_v / ratio_b - 1.0) * 100.0))
    amp_v, amp_b = s1._lockin_amp_mK(ds), s1._lockin_amp_mK(bds)
    ph_v, ph_b = s1._phase_deg(ds), s1._phase_deg(bds)
    lk = {}
    for f in ("pump", "far"):
        dphi = ph_v[f] - ph_b[f]
        damp = (amp_v[f] / amp_b[f] - 1.0) * 100.0
        ip, ia = int(np.argmax(np.abs(dphi))), int(np.argmax(np.abs(damp)))
        lk[f] = dict(dphi_deg=dphi.tolist(), damp_pct=damp.tolist(), amp_base_mK=amp_b[f].tolist(), phase_base_deg=ph_b[f].tolist(),
                     max_dphi_deg=float(dphi[ip]), f_max_dphi=float(s1.FREQS[ip]), amp_at_f_dphi_mK=float(amp_b[f][ip]),
                     max_damp_pct=float(damp[ia]), f_max_damp=float(s1.FREQS[ia]), amp_at_f_damp_mK=float(amp_b[f][ia]))
    res["lockin"] = lk
    av = an.analyze(t, resp.dT_pump, resp.dT_far, resp.P_abs, s1.L, s1.A, COPPER.rho_cp, spot_radius=cfg.laser.w,
                    freqs=s1.FREQS, H_far=ds.H_far, shell=s1.SHELL)
    ab = an.analyze(t, bresp.dT_pump, bresp.dT_far, bresp.P_abs, s1.L, s1.A, COPPER.rho_cp, spot_radius=cfg.laser.w,
                    freqs=s1.FREQS, H_far=bds.H_far, shell=s1.SHELL)
    ind = {}
    for m, (b, v, dlt, rel) in an.void_indicators(ab, av).items():
        ind[m] = dict(key=an.INDICATOR_KEYS[m], base=b, void=v, delta=dlt, rel_pct=rel * 100.0)
    ind["slope_max_K_per_s"] = float(ab["step_intercept"]["slope"])
    res["indicators"] = ind
    CASES2.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CASES2 / f"{c['id']}.npz", t=t, dT_pump=resp.dT_pump, dT_far=resp.dT_far,
                        dT_pump_base=bresp.dT_pump, dT_far_base=bresp.dT_far, freqs=s1.FREQS,
                        H_far=ds.H_far, H_pump=ds.H_pump, H_far_base=bds.H_far, H_pump_base=bds.H_pump, P_abs=resp.P_abs)
    jpath.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- criteria and search
def thresholds() -> dict:
    sk = s1.reference_via()["sigma_k_pct"]["step_fft"]
    return {"k": s1.N_SIGMA * sk, "p": s1.N_SIGMA * s1.SIGMA_T_MK}


def signal(r: dict, crit: str) -> float:
    return abs(r["dk_pct"]) if crit == "k" else abs(r["time"]["pump"]["max_abs_mK"])


def _loglog_x(p1, p2, target):
    (d1, s1_), (d2, s2_) = p1, p2
    s1_, s2_ = max(s1_, 1e-12), max(s2_, 1e-12)
    if abs(math.log(s2_) - math.log(s1_)) < 1e-12:
        return 0.5 * (d1 + d2)
    return math.exp(math.log(d1) + (math.log(target) - math.log(s1_)) * (math.log(d2) - math.log(d1)) / (math.log(s2_) - math.log(s1_)))


def _exponent(pts, crit):
    """(exponent, measured?) from the two runs nearest in d (used only for extrapolation); default when not sensible."""
    if len(pts) >= 2:
        (d1, a), (d2, b) = pts[0], pts[1]
        if d1 != d2 and a > 0 and b > 0:
            p = math.log(b / a) / math.log(d2 / d1)
            if 0.5 <= p <= 8.0:
                return p, True
    return P_DEFAULT[crit], False


def _margin(x, measured):
    """Step past the extrapolated crossing so that the next run lands on the other side: half the final bracket
    when the exponent is measured, 2 % when it is the default."""
    return 0.5 * TOL_UM if measured else max(0.5 * TOL_UM, 0.02 * x)


def bracket(pts, T):
    """(lo, hi) points: hi = smallest detected diameter, lo = largest undetected diameter below it."""
    det = [p for p in pts if p[1] >= T]
    hi = min(det) if det else None
    und = [p for p in pts if p[1] < T and (hi is None or p[0] < hi[0])]
    lo = max(und) if und else None
    return lo, hi


def search_state(pts, T, crit):
    """'done' with the result, or 'next' with the diameter of the next run."""
    lo, hi = bracket(pts, T)
    if hi is not None and lo is not None:
        if hi[0] - lo[0] <= TOL_UM + 1e-9:
            return dict(state="done", d=_loglog_x(lo, hi, T), lo=lo[0], hi=hi[0], kind="bracket")
        x = _loglog_x(lo, hi, T)
        w = hi[0] - lo[0]
        if x - lo[0] <= 0.6 * TOL_UM:
            x = lo[0] + TOL_UM                    # truth just above lo: close the bracket from above
        elif hi[0] - x <= 0.6 * TOL_UM:
            x = hi[0] - TOL_UM                    # truth just below hi: close it from below
        if len(pts) >= 5:                         # slow convergence (signal far from a power law): plain bisection
            x = 0.5 * (lo[0] + hi[0])
        x = min(max(x, lo[0] + 0.1), hi[0] - 0.1)
        return dict(state="next", d=round(x, 1), lo=lo[0], hi=hi[0])
    if hi is not None:                            # everything detected so far
        if hi[0] <= D_MIN_UM + 1e-9:
            return dict(state="done", d=None, lo=None, hi=hi[0], kind="below_min")
        near = sorted(pts)[:2]
        p, meas = _exponent(near, crit)
        x = hi[0] * (T / max(hi[1], 1e-12)) ** (1.0 / p)
        x -= _margin(x, meas)
        return dict(state="next", d=round(min(max(x, D_MIN_UM, 0.33 * hi[0]), hi[0] - TOL_UM), 1), lo=None, hi=hi[0])
    if lo is not None:                            # nothing detected so far
        if lo[0] >= D_MAX_UM - 1e-9:
            return dict(state="done", d=None, lo=lo[0], hi=None, kind="above_max")
        near = sorted(pts, reverse=True)[:2]
        p, meas = _exponent(near[::-1], crit)
        x = lo[0] * (T / max(lo[1], 1e-12)) ** (1.0 / p)
        x += _margin(x, meas)
        return dict(state="next", d=round(max(min(x, D_MAX_UM, 3.0 * lo[0]), lo[0] + TOL_UM), 1), lo=lo[0], hi=None)
    return dict(state="start")


class Study:
    def __init__(self):
        self.T = thresholds()
        self.runs = {}                                    # id -> result
        for p in CASES2.glob("S2_*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            self.runs[r["id"]] = r
        self.stage1 = [json.loads(p.read_text(encoding="utf-8")) for p in s1.CASES_DIR.glob("[ABC]_*.json")]
        self.conditions = [(s, ro, z) for s in SHAPES2 for ro in R_OFFS_UM for z in DEPTHS2_UM]
        self.inflight = {}                                # condition -> case id

    # runs of a condition, (d, signal) sorted by d
    def points(self, cond, crit, z_override=None):
        s, ro, z = cond
        z = z if z_override is None else z_override
        pts = [(r["d_eq_um"], signal(r, crit)) for r in self.runs.values()
               if r["shape"] == s and r["r_off_um"] == ro and r["z_req_um"] == z]
        return sorted(pts)

    def symmetry(self):
        """None (not checked yet), True / False, and the check run's pair."""
        s, ro, za, zb = SYM_CHECK
        ra = sorted((r for r in self.runs.values() if r["shape"] == s and r["r_off_um"] == ro and r["z_req_um"] == za), key=lambda r: r["id"])
        if not ra:
            return None, None
        d = ra[0]["d_eq_um"]
        rb = self.runs.get(case_id(s, ro, zb, d))
        if rb is None:
            return None, (s, ro, zb, d)
        return abs(ra[0]["dk_pct"] - rb["dk_pct"]) <= SYM_TOL_PCT, (ra[0]["dk_pct"], rb["dk_pct"], d)

    def criteria(self, cond):
        sym, _ = self.symmetry()
        if cond[2] in MIRROR and sym is not False:
            return ["p"]                                   # k_eff mirrored from L - z (pending the check if sym is None)
        return ["k", "p"]

    def first_guess(self, cond, crit):
        """Start of a condition: the finished point of the nearest condition (same shape; stage-1 on-axis cases
        count as r_off 0) whose signal is nearest the criterion, scaled with |signal| ~ d^p."""
        s, ro, z = cond
        T = self.T[crit]
        cand = [(r["d_eq_um"], signal(r, crit), r.get("r_off_um", 0.0), r.get("z_req_um", r["z_c_requested_um"]))
                for r in list(self.runs.values()) + self.stage1 if r["shape"] == s]
        cand = [c for c in cand if c[1] > 0]
        if not cand:
            return 44.5
        dist = lambda c: abs(c[2] - ro) / 10.0 + abs(c[3] - z) / 50.0  # noqa: E731
        dmin = min(dist(c) for c in cand)
        near = [c for c in cand if dist(c) <= dmin + 1e-9]
        d0, s0 = min(near, key=lambda c: abs(math.log(c[1] / T)))[:2]
        x = d0 * (T / s0) ** (1.0 / P_DEFAULT[crit])
        return round(min(max(x, D_MIN_UM), D_MAX_UM), 1)

    def next_run(self, cond):
        """Diameter of the next run of `cond`, or None when its searches are finished (or it has a run in flight)."""
        if cond in self.inflight:
            return None
        s, ro, z = cond
        sym, chk = self.symmetry()
        if sym is None and chk is not None and cond == SYM_CHECK[:2] + (SYM_CHECK[3],):
            return chk[3]                                  # the symmetry check run first
        n = len(self.points(cond, "k"))
        if n >= MAX_RUNS:
            return None
        for crit in self.criteria(cond):
            st = search_state(self.points(cond, crit), self.T[crit], crit)
            if st["state"] == "next":
                return st["d"]
            if st["state"] == "start":
                return self.first_guess(cond, crit)
        return None

    def priority(self, cond):
        s, ro, z = cond
        return (list(SHAPES2).index(s), abs(ro - 20) / 10.0 + [250, 100, 25, 400, 475].index(z) * 0.6, ro, z)

    def result(self, cond, crit):
        s, ro, z = cond
        if crit == "k" and z in MIRROR and self.symmetry()[0]:
            st = search_state(self.points((s, ro, MIRROR[z]), "k"), self.T["k"], "k")
            return dict(st, mirrored_from=MIRROR[z])
        if crit == "k" and z in MIRROR and self.symmetry()[0] is None:
            return dict(state="sym_pending")
        return search_state(self.points(cond, crit), self.T[crit], crit)

    def write_status(self):
        sym, info = self.symmetry()
        lines = ["# 2단계 진행 상황 (자동 갱신)", "", time.strftime("갱신: %Y-%m-%d %H:%M"), "",
                 f"기준: k_eff |Δk/k| ≥ {self.T['k']:.2f} % (3σ_k, 스텝 FFT 경로), 펌프면 ΔT 변화 ≥ {self.T['p']:.1f} mK (3σ_T). "
                 f"구간 폭 ≤ {TOL_UM:g} µm 에서 종료, 탐색 범위 {D_MIN_UM:g}–{D_MAX_UM:g} µm.", "",
                 f"깊이 대칭 확인: {'대기' if sym is None else ('성립' if sym else '불성립')} {info if info else ''}", "",
                 f"완료된 3-D 계산: {len(self.runs)} 회 (총 {sum(r['wall_s'] for r in self.runs.values()) / 3600:.1f} h, 실행 중 {len(self.inflight)})", "",
                 "| 형태 | r_off | z | k_eff 한계 | 펌프면 ΔT 한계 | 계산 횟수 |", "|---|---|---|---|---|---|"]
        for cond in sorted(self.conditions, key=self.priority):
            cells = []
            for crit in ("k", "p"):
                if crit not in self.criteria(cond) and not (crit == "k" and cond[2] in MIRROR):
                    cells.append("—"); continue
                st = self.result(cond, crit)
                mir = f" (z {st['mirrored_from']} 대칭)" if "mirrored_from" in st else ""
                if st["state"] == "done":
                    txt = {"bracket": lambda: f"**{st['d']:.1f} µm** [{st['lo']:.1f}, {st['hi']:.1f}]",
                           "below_min": lambda: f"≤ {D_MIN_UM:g} µm", "above_max": lambda: f"> {D_MAX_UM:g} µm"}[st["kind"]]()
                else:
                    txt = {"next": f"진행 [{st.get('lo')}, {st.get('hi')}]", "sym_pending": "대칭 확인 대기"}.get(st["state"], "대기")
                cells.append(txt + mir)
            n = len(self.points(cond, "k"))
            lines.append(f"| {cond[0]} | {cond[1]} | {cond[2]} | {cells[0]} | {cells[1]} | {n}{' +1' if cond in self.inflight else ''} |")
        (OUT2 / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_csv(self):
        keys = ["id", "shape", "r_off_um", "z_req_um", "z_c_um", "d_eq_um", "r_half_um", "t_um", "copper_frac", "dk_pct",
                "dT_pump_max_mK", "dT_far_max_mK", "dT_diff_max_mK", "dRR_pump_1e6", "dRR_far_1e6", "dphi_far_deg", "f_dphi_far",
                "damp_far_pct", "dphi_pump_deg", "damp_pump_pct", "R_th_rel_pct", "t_int_rel_pct", "t_half_rel_pct", "unknowns", "wall_min"]
        with open(OUT2 / "runs.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for r in sorted(self.runs.values(), key=lambda r: (r["shape"], r["r_off_um"], r["z_req_um"], r["d_eq_um"])):
                bp = s1.best_phase(r["lockin"]["far"], s1.SIGMA_T_MK)
                w.writerow([r["id"], r["shape"], r["r_off_um"], r["z_req_um"], f"{r['z_c_um']:.2f}", r["d_eq_um"], f"{r['r_half_um']:.2f}",
                            f"{r['t_um']:.2f}", f"{r['copper_frac']:.3f}", f"{r['dk_pct']:.4f}",
                            f"{r['time']['pump']['max_abs_mK']:.3f}", f"{r['time']['far']['max_abs_mK']:.3f}", f"{r['time']['diff']['max_abs_mK']:.3f}",
                            f"{r['dRR_1e6']['pump']['max_abs']:.3f}", f"{r['dRR_1e6']['far']['max_abs']:.3f}", f"{bp['dphi']:.3f}", f"{bp['f']:.0f}",
                            f"{r['lockin']['far']['max_damp_pct']:.3f}", f"{r['lockin']['pump']['max_dphi_deg']:.3f}", f"{r['lockin']['pump']['max_damp_pct']:.3f}",
                            f"{r['indicators']['two_face']['rel_pct']:.3f}", f"{r['indicators']['step_intercept']['rel_pct']:.3f}",
                            f"{r['indicators']['parker']['rel_pct']:.3f}", r["unknowns"], f"{r['wall_s'] / 60:.1f}"])


def log2(msg: str):
    print(msg, flush=True)
    with open(OUT2 / "run.log", "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def plan():
    """Dry run: grid sizes over the search range, copper fractions and the first guesses (no solve)."""
    st = Study()
    print(f"thresholds {st.T}")
    for s in SHAPES2:
        for ro in R_OFFS_UM:
            for d in (D_MIN_UM, 20.0, 44.5, D_MAX_UM):
                for z in (25, 250):
                    c = make_case2(s, ro, z, d)
                    cfg = case_config(c)
                    print(f"{s:9s} r_off {ro:2d} z {z:3d} d {d:5.1f}: r_half {c['r_half_um']:5.1f} t {c['t_um']:5.1f} z_c {c['z_c_um']:6.1f} "
                          f"Cu {c['copper_frac']:.2f}  unknowns {count_unknowns(cfg)}  void_cells {cfg.numerics.void_cells}")
    for cond in sorted(st.conditions, key=st.priority):
        print(cond, "first run d =", st.next_run(cond))


def main(n_workers: int):
    OUT2.mkdir(parents=True, exist_ok=True)
    st = Study()
    log2(f"start: {len(st.runs)} cached runs, thresholds {st.T}, workers {n_workers}")
    seed_id = case_id("sphere", 20, 250, 44.5)
    if seed_id not in st.runs:
        r = run_case2(make_case2("sphere", 20, 250, 44.5))
        st.runs[r["id"]] = r
        log2(f"seeded {seed_id} from the timing run: Δk/k {r['dk_pct']:+.3f} %, pump ΔT {r['time']['pump']['max_abs_mK']:+.2f} mK")
    st.write_status()
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        while True:
            for cond in sorted(st.conditions, key=st.priority):
                if len(futures) >= n_workers:
                    break
                d = st.next_run(cond)
                if d is None:
                    continue
                c = make_case2(cond[0], cond[1], cond[2], d)
                if c["id"] in st.runs:                     # should not happen; guards against an endless loop
                    log2(f"skip duplicate {c['id']}"); continue
                st.inflight[cond] = c["id"]
                futures[ex.submit(run_case2, c)] = cond
                log2(f"submit {c['id']}  (z_c {c['z_c_um']:.1f}, Cu {c['copper_frac']:.2f})")
            st.write_status()
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fu in done:
                cond = futures.pop(fu)
                st.inflight.pop(cond, None)
                try:
                    r = fu.result()
                except Exception as e:                     # record and stop scheduling this condition
                    log2(f"FAILED {cond}: {e!r}")
                    st.conditions.remove(cond)
                    continue
                st.runs[r["id"]] = r
                log2(f"done {r['id']}: Δk/k {r['dk_pct']:+.3f} %  pump ΔT {r['time']['pump']['max_abs_mK']:+.2f} mK  "
                     f"far ΔT {r['time']['far']['max_abs_mK']:+.2f} mK  {r['wall_s'] / 60:.1f} min")
            st.write_csv()
    st.write_status()
    st.write_csv()
    log2("stage 2 finished")


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    else:
        main(int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2)
