"""Analysis of the superposition study (reads results/superposition and results/multi_void, both read-only
except for the analysis outputs written to results/superposition).

A1  computed multi-void signal versus the sum of the single-void curves: pairs P1/P2/P3 and the first
    study's B, C, D cases (peak ratio, peak-time shift, relative L2 curve difference, detection verdicts from
    both curves)
A2  shape scaling at 120 um (P4) and 70 um (E): peak versus the dipole polarisability V/(1 - L_z), and the
    normalised curve shapes
A3  single-void library s(t; depth, r) from the dense on-axis depth series and the radial factors measured at
    70/90/150 um; prediction of the random arrangements F / F0 by summation; comparison with the computed runs

Outputs: results/superposition/analysis.json, analysis.md, summary/*.png
Run from the src directory:   .venv\\Scripts\\python.exe studies\\superposition_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "studies"))

from detection_map import CRITERIA_DEFAULT, N_SIGMA, judge  # noqa: E402
from multi_void import DEEP_DEPTHS, WINDOW_US, setup_mpl  # noqa: E402
from multi_void_report import depolarisation_z, polarisability_um3  # noqa: E402
from superposition import OUT, P0_DEPTHS, P0P, P1_GAPS, P2_SPACINGS, P3_RADII, P4_SHAPES  # noqa: E402
from ttr_sim.materials import COPPER_C_TR_PROBE  # noqa: E402

MV = SRC / "results" / "multi_void" / "cases"
SP = OUT / "cases"
SUM = OUT / "summary"
SIGMA3 = N_SIGMA * CRITERIA_DEFAULT["sigma"]
C_MIN = CRITERIA_DEFAULT["c_min"]


# ----------------------------------------------------------------------------- data access
class Cases:
    def __init__(self):
        self.rows, self.src = {}, {}
        for folder, tag in ((MV, "multi_void"), (SP, "superposition")):
            for jp in sorted(folder.glob("*.json")):
                r = json.loads(jp.read_text(encoding="utf-8"))
                self.rows[r["id"]] = r
                self.src[r["id"]] = folder
        self._npz = {}

    def has(self, cid):
        return cid in self.rows

    def npz(self, cid):
        if cid not in self._npz:
            self._npz[cid] = dict(np.load(self.src[cid] / f"{cid}.npz"))
        return self._npz[cid]

    def diff(self, cid):
        return self.npz(cid)["dT_diff"]

    def base(self, cid):
        return self.npz(cid)["dT_base"]

    def times(self, cid):
        return self.npz(cid)["times"]


def metrics_from_curves(dT_b: np.ndarray, dT: np.ndarray, t: np.ndarray) -> dict:
    """Headline metrics and the detection verdict from a baseline curve and a void-minus-baseline curve
    (same definitions as ttr_sim.solver.void_signal and detection_map.judge)."""
    valid = dT_b > 1e-3 * dT_b.max()
    contrast = np.zeros_like(dT)
    contrast[valid] = dT[valid] / dT_b[valid]
    i_pk = int(np.argmax(np.abs(dT)))
    i_c = int(np.argmax(np.abs(contrast)))
    row = dict(dT_base_max_K=float(dT_b.max()), void_minus_base_K=float(dT[i_pk]), t_peak_s=float(t[i_pk]),
               ratio=float(dT[i_pk] / dT_b[i_pk]) if dT_b[i_pk] > 0 else float("nan"),
               contrast_max=float(contrast[i_c]), t_contrast_max_s=float(t[i_c]))
    row.update(judge(row, **CRITERIA_DEFAULT))
    return row


def verdict(m: dict) -> str:
    return "가능" if m["detectable"] else f"불가({m['limiting']})"


def compare(C: Cases, cid: str, parts: list[str]) -> dict | None:
    """Computed case `cid` against the sum of the single-void curves `parts` (None if anything is missing)."""
    if not C.has(cid) or not all(C.has(p) for p in parts):
        return None
    t = C.times(cid)
    dT = C.diff(cid)
    s = np.zeros_like(dT)
    for p in parts:
        s += np.interp(t, C.times(p), C.diff(p))
    m_c = metrics_from_curves(C.base(cid), dT, t)
    m_s = metrics_from_curves(C.base(cid), s, t)
    i_c, i_s = int(np.argmax(np.abs(dT))), int(np.argmax(np.abs(s)))
    return dict(
        id=cid, parts=parts, n_voids=C.rows[cid]["n_voids"],
        peak_computed_mK=float(dT[i_c] * 1e3), peak_sum_mK=float(s[i_s] * 1e3),
        R_peak=float(dT[i_c] / s[i_s]), dt_peak_us=float((t[i_c] - t[i_s]) * 1e6),
        l2_rel=float(np.linalg.norm(dT - s) / np.linalg.norm(dT)),
        computed=dict(noise_margin=m_c["noise_margin"], ratio_margin=m_c["ratio_margin"], detectable=m_c["detectable"], verdict=verdict(m_c)),
        summed=dict(noise_margin=m_s["noise_margin"], ratio_margin=m_s["ratio_margin"], detectable=m_s["detectable"], verdict=verdict(m_s)),
        verdict_flips=bool(m_c["detectable"] != m_s["detectable"]),
    )


# ----------------------------------------------------------------------------- A1
def a1(C: Cases, plt) -> dict:
    out = {"P1": [], "P2": [], "P3": [], "B": [], "C": [], "D": []}
    for g in P1_GAPS:
        cid = f"P1_g{int(g):02d}"
        r = compare(C, cid, ["S_axis_d070", f"S_axis_d{int(80 + g):03d}"])
        if r:
            r["gap_um"] = g
            out["P1"].append(r)
    for s in P2_SPACINGS:
        cid = f"P2_s{int(s):02d}"
        r = compare(C, cid, ["A_d070_r25", "A_d070_r25"])
        if r:
            r["spacing_um"] = s
            out["P2"].append(r)
    r = compare(C, "P1_g10", ["S_axis_d070", "S_axis_d090"])
    if r:
        r["rc_lower_um"] = 0.0
        out["P3"].append(r)
    for rc in P3_RADII:
        cid = f"P3_r{int(rc):02d}"
        lower = "A_d090_r25" if rc == 25 else f"S_d090_r{int(rc):02d}"
        r = compare(C, cid, ["S_axis_d070", lower])
        if r:
            r["rc_lower_um"] = rc
            out["P3"].append(r)
    for n in (1, 3, 6, 9):
        cid = f"B_N{n}"
        r = compare(C, cid, ["A_d070_axis"] + [f"A_d{int(d):03d}_r25" for d in DEEP_DEPTHS[:n]])
        if r:
            r["n_added"] = n
            out["B"].append(r)
    r = compare(C, "C_stack3", ["S_axis_d070", "S_axis_d090", "S_axis_d105", "S_axis_d120"])
    r_old = compare(C, "C_stack3", ["A_d070_axis", "A_d090_r25", "A_d105_r25", "A_d120_r25"])
    if r and r_old:
        r["R_peak_with_r25_proxy"] = r_old["R_peak"]
        out["C"].append(r)
    for n in (2, 3, 6):
        cid = f"D_N{n}"
        r = compare(C, cid, ["A_d070_axis"] + ["A_d070_r25"] * (n - 1))
        if r:
            r["n_total"] = n
            out["D"].append(r)

    # ---- figure: R_peak versus distance
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    ax = axs[0]
    if out["P1"]:
        ax.plot([r["gap_um"] for r in out["P1"]], [r["R_peak"] for r in out["P1"]], "o-", label="P1 위아래 쌍 (축 위)")
    if out["C"]:
        ax.plot([5.0], [out["C"][0]["R_peak"]], "s", color="#d62728", ms=9, label="C_stack3 (적층 3개, 틈 10/5/5 μm)")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("두 void 사이 틈 g [μm]")
    ax.set_ylabel("피크 비율  계산 / 단독 합")
    ax.set_title("위아래로 겹칠 때")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axs[1]
    if out["P2"]:
        ax.plot([r["spacing_um"] for r in out["P2"]], [r["R_peak"] for r in out["P2"]], "o-", label="P2 나란히 쌍 (70 μm, r = 25 μm)")
    if out["D"]:
        ax.plot([25.0] * len(out["D"]), [r["R_peak"] for r in out["D"]], "^", color="#2ca02c", ms=9, label="D (축 + 고리, 중심 간격 25 μm): 2, 3, 6개")
        for r in out["D"]:
            ax.annotate(f"{r['n_total']}개", (25.0, r["R_peak"]), xytext=(6, -3), textcoords="offset points", fontsize=8)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("중심 간격 s [μm]  (반폭 10 μm: s = 20 이 접촉)")
    ax.set_title("같은 깊이에 나란히 놓일 때")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axs[2]
    if out["P3"]:
        ax.plot([r["rc_lower_um"] for r in out["P3"]], [r["R_peak"] for r in out["P3"]], "o-", label="P3 대각선 쌍 (70 μm 축 + 90 μm r)")
    if out["B"]:
        ax.plot([25.0] * len(out["B"]), [r["R_peak"] for r in out["B"]], "v", color="#9467bd", ms=9, label="B (70 μm 축 + 깊은 void N개, r = 25 μm)")
        for r in out["B"]:
            ax.annotate(f"N={r['n_added']}", (25.0, r["R_peak"]), xytext=(6, -3), textcoords="offset points", fontsize=8)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("아래 void 의 축 거리 r [μm]")
    ax.set_title("위아래 차폐에서 가로 분리로")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SUM / "fig_A1_ratio.png", dpi=120)
    plt.close(fig)

    # ---- figure: curves, computed vs sum
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for ax, key, title in zip(axs, ("P1", "P2", "P3"), ("P1 위아래 쌍", "P2 나란히 쌍", "P3 대각선 쌍")):
        cm = plt.get_cmap("plasma")
        for k, r in enumerate(out[key]):
            t = C.times(r["id"]) * 1e6
            s = np.zeros_like(t)
            for p in r["parts"]:
                s += np.interp(C.times(r["id"]), C.times(p), C.diff(p))
            col = cm(0.1 + 0.8 * k / max(1, len(out[key]) - 1))
            lab = {"P1": f"g = {r.get('gap_um', 0):g}", "P2": f"s = {r.get('spacing_um', 0):g}", "P3": f"r = {r.get('rc_lower_um', 0):g}"}[key]
            ax.plot(t, C.diff(r["id"]) * 1e3, color=col, label=f"{lab} μm 계산")
            ax.plot(t, s * 1e3, color=col, ls="--", alpha=0.8)
        ax.set_xlim(0, 400)
        ax.set_xlabel("t [μs]")
        ax.set_ylabel("void - baseline ΔT [mK]")
        ax.set_title(f"{title}: 계산(실선) 대 단독 합(점선)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SUM / "fig_A1_curves.png", dpi=120)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- A2
def a2(C: Cases, plt) -> dict:
    sets = {
        "70": [("A_d070_axis", (10.0, 10.0))] + [(f"E_rh{rh:g}_th{th:g}", (rh, th)) for rh, th in P4_SHAPES],
        "120": [("S_axis_d120", (10.0, 10.0))] + [(f"P4_rh{rh:g}_th{th:g}", (rh, th)) for rh, th in P4_SHAPES],
    }
    out = {}
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for col_i, (depth, items) in enumerate(sets.items()):
        items = [(cid, sh) for cid, sh in items if C.has(cid)]
        if len(items) < 3:
            continue
        pol = np.array([polarisability_um3(dict(rh=sh[0], thick=sh[1])) for _, sh in items])
        pk = np.array([abs(C.rows[cid]["void_minus_base_K"]) for cid, _ in items])
        slope, icpt = np.polyfit(np.log(pol), np.log(pk), 1)
        resid = np.std(np.log(pk) - np.log(pol) - np.mean(np.log(pk) - np.log(pol)))
        ref = next(cid for cid, sh in items if sh == (10.0, 10.0))
        t = C.times(ref)
        nref = C.diff(ref) / np.max(np.abs(C.diff(ref)))
        shapes = []
        for cid, sh in items:
            n = C.diff(cid) / np.max(np.abs(C.diff(cid)))
            shapes.append(dict(id=cid, rh=sh[0], thick=sh[1], L_z=depolarisation_z(sh[0], sh[1] / 2), pol_um3=float(polarisability_um3(dict(rh=sh[0], thick=sh[1]))),
                               peak_mK=float(C.rows[cid]["void_minus_base_K"] * 1e3), t_peak_us=float(C.rows[cid]["t_peak_s"] * 1e6),
                               l2_shape_vs_ref=float(np.linalg.norm(n - nref) / np.linalg.norm(nref))))
        out[depth] = dict(slope=float(slope), scatter=float(resid), shapes=shapes)
        ax = axs[0]
        ax.loglog(pol, pk * 1e3, "o" if depth == "70" else "s", label=f"깊이 {depth} μm: 기울기 {slope:.2f}, 산포 {resid * 100:.0f} %")
        xx = np.linspace(pol.min() * 0.8, pol.max() * 1.2, 30)
        ax.loglog(xx, np.exp(icpt) * xx ** slope * 1e3, ":", color="gray")
        ax = axs[1 + col_i]
        for cid, sh in items:
            ax.plot(t * 1e6, C.diff(cid) / np.max(np.abs(C.diff(cid))), label=f"({sh[0]:g}, {sh[1]:g})")
        ax.set_xlim(0, 400 if depth == "70" else 600)
        ax.set_xlabel("t [μs]")
        ax.set_ylabel("ΔT / 피크")
        ax.set_title(f"깊이 {depth} μm: 피크로 정규화한 곡선 (반폭, 두께)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axs[0].set_xlabel("쌍극자 분극률 V / (1 - L_z) [μm³]")
    axs[0].set_ylabel("|ΔT| 피크 [mK]")
    axs[0].set_title("형상 6종의 피크 대 분극률")
    axs[0].legend(fontsize=9)
    axs[0].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(SUM / "fig_A2_shape.png", dpi=120)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- A3
class Library:
    """s(t; depth, r) for the 10/10 um void: on-axis curves at every library depth (linear interpolation between
    the two neighbouring depths, exact at the nodes) times a radial factor f(depth, r) measured at 70/90/150 um
    (linear in r, linear in depth between the measured depths, constant outside)."""

    def __init__(self, C: Cases):
        self.C = C
        self.depths = np.array([d for d in P0_DEPTHS if C.has(f"S_axis_d{d:03d}")], dtype=float)
        self.t = C.times(f"S_axis_d{int(self.depths[0]):03d}")
        self.curves = {d: C.diff(f"S_axis_d{int(d):03d}") for d in self.depths}
        self.peaks = {d: float(np.max(np.abs(self.curves[d]))) for d in self.depths}
        # radial factors
        self.rf = {}
        for d, axis_id, lst in ((70.0, "A_d070_axis", [(10, "S_d070_r10"), (15, "S_d070_r15"), (20, "S_d070_r20"), (25, "A_d070_r25"), (28, "S_d070_r28")]),
                                (90.0, "S_axis_d090", [(10, "S_d090_r10"), (15, "S_d090_r15"), (20, "S_d090_r20"), (25, "A_d090_r25")]),
                                (150.0, "S_axis_d150", [(10, "S_d150_r10"), (20, "S_d150_r20"), (28, "S_d150_r28")])):
            if not C.has(axis_id):
                continue
            p0 = float(np.max(np.abs(C.diff(axis_id))))
            pts = [(0.0, 1.0, 0.0)]
            n0 = C.diff(axis_id) / p0
            for r, cid in lst:
                if C.has(cid):
                    c = C.diff(cid)
                    pts.append((float(r), float(np.max(np.abs(c)) / p0), float(np.linalg.norm(c / np.max(np.abs(c)) - n0) / np.linalg.norm(n0))))
            if len(pts) >= 2:
                self.rf[d] = sorted(pts)
        self.rf_depths = sorted(self.rf)

    def radial_factor(self, depth: float, r: float) -> tuple[float, bool]:
        def f_at(d):
            pts = self.rf[d]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if r <= xs[-1]:
                return float(np.interp(r, xs, ys))
            return float(ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (r - xs[-1]))     # linear extrapolation beyond the last r
        ds = self.rf_depths
        if not ds:
            return 1.0, True
        extrap = r > max(p[0] for p in self.rf[ds[0]])
        if depth <= ds[0]:
            return f_at(ds[0]), extrap
        if depth >= ds[-1]:
            return f_at(ds[-1]), extrap
        i = int(np.searchsorted(ds, depth)) - 1
        w = (depth - ds[i]) / (ds[i + 1] - ds[i])
        return (1 - w) * f_at(ds[i]) + w * f_at(ds[i + 1]), extrap

    def curve(self, depth: float, r: float) -> tuple[np.ndarray, list[str]]:
        notes = []
        d = depth
        if d < self.depths[0] or d > self.depths[-1]:
            notes.append(f"depth {depth:.1f} outside library ({self.depths[0]:g}..{self.depths[-1]:g}): nearest node used")
            d = float(np.clip(d, self.depths[0], self.depths[-1]))
        i = int(np.searchsorted(self.depths, d))
        if i == 0 or self.depths[min(i, len(self.depths) - 1)] == d:
            base = self.curves[float(self.depths[min(i, len(self.depths) - 1)])].copy()
        else:
            d0, d1 = self.depths[i - 1], self.depths[i]
            w = (d - d0) / (d1 - d0)
            base = (1 - w) * self.curves[d0] + w * self.curves[d1]
        f, extrap = self.radial_factor(depth, r)
        if extrap:
            notes.append(f"r {r:.1f} beyond the measured radii (<= 28): linear extrapolation of the radial factor")
        if self.rf_depths and depth < self.rf_depths[0] and r > 0:
            notes.append(f"depth {depth:.1f} shallower than the radial-factor measurements (>= {self.rf_depths[0]:g}): factor of {self.rf_depths[0]:g} um used")
        return base * f, notes


def a3(C: Cases, plt) -> dict:
    lib = Library(C)
    out = dict(library=dict(depths=lib.depths.tolist(), peaks_mK={f"{d:g}": p * 1e3 for d, p in lib.peaks.items()},
                            radial_factors={f"{d:g}": [dict(r=p[0], factor=p[1], l2_shape_vs_axis=p[2]) for p in pts] for d, pts in lib.rf.items()},
                            method="depth: linear interpolation between neighbouring on-axis curves (10 um grid); "
                                   "r: peak factor f(depth, r) linear in r and in depth between 70/90/150 um, constant outside; "
                                   "curve shape assumed independent of r (checked: l2_shape_vs_axis)"),
               predictions=[])
    seeds = [f"F_seed{s}" for s in range(1, 6)] + [f"F0_seed{s}" for s in (1, 2, 4, 5)]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for cid in seeds:
        if not C.has(cid):
            continue
        row = C.rows[cid]
        t = C.times(cid)
        pred = np.zeros_like(t)
        notes = []
        for k, v in enumerate(row["voids"]):
            c, n = lib.curve(v["depth"], v["rc"])
            pred += np.interp(t, lib.t, c)
            notes += [f"void {k + 1}: {x}" for x in n]
        m_c = metrics_from_curves(C.base(cid), C.diff(cid), t)
        m_p = metrics_from_curves(C.base(cid), pred, t)
        i_c, i_p = int(np.argmax(np.abs(C.diff(cid)))), int(np.argmax(np.abs(pred)))
        out["predictions"].append(dict(
            id=cid, depth_min_um=row["depth_min_um"], n_voids=row["n_voids"],
            peak_computed_mK=float(C.diff(cid)[i_c] * 1e3), peak_predicted_mK=float(pred[i_p] * 1e3),
            ratio_pred_over_computed=float(pred[i_p] / C.diff(cid)[i_c]), t_peak_computed_us=float(t[i_c] * 1e6), t_peak_predicted_us=float(t[i_p] * 1e6),
            l2_rel=float(np.linalg.norm(C.diff(cid) - pred) / np.linalg.norm(C.diff(cid))),
            computed=dict(noise_margin=m_c["noise_margin"], ratio_margin=m_c["ratio_margin"], verdict=verdict(m_c)),
            predicted=dict(noise_margin=m_p["noise_margin"], ratio_margin=m_p["ratio_margin"], verdict=verdict(m_p)),
            verdict_flips=bool(m_c["detectable"] != m_p["detectable"]), extrapolation_notes=notes,
            within_library=not any("shallower" in n for n in notes)))
        ax = axs[0] if cid.startswith("F_") else axs[1]
        ln, = ax.plot(t * 1e6, C.diff(cid) * 1e3, label=f"{cid} 계산")
        ax.plot(t * 1e6, pred * 1e3, ls="--", color=ln.get_color(), label=f"{cid} 예측")
    out["excluded"] = ["F0_seed3: a void breaks the surface (top at 0.7 um); the signal is that of a surface hole, outside the library"]
    for ax, title in zip(axs[:2], ("F (윗면 깊이 ≥ 70 μm)", "F0 (로드 전체, 시드 3 제외)")):
        ax.set_xlim(0, WINDOW_US)
        ax.set_xlabel("t [μs]")
        ax.set_ylabel("void - baseline ΔT [mK]")
        ax.set_title(f"무작위 10개: 계산(실선) 대 라이브러리 합 예측(점선) — {title}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    ax = axs[2]
    pc = [p["peak_computed_mK"] for p in out["predictions"]]
    pp = [p["peak_predicted_mK"] for p in out["predictions"]]
    ax.plot(pc, pp, "o")
    for p in out["predictions"]:
        ax.annotate(p["id"].replace("_seed", " s"), (p["peak_computed_mK"], p["peak_predicted_mK"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [0, max(pc + pp) * 1.1]
    ax.plot(lim, lim, ":", color="gray")
    ax.set_xlabel("계산 피크 [mK]")
    ax.set_ylabel("예측 피크 [mK]")
    ax.set_title("피크: 예측 대 계산")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SUM / "fig_A3_prediction.png", dpi=120)
    plt.close(fig)

    # library figure: depth curve of the peak and radial factors
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    ax.semilogy(lib.depths, [lib.peaks[d] * 1e3 for d in lib.depths], "o-", ms=3, label="축 위 단독 (10/10 μm), 피크")
    a_ids = [(C.rows[i]["depth_min_um"], abs(C.rows[i]["void_minus_base_K"]) * 1e3) for i in C.rows if i.startswith("A_d") and i.endswith("_r25")]
    if a_ids:
        a_ids.sort()
        ax.semilogy([d for d, _ in a_ids], [p for _, p in a_ids], "s--", ms=4, label="r = 25 μm 단독 (A 계열)")
    ax.set_xlabel("void 윗면 깊이 [μm]")
    ax.set_ylabel("|ΔT| 피크 [mK]")
    ax.set_title("단독 응답 라이브러리: 깊이 축")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax = axs[1]
    for d, pts in lib.rf.items():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"깊이 {d:g} μm")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("void 중심의 축 거리 r [μm]")
    ax.set_ylabel("피크 비율  f(r) = 피크(r) / 피크(축)")
    ax.set_title("반경 위치 인자")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SUM / "fig_A3_library.png", dpi=120)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- run table / checks
def run_table(C: Cases) -> list[dict]:
    rows = []
    for cid, r in C.rows.items():
        if C.src[cid] == SP:
            rows.append(dict(id=cid, series=r["series"], dim="3D" if r["is_3d"] else "2D", n_unknowns=r["n_unknowns"], wall_min=r["wall_s"] / 60,
                             energy_error=r["energy_error"], mean_iterations=r["mean_iterations"], peak_mK=r["void_minus_base_K"] * 1e3,
                             t_peak_us=r["t_peak_s"] * 1e6, verdict="가능" if r["detectable"] else f"불가({r['limiting']})"))
    return rows


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def write_md(A, runs, checks, geo):
    L = ["# 중첩 검증 스터디 분석 (superposition)", ""]
    L += ["## 회귀 확인", md_table(["케이스", "기준", "같은 시간축", "dT_diff 상대오차", "dT_base 상대오차", "결과"],
                                 [[c["id"], c["ref"], c["same_time_axis"], f"{c['rel_err_dT_diff']:.1e}", f"{c['rel_err_dT_base']:.1e}", "통과" if c["passed"] else "실패"] for c in checks]), ""]
    bad = {k: v for k, v in geo.items() if v}
    L += ["## 기하 검사", f"검사 {len(geo)}건, 위반 {len(bad)}건" + (": " + "; ".join(f"{k} ({'; '.join(v)})" for k, v in bad.items()) if bad else ""), ""]
    L += ["## 실행한 케이스", md_table(["케이스", "계열", "차원", "미지수", "분", "에너지 오차", "평균 반복", "피크 [mK]", "t_peak [μs]", "판정"],
                                   [[r["id"], r["series"], r["dim"], f"{r['n_unknowns']:,}", f"{r['wall_min']:.1f}", f"{r['energy_error']:.1e}", f"{r['mean_iterations']:.1f}",
                                     f"{r['peak_mK']:.4g}", f"{r['t_peak_us']:.0f}", r["verdict"]] for r in runs]), ""]
    L += ["## A1. 계산 대 단독 곡선의 합", ""]
    for key, xname, xkey in (("P1", "틈 g [μm]", "gap_um"), ("P2", "중심 간격 s [μm]", "spacing_um"), ("P3", "아래 void r [μm]", "rc_lower_um"),
                             ("B", "추가 N", "n_added"), ("C", "-", None), ("D", "총 개수", "n_total")):
        rows = A["A1"][key]
        if not rows:
            continue
        L += [f"### {key}", md_table(["케이스", xname, "계산 피크 [mK]", "합 피크 [mK]", "R_peak", "Δt_peak [μs]", "L2 상대", "판정(계산)", "판정(합)", "뒤바뀜"],
                                   [[r["id"], r.get(xkey, "-") if xkey else "-", f"{r['peak_computed_mK']:.4g}", f"{r['peak_sum_mK']:.4g}", f"{r['R_peak']:.3f}",
                                     f"{r['dt_peak_us']:+.1f}", f"{r['l2_rel']:.3f}", r["computed"]["verdict"], r["summed"]["verdict"], "예" if r["verdict_flips"] else ""] for r in rows]), ""]
        if key == "C":
            L += [f"C_stack3 의 합 대비 비율: 축 위 단독으로 {rows[0]['R_peak']:.3f} (r = 25 μm 대용으로는 {rows[0]['R_peak_with_r25_proxy']:.3f})", ""]
    L += ["## A2. 형상 비례", ""]
    for depth, d in A["A2"].items():
        L += [f"### 깊이 {depth} μm: 로그 기울기 {d['slope']:.3f}, 산포 {d['scatter'] * 100:.1f} %",
              md_table(["케이스", "반폭", "두께", "L_z", "V/(1-L_z) [μm³]", "피크 [mK]", "t_peak [μs]", "정규화 곡선 L2 (대 10/10)"],
                       [[s["id"], s["rh"], s["thick"], f"{s['L_z']:.3f}", f"{s['pol_um3']:,.0f}", f"{s['peak_mK']:.4g}", f"{s['t_peak_us']:.0f}", f"{s['l2_shape_vs_ref']:.3f}"] for s in d["shapes"]]), ""]
    L += ["## A3. 단독 라이브러리로 무작위 배치 예측", "", "방법: " + A["A3"]["library"]["method"], ""]
    L += ["반경 인자 f(depth, r) (피크 비율, 정규화 곡선 L2 대 축):", md_table(["깊이", "r", "f", "곡선 L2"], [[d, p["r"], f"{p['factor']:.3f}", f"{p['l2_shape_vs_axis']:.3f}"] for d, pts in A["A3"]["library"]["radial_factors"].items() for p in pts]), ""]
    L += [md_table(["케이스", "얕은 깊이", "계산 피크 [mK]", "예측 피크 [mK]", "예측/계산", "t_peak 계산/예측 [μs]", "L2 상대", "잡음 여유 계산/예측", "대비 여유 계산/예측", "판정 계산/예측", "외삽"],
                   [[p["id"], f"{p['depth_min_um']:.0f}", f"{p['peak_computed_mK']:.4g}", f"{p['peak_predicted_mK']:.4g}", f"{p['ratio_pred_over_computed']:.3f}",
                     f"{p['t_peak_computed_us']:.0f} / {p['t_peak_predicted_us']:.0f}", f"{p['l2_rel']:.3f}",
                     f"×{p['computed']['noise_margin']:.2f} / ×{p['predicted']['noise_margin']:.2f}", f"×{p['computed']['ratio_margin']:.2f} / ×{p['predicted']['ratio_margin']:.2f}",
                     f"{p['computed']['verdict']} / {p['predicted']['verdict']}", len(p["extrapolation_notes"])] for p in A["A3"]["predictions"]]), ""]
    L += ["제외: " + "; ".join(A["A3"]["excluded"]), ""]
    notes = [f"- {p['id']}: " + "; ".join(p["extrapolation_notes"]) for p in A["A3"]["predictions"] if p["extrapolation_notes"]]
    if notes:
        L += ["외삽 상세:"] + notes + [""]
    (OUT / "analysis.md").write_text("\n".join(L), encoding="utf-8")


def main():
    SUM.mkdir(parents=True, exist_ok=True)
    plt = setup_mpl()
    C = Cases()
    checks = json.loads((OUT / "regression_check.json").read_text(encoding="utf-8")) if (OUT / "regression_check.json").exists() else []
    geo = json.loads((OUT / "geometry_check.json").read_text(encoding="utf-8")) if (OUT / "geometry_check.json").exists() else {}
    A = dict(A1=a1(C, plt), A2=a2(C, plt), A3=a3(C, plt))
    runs = run_table(C)
    A["runs"] = runs
    A["regression"] = checks
    A["geometry"] = geo
    (OUT / "analysis.json").write_text(json.dumps(A, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(A, runs, checks, geo)
    print("A1 P1:", [round(r["R_peak"], 3) for r in A["A1"]["P1"]])
    print("A1 P2:", [round(r["R_peak"], 3) for r in A["A1"]["P2"]])
    print("A1 P3:", [round(r["R_peak"], 3) for r in A["A1"]["P3"]])
    print("A1 C :", [round(r["R_peak"], 3) for r in A["A1"]["C"]])
    print("A2   :", {k: (round(v["slope"], 3), round(v["scatter"], 3)) for k, v in A["A2"].items()})
    print("A3   :", [(p["id"], round(p["ratio_pred_over_computed"], 3)) for p in A["A3"]["predictions"]])


if __name__ == "__main__":
    main()
