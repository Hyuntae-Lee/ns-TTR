"""Figures and Word report for the detection-limit study (reads results/detection_map/cases.csv + summary.json).

Run from the src directory after detection_map.py:
    .venv\\Scripts\\python.exe studies\\detection_map_report.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "studies"))
from detection_map import CRITERIA_DEFAULT, N_SIGMA, judge, min_detectable_half_width  # noqa: E402

OUT = SRC / "results" / "detection_map"
META = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"], "mathtext.fontset": "dejavusans", "axes.unicode_minus": False, "font.size": 15,
    "axes.titlesize": 18, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
    "figure.dpi": 100, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
})
FIGSIZE = (12.0, 6.75)                     # 16:9 -> 2400 x 1350 px at 200 dpi
BLUE, ORANGE, INK, MUTED = "#1f5fa8", "#e07b00", "#222222", "#666666"
BLUE_LIGHT, ORANGE_LIGHT = "#cfe0f3", "#fde3c4"


def load():
    rows = []
    with open(OUT / "cases.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if k in ("group", "preset", "limiting"):
                    continue
                if v in ("True", "False"):
                    r[k] = v == "True"
                else:
                    r[k] = float(v)
            rows.append(r)
    return rows


ROWS = load()
DEPTHS = META["depths"]
WIDTHS = META["half_widths_main"]
GRID = [r for r in ROWS if r["group"] in ("main", "boundary")]
MAIN = {(r["depth_um"], r["half_width_um"]): r for r in ROWS if r["group"] == "main"}


def edges(vals):
    """Cell edges at the geometric means between neighbouring values (for log axes)."""
    v = np.asarray(vals, dtype=float)
    mid = np.sqrt(v[:-1] * v[1:])
    return np.concatenate([[v[0] ** 2 / mid[0]], mid, [v[-1] ** 2 / mid[-1]]])


XE, YE = edges(WIDTHS), edges(DEPTHS)
WIDTH_LABELS = [("%g" % w) if w <= 40 else "80\n(단면 전체)" for w in WIDTHS]


def map_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XE[0], XE[-1])
    ax.set_ylim(YE[-1], YE[0])                      # depth increases downwards
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels(WIDTH_LABELS)
    ax.set_yticks(DEPTHS)
    ax.set_yticklabels(["%g" % d for d in DEPTHS])
    ax.minorticks_off()
    ax.set_xlabel("void 가로 반폭 [µm]  (구리 기둥 반경 40 µm)")
    ax.set_ylabel("void 깊이 [µm]")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)


def heatmap(values, title, cbar_label, cmap, norm, fmt, fname, note=None, cbar_ticks=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    Z = np.array([[values(MAIN[(d, w)]) for w in WIDTHS] for d in DEPTHS])
    pm = ax.pcolormesh(XE, YE, Z, cmap=cmap, norm=norm, edgecolors="white", linewidth=2)
    map_axes(ax)
    for i, d in enumerate(DEPTHS):
        for j, w in enumerate(WIDTHS):
            rgba = pm.cmap(pm.norm(Z[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(w, d, fmt(Z[i, j]), ha="center", va="center", fontsize=14, color="white" if lum < 0.55 else INK,
                    fontweight="bold")
    cb = fig.colorbar(pm, ax=ax, pad=0.02)
    cb.set_label(cbar_label)
    if cbar_ticks:
        cb.set_ticks(list(cbar_ticks))
        cb.set_ticklabels(list(cbar_ticks.values()))
        cb.ax.minorticks_off()
    ax.set_title(title, loc="left", fontweight="bold")
    if note:
        fig.text(0.01, 0.01, note, fontsize=13, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(OUT / fname)
    plt.close(fig)


def fmt_pct(v):
    return ("%.0f" % v) if abs(v) >= 100 else ("%.1f" % v) if abs(v) >= 1 else ("%.2f" % v) if abs(v) >= 0.01 else "<0.01"


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_sci(v):
    e = math.floor(math.log10(v))
    return "%.1f×10%s" % (v / 10 ** e, str(e).translate(SUPERSCRIPT))


# ----------------------------------------------------------------------------- fig 1, 2: heatmaps
heatmap(lambda r: abs(r["ratio"]) * 100, "비율  (void − baseline) / baseline  [%]   · 두께 10 µm",
        "비율 [%]", "Blues", LogNorm(vmin=0.01, vmax=500), fmt_pct, "fig1_ratio_map.png",
        note="셀 안의 숫자는 비율(%). 판정 기준 C_min = 2 % 이상이면 계통 오차와 구별 가능.",
        cbar_ticks={0.01: "0.01", 0.1: "0.1", 1: "1", 10: "10", 100: "100"})
def fmt_ppm(v):
    return ("%.0f" % v) if v >= 100 else ("%.1f" % v) if v >= 10 else ("%.2g" % v)


heatmap(lambda r: r["scaled_dRR_diff"] * 1e6, "스케일된 ΔR/R 차이 [×10⁻⁶]  (baseline 피크를 10 K 로 맞춘 경우)   · 두께 10 µm",
        "|ΔR/R 차이| [×10⁻⁶]", "Oranges", LogNorm(vmin=1e-3, vmax=3e3), fmt_ppm, "fig2_scaled_dRR_map.png",
        note="셀 안의 숫자는 ×10⁻⁶ 단위. 잡음 조건: 3σ = 3 (σ = 1×10⁻⁶) 이상이면 측정 가능. ΔR/R = 1.5×10⁻⁴ /K × (스케일된 온도 차이).",
        cbar_ticks={0.001: "0.001", 0.01: "0.01", 0.1: "0.1", 1: "1", 10: "10", 100: "100", 1000: "1000"})


# ----------------------------------------------------------------------------- boundaries for all criteria variants
def boundary_curve(crit):
    """min detectable half-width per depth; np.nan where even a fully blocking void is not detectable."""
    out = []
    for d in DEPTHS:
        b = min_detectable_half_width([r for r in GRID if r["depth_um"] == d], crit)
        out.append(np.nan if b["min_half_width_um"] is None else b["min_half_width_um"])
    return np.array(out)


VARIANTS = META["criteria_variants"]
CURVES = {name: boundary_curve(crit) for name, crit in VARIANTS.items()}
DEFAULT_NAME = list(VARIANTS)[0]
# variants that give exactly the same curve are drawn once, with a joint label
groups = []
for name, c in CURVES.items():
    for g in groups:
        if np.allclose(g["curve"], c, rtol=1e-6, equal_nan=True):
            g["names"].append(name)
            break
    else:
        groups.append(dict(names=[name], curve=c))
STYLES = [dict(color=INK, ls="-", lw=3.5, marker="o"), dict(color=ORANGE, ls="--", lw=2.5, marker="s"),
          dict(color=BLUE, ls="--", lw=2.5, marker="^"), dict(color="#8c4a00", ls=":", lw=2.8, marker="D"),
          dict(color="#6b9bd1", ls="-.", lw=2.5, marker="v"), dict(color=MUTED, ls=":", lw=2.5, marker="x")]


def group_label(g):
    names = [n.replace("기본 (σ=1e-6, C=2 %, 10 K)", "기본 기준") for n in g["names"]]
    return " · ".join(names)


# ----------------------------------------------------------------------------- fig 3: detectability map
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, d in enumerate(DEPTHS):
    for j, w in enumerate(WIDTHS):
        r = MAIN[(d, w)]
        ok = r["detectable"]
        ax.fill_between([XE[j], XE[j + 1]], YE[i], YE[i + 1], facecolor=BLUE_LIGHT if ok else ORANGE_LIGHT,
                        edgecolor="white" if ok else ORANGE, hatch=None if ok else "//", linewidth=0)
        label = "가능" if ok else "불가\n" + r["limiting"].replace("+", "·")
        ax.text(w, d, label, ha="center", va="center", fontsize=14, linespacing=1.05,
                color=BLUE if ok else "#7a3f00", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3, foreground=BLUE_LIGHT if ok else ORANGE_LIGHT)])
for x in XE:
    ax.plot([x, x], [YE[0], YE[-1]], color="white", lw=2)
for y in YE:
    ax.plot([XE[0], XE[-1]], [y, y], color="white", lw=2)
map_axes(ax)
handles = [Patch(facecolor=BLUE_LIGHT, edgecolor=BLUE, label="검출 가능 (기본 기준)"),
           Patch(facecolor=ORANGE_LIGHT, edgecolor=ORANGE, hatch="//", label="검출 불가 (아래 줄: 걸린 조건)")]
for g, st in zip(groups, STYLES):
    c = np.clip(np.nan_to_num(g["curve"], nan=XE[-1]), XE[0], XE[-1])
    lw = 3.5 if st["ls"] == "-" else 2.5
    ax.plot(c, DEPTHS, color=st["color"], ls=st["ls"], lw=lw, zorder=5,
            path_effects=[pe.withStroke(linewidth=lw + 2.5, foreground="white")])
    handles.append(Line2D([0], [0], color=st["color"], ls=st["ls"], lw=lw, label="경계: " + group_label(g)))
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=14)
ax.set_title("검출 가능 영역  · 두께 10 µm · 각 경계선의 오른쪽이 검출 가능", loc="left", fontweight="bold")
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(OUT / "fig3_detectability_map.png")
plt.close(fig)

# ----------------------------------------------------------------------------- fig 4: minimum detectable half-width
fig, ax = plt.subplots(figsize=FIGSIZE)
for g, st in zip(groups, STYLES):
    ax.plot(DEPTHS, g["curve"], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=9, label=group_label(g))
ax.axhline(40, color=MUTED, lw=1.5)
ax.text(DEPTHS[0], 40.8, "구리 기둥 반경 40 µm (void 가 단면 전체를 막는 크기)", color=MUTED, fontsize=14, va="bottom")
c0 = CURVES[DEFAULT_NAME]
for d, v in zip(DEPTHS, c0):
    ax.annotate("%.1f" % v if v > 2.0 else "≤2", (d, v), textcoords="offset points", xytext=(-4, 12), ha="right",
                fontsize=15, color=INK, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")])
ax.set_xscale("log")
ax.set_xticks(DEPTHS)
ax.set_xticklabels(["%g" % d for d in DEPTHS])
ax.minorticks_off()
ax.set_ylim(0, 45)
ax.set_xlabel("void 깊이 [µm]")
ax.set_ylabel("검출 가능한 최소 반폭 [µm]")
ax.grid(axis="y", color="#dddddd", lw=1)
ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="판정 기준", title_fontsize=14)
ax.set_title("깊이별 검출 가능한 최소 반폭  · 두께 10 µm · 곡선 위쪽이 검출 가능", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig4_min_half_width.png")
plt.close(fig)

# ----------------------------------------------------------------------------- fig 5: thickness
TH = [r for r in ROWS if r["group"] == "thickness"]
DEPTH_COLORS = {20: "#9cc3ea", 100: "#3d7fc4", 400: "#123f73"}
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True)
for ax, key, ylabel, thr, thr_label in (
        (axes[0], lambda r: abs(r["ratio"]) * 100, "비율 [%]", CRITERIA_DEFAULT["c_min"] * 100, "C_min = 2 %"),
        (axes[1], lambda r: r["scaled_dRR_diff"] * 1e6, "스케일된 |ΔR/R 차이| [×10⁻⁶]  (10 K)",
         N_SIGMA * CRITERIA_DEFAULT["sigma"] * 1e6, "3σ = 3 ×10⁻⁶")):
    for d in META["thick_sweep"]["depths"]:
        for w in META["thick_sweep"]["half_widths"]:
            pts = sorted([r for r in TH if r["depth_um"] == d and r["half_width_um"] == w], key=lambda r: r["thickness_um"])
            ax.plot([p["thickness_um"] for p in pts], [key(p) for p in pts], color=DEPTH_COLORS[d],
                    ls="-" if w == 40 else "--", lw=3, marker="o" if w == 40 else "s", ms=9)
    ax.axhline(thr, color=ORANGE, lw=2, ls=":")
    ax.text(40, thr / 1.12, thr_label, color="#8c4a00", fontsize=14, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, thr / 2.2), hi)
    ax.set_xticks(META["thick_sweep"]["thicknesses"])
    ax.set_xticklabels(["%g" % t for t in META["thick_sweep"]["thicknesses"]])
    ax.minorticks_off()
    ax.set_xlabel("void 세로 두께 [µm]")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", lw=1)
    ax.set_axisbelow(True)
handles = [Line2D([0], [0], color=DEPTH_COLORS[d], lw=3, label="깊이 %g µm" % d) for d in META["thick_sweep"]["depths"]]
handles += [Line2D([0], [0], color=INK, lw=3, ls="-", marker="o", ms=9, label="반폭 40 µm"),
            Line2D([0], [0], color=INK, lw=3, ls="--", marker="s", ms=9, label="반폭 20 µm")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=14)
fig.suptitle("세로 두께의 영향  · 두께를 40배 바꿔도 신호 변화는 3배 이내", x=0.01, ha="left", fontweight="bold", fontsize=18)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(OUT / "fig5_thickness.png")
plt.close(fig)

# ----------------------------------------------------------------------------- fig 6: pulse width
PU = [r for r in ROWS if r["group"] == "pulse"]
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True)
for ax, key, ylabel in ((axes[0], lambda r: r["scaled_dRR_diff"], "스케일된 |ΔR/R 차이| ÷ 기존 규칙(1×) 값"),
                        (axes[1], lambda r: abs(r["ratio"]), "비율 ÷ 기존 규칙(1×) 값")):
    for d in META["pulse_sweep"]["depths"]:
        for w in META["pulse_sweep"]["half_widths"]:
            pts = sorted([r for r in PU if r["depth_um"] == d and r["half_width_um"] == w], key=lambda r: r["tau_factor"])
            ref = key(next(p for p in pts if p["tau_factor"] == 1.0))
            ax.plot([p["tau_factor"] for p in pts], [key(p) / ref for p in pts], color=DEPTH_COLORS[d],
                    ls="-" if w == 40 else "--", lw=3, marker="o" if w == 40 else "s", ms=9)
    ax.axhline(1.0, color=MUTED, lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(META["pulse_sweep"]["factors"])
    ax.set_xticklabels(["%g×" % f for f in META["pulse_sweep"]["factors"]])
    ax.minorticks_off()
    ax.set_xlabel("펄스폭 (기존 규칙값 대비)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", lw=1)
    ax.set_axisbelow(True)
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=14)
fig.suptitle("펄스폭의 영향  · 표면 온도 상승을 10 K 로 맞춘 조건에서 비교", x=0.01, ha="left", fontweight="bold", fontsize=18)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(OUT / "fig6_pulse_width.png")
plt.close(fig)

print("figures written to", OUT)
for g in groups:
    print(" ", group_label(g), np.round(g["curve"], 1))

# ----------------------------------------------------------------------------- fig 7: void displaced from the axis
OFF_CSV = OUT / "offaxis_cases.csv"
if OFF_CSV.exists():
    OFF = []
    with open(OFF_CSV, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if k in ("set", "beam", "limiting"):
                    continue
                r[k] = (v == "True") if v in ("True", "False") else float(v)
            OFF.append(r)
    SERIES = [("depth5", "깊이 5 µm · 반폭 10 · 두께 2.5 µm", BLUE, "-", "o"),
              ("depth20", "깊이 20 µm · 반폭 10 · 두께 10 µm", "#7fb0e0", "--", "s")]
    WIDE = [("flat40_probe5", "깊이 5 µm void, 넓은 펌프(flat-top 40 µm) · 프로브 5 µm", ORANGE, "^"),
            ("flat40_probe40", "깊이 5 µm void, 넓은 펌프(flat-top 40 µm) · 프로브 40 µm", "#8c4a00", "D")]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True)
    for ax, key, ylabel, thr, thr_label in (
            (axes[0], lambda r: abs(r["ratio"]) * 100, "비율 [%]", CRITERIA_DEFAULT["c_min"] * 100, "C_min = 2 %"),
            (axes[1], lambda r: r["scaled_dRR_diff"] * 1e6, "스케일된 |ΔR/R 차이| [×10⁻⁶]  (10 K)",
             N_SIGMA * CRITERIA_DEFAULT["sigma"] * 1e6, "3σ = 3 ×10⁻⁶")):
        for name, label, color, ls, marker in SERIES:
            pts = sorted([r for r in OFF if r["set"] == name], key=lambda r: r["offset_um"])
            ax.plot([p["offset_um"] for p in pts], [key(p) for p in pts], color=color, ls=ls, lw=3, marker=marker, ms=10)
        for name, label, color, marker in WIDE:
            pts = [r for r in OFF if r["set"] == name and r["offset_um"] > 0]
            ax.plot([p["offset_um"] for p in pts], [key(p) for p in pts], color=color, ls="none", marker=marker, ms=13,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.axhline(thr, color=ORANGE, lw=2, ls=":")
        ax.text(-0.5, thr / 1.12, thr_label, color="#8c4a00", fontsize=14, va="top", ha="left")
        ax.axvline(10, color=MUTED, lw=1.2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, thr / 2.5), hi)
        ax.text(10.4, ax.get_ylim()[1], "펌프 1/e² 반경", color=MUTED, fontsize=13, va="top")
        ax.set_xlim(-1, 31)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xlabel("void 중심이 기둥 축에서 벗어난 거리 [µm]")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", lw=1)
        ax.set_axisbelow(True)
    handles = [Line2D([0], [0], color=c, ls=ls, lw=3, marker=m, ms=10, label=lab) for _, lab, c, ls, m in SERIES]
    handles += [Line2D([0], [0], color=c, ls="none", marker=m, ms=13, markeredgecolor="white", label=lab) for _, lab, c, m in WIDE]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=14)
    fig.suptitle("축에서 벗어난 void  · 펌프·프로브는 축 위에 고정 (반경 10 / 5 µm)", x=0.01, ha="left", fontweight="bold", fontsize=18)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(OUT / "fig7_offaxis.png")
    plt.close(fig)
    print("fig7 written")
