"""Multi-void study: do several individually undetectable voids add up to a detectable signal, and is it the
void volume or its projected area that matters?

Fixed conditions (all cases): void half-width 10 um, thickness 10 um (series E excepted), air (k = 0.026),
pulse preset of the 70 um depth (40~100 um: tau_p = 27.6 us, dz = 5.7 um), pump 10 um / probe 5 um on the
axis, GUI default pulse energy, observation window 1000 us for every case (covers 2.9 d^2/D of the deepest
void at 210 um) so that all curves share one time axis.

Series (void positions are explicit, not random, except F):
  A  single void: 70 um on the axis, 70 um at r = 25 um, and 90..210 um (15 um steps) at r = 25 um
     -> per-void contribution versus depth, and the building blocks of the superposition check for B
  B  70 um void on the axis + N deeper voids (N = 1, 3, 6, 9) at 90 + 15 k um, r = 25 um, theta = 40 k deg
     (nested sets; none of them in the projected shadow of the 70 um void)
  C  70 um void + 3 voids stacked directly below it (90/105/120 um, on the axis): shadowing, compare with B_N3
  D  voids all at 70 um: axis + 1, 2, 5 on the r = 25 um ring (2, 3, 6 voids): blocked cross-section fraction
  E  single void at 70 um, same volume with other shapes and same area with other thicknesses: volume vs area
  F  10 voids spread at random with top faces >= 70 um (seeds 1..5): realistic arrangements of voids at the
     undetectable depth and deeper;  F0: the same over the whole rod (GUI random mode), reference only

Outputs (results/multi_void):
  cases/<id>.npz, cases/<id>.json   curves, snapshot planes and metrics of each case (cache: finished cases
                                    are not recomputed, so an interrupted run can simply be restarted)
  figs/<id>.png                     per-case figure: MAP with the void positions, plan view, the two signal
                                    charts of the GUI's 신호 page, the headline metrics table, void info
  run.log                           one line per finished case
Run from the src directory:   .venv\\Scripts\\python.exe studies\\multi_void.py  [--workers 4] [--only A,B]
Summary figures, cases.csv and the Word report are made by multi_void_report.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "studies"))

from detection_map import CRITERIA_DEFAULT, N_SIGMA, UM, judge, make_config  # noqa: E402
from ttr_sim import Geometry, VoidSpec, random_voids  # noqa: E402
from ttr_sim.materials import AIR, COPPER, COPPER_C_TR_PROBE  # noqa: E402
from ttr_sim.solver3d import count_unknowns, fit_resolution, full_circle, needs_3d  # noqa: E402
from ttr_sim.validation import analytic_comparison, run_pair  # noqa: E402

OUT = SRC / "results" / "multi_void"
CASES_DIR, FIG_DIR = OUT / "cases", OUT / "figs"
LOG = OUT / "run.log"

WINDOW_US = 1000.0
RH, TH = 10.0, 10.0                       # um
BASE_DEPTH = 70.0                         # um, top face
RING_R = 25.0                             # um, centre radius of the off-axis voids (edge at 35 um < R_cu = 40 um)
DEEP_DEPTHS = [90.0 + 15.0 * k for k in range(9)]      # 90 .. 210 um
DEEP_THETAS = [40.0 * k for k in range(9)]             # deg
ROD_R_UM, PUMP_W_UM, PROBE_W_UM = 40.0, 10.0, 5.0
K_VOID = 0.026


def V(depth, rc=0.0, theta=0.0, rh=RH, thick=TH) -> dict:
    return dict(depth=float(depth), rh=float(rh), thick=float(thick), rc=float(rc), theta=float(theta))


def random_specs(seed: int, z_min_um: float = 0.0) -> list[dict]:
    """10 voids spread uniformly over the rod (GUI random mode).  `z_min_um` > 0 restricts the top faces to
    depths >= z_min_um (the generator is run on a rod shortened by z_min_um and the depths are shifted)."""
    g = Geometry()
    vs = random_voids(10, RH * UM, TH * UM, replace(g, L=g.L - z_min_um * UM), seed=seed)
    return [V(v.depth / UM + z_min_um, v.r_center / UM, math.degrees(v.theta)) for v in vs]


def all_cases() -> list[dict]:
    base = V(BASE_DEPTH)
    C = [dict(id="A_d070_axis", series="A", label="단독 70 μm, 축 위", voids=[base]),
         dict(id="A_d070_r25", series="A", label="단독 70 μm, r = 25 μm", voids=[V(BASE_DEPTH, RING_R)])]
    for d in DEEP_DEPTHS:
        C.append(dict(id=f"A_d{int(d):03d}_r25", series="A", label=f"단독 {d:g} μm, r = 25 μm", voids=[V(d, RING_R)]))
    for n in (1, 3, 6, 9):
        C.append(dict(id=f"B_N{n}", series="B", label=f"70 μm + 더 깊은 void {n}개 (90~{DEEP_DEPTHS[n - 1]:g} μm)",
                      voids=[base] + [V(DEEP_DEPTHS[k], RING_R, DEEP_THETAS[k]) for k in range(n)]))
    C.append(dict(id="C_stack3", series="C", label="70 μm + 바로 아래 적층 3개 (90/105/120 μm, 축 위)",
                  voids=[base] + [V(DEEP_DEPTHS[k]) for k in range(3)]))
    for n, ths in ((1, [0.0]), (2, [0.0, 180.0]), (5, [0.0, 72.0, 144.0, 216.0, 288.0])):
        C.append(dict(id=f"D_N{n + 1}", series="D", label=f"70 μm 에 {n + 1}개 (축 + r = 25 μm 고리 {n}개)",
                      voids=[base] + [V(BASE_DEPTH, RING_R, t) for t in ths]))
    for rh, th in ((14.1, 5.0), (7.1, 20.0), (10.0, 2.0), (10.0, 5.0), (10.0, 20.0)):
        C.append(dict(id=f"E_rh{rh:g}_th{th:g}", series="E", label=f"단독 70 μm, 반폭 {rh:g} μm, 두께 {th:g} μm",
                      voids=[V(BASE_DEPTH, rh=rh, thick=th)]))
    for seed in range(1, 6):      # whole rod (GUI random mode): may put a void at the surface, kept as a reference set
        C.append(dict(id=f"F0_seed{seed}", series="F0", label=f"무작위 10개, 로드 전체 (시드 {seed})", voids=random_specs(seed), seed=seed))
    for seed in range(1, 6):      # the study's question: voids at the undetectable depth and deeper
        C.append(dict(id=f"F_seed{seed}", series="F", label=f"무작위 10개, 윗면 깊이 ≥ 70 μm (시드 {seed})",
                      voids=random_specs(seed, BASE_DEPTH), seed=seed))
    return C


def build_cfg(case: dict):
    _, cfg = make_config(BASE_DEPTH, RH, TH)          # 40~100 um preset, GUI default energy / beam / bottom stack
    vs = [VoidSpec(enabled=True, depth=v["depth"] * UM, thickness=v["thick"] * UM, r_center=v["rc"] * UM,
                   theta=math.radians(v["theta"]), r_half=v["rh"] * UM, k=K_VOID, rho_cp=AIR.rho_cp, shape="ellipse")
          for v in case["voids"]]
    cfg = replace(cfg, void=vs[0], extra_voids=tuple(vs[1:]),
                  numerics=replace(cfg.numerics, t_end_abs=WINDOW_US * 1e-6, n_snapshots=24))
    return fit_resolution(cfg)


def cost(case: dict) -> float:
    """Rough relative cost, for scheduling the long cases first."""
    cfg = build_cfg(case)
    if not needs_3d(cfg):
        return 1.0
    return count_unknowns(cfg)[2] / 1e5


# ----------------------------------------------------------------------------- one case
def plane_frames(res, cfg):
    """(x_um, z_um, frames (n, nz, nx), times) of the copper cross-section |x| <= R_cu, 0 <= z <= L (GUI MAP)."""
    T0 = cfg.numerics.T0
    grid = res.grid
    ir = int(np.argmin(np.abs(grid.r_faces - cfg.geometry.R_cu)))
    nz = int(np.searchsorted(grid.z_c, cfg.geometry.L))
    x_um = np.concatenate([-grid.r_c[:ir][::-1], grid.r_c[:ir]]) * 1e6
    z_um = grid.z_c[:nz] * 1e6
    frames, times = [], []
    for t, T in res.snapshots:
        T = T[:nz] - T0
        if res.is_3d:
            f = T[:, grid.nr - ir: grid.nr + ir]
        else:
            f = np.concatenate([T[:, :ir][:, ::-1], T[:, :ir]], axis=1)
        frames.append(f.astype(np.float32))
        times.append(t)
    return x_um, z_um, np.stack(frames), np.array(times)


def run_case(case: dict) -> dict:
    cid = case["id"]
    jpath = CASES_DIR / f"{cid}.json"
    if jpath.exists():
        return json.loads(jpath.read_text(encoding="utf-8"))
    t0 = time.time()
    cfg = build_cfg(case)
    base, void, sig = run_pair(cfg, store_fields=True)
    ana = analytic_comparison(base)
    T0 = cfg.numerics.T0
    t = void.times
    dT_b = np.interp(t, base.times, base.dT_probe)
    dT_v = void.dT_probe
    contrast = sig["contrast"]
    i_c = int(np.argmax(np.abs(contrast)))
    depth_min = min(v["depth"] for v in case["voids"])
    depth_max = max(v["depth"] for v in case["voids"])
    row = dict(
        id=cid, series=case["series"], label=case["label"], n_voids=len(case["voids"]),
        depth_min_um=depth_min, depth_max_um=depth_max,
        blocked_area_frac=float(sum(math.pi * v["rh"] ** 2 for v in case["voids"]) / (math.pi * ROD_R_UM ** 2)),
        void_volume_um3=float(sum(4.0 / 3.0 * math.pi * v["rh"] ** 2 * (v["thick"] / 2.0) for v in case["voids"])),
        tau_p_s=cfg.laser.tau_p, energy_J=cfg.laser.energy, window_s=cfg.t_end,
        dT_base_max_K=sig["peak_rise_base"], t_base_peak_s=float(base.times[int(np.argmax(base.dT_probe))]),
        void_minus_base_K=sig["peak_dT"], t_peak_s=sig["t_peak"], ratio=sig["contrast_at_peak"],
        base_at_peak_K=float(np.interp(sig["t_peak"], t, dT_b)),
        dRR_diff=COPPER_C_TR_PROBE * sig["peak_dT"],
        contrast_max=float(contrast[i_c]), t_contrast_max_s=float(t[i_c]),
        dT_at_contrast_max_K=float(sig["dT"][i_c]),
        peak_at_window_end=bool(sig["t_peak"] >= 0.98 * cfg.t_end),
        is_3d=bool(void.is_3d), full_circle=bool(full_circle(cfg)) if needs_3d(cfg) else False,
        n_unknowns=int(void.diagnostics.get("n_cells", void.n_cells)), n_theta=int(void.n_theta),
        void_cells=int(cfg.numerics.void_cells), n_steps=int(len(t) - 1),
        solver=str(void.diagnostics.get("solver", "")), mean_iterations=float(void.diagnostics.get("mean_iterations", 0.0)),
        energy_error=float(void.energy_error), wall_s=time.time() - t0,
    )
    row.update({k: v for k, v in judge(row, **CRITERIA_DEFAULT).items()})
    x_um, z_um, frames, ft = plane_frames(void, cfg)
    np.savez_compressed(CASES_DIR / f"{cid}.npz", times=t, dT_base=dT_b, dT_void=dT_v, dT_diff=sig["dT"],
                        contrast=contrast, pulse=void.pulse,
                        analytic=np.asarray(ana["analytic"]) if ana.get("available", True) else np.zeros(0),
                        base_times=base.times, x_um=x_um, z_um=z_um, frames=frames, frame_times=ft)
    row["voids"] = case["voids"]
    jpath.write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        make_case_figure(case, row)
    except Exception as e:  # noqa: BLE001  (a plotting problem must not lose the computed case)
        print(f"  [figure failed for {cid}: {e!r}]", flush=True)
    line = (f"{time.strftime('%H:%M:%S')}  {cid:14s} voids {row['n_voids']:2d}  {'3D' if row['is_3d'] else '2D'} "
            f"N={row['n_unknowns']:>9,}  peak {row['void_minus_base_K'] * 1e3:9.4g} mK at {row['t_peak_s'] * 1e6:6.1f} us  "
            f"ratio {row['ratio'] * 100:7.3f} %  contrast max {row['contrast_max'] * 100:7.3f} %  "
            f"{'가능' if row['detectable'] else '불가(' + row['limiting'] + ')'}  ({row['wall_s'] / 60:.1f} min)")
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return row


# ----------------------------------------------------------------------------- per-case figure
TEMP_COLORS = [(0.00, "#141466"), (0.14, "#0000ff"), (0.32, "#00c8ff"), (0.50, "#00e000"),
               (0.68, "#ffff00"), (0.85, "#ff8c00"), (1.00, "#ff0000")]


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]      # DejaVu as fallback for glyphs Malgun lacks (minus signs)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    return plt


def draw_voids_xz(ax, voids, rod_r=ROD_R_UM):
    """Void outlines on the (x, z) plane: x = r cos(theta); voids far from the plane (|y| large) are faded/dashed."""
    from matplotlib.patches import Ellipse
    for i, v in enumerate(voids):
        x = v["rc"] * math.cos(math.radians(v["theta"]))
        y = v["rc"] * math.sin(math.radians(v["theta"]))
        off = abs(y) / rod_r
        cut = abs(y) < v["rh"]                                  # the display plane passes through the void
        ax.add_patch(Ellipse((x, v["depth"] + v["thick"] / 2), 2 * v["rh"], v["thick"], fill=False,
                             edgecolor="white" if cut else "#ff80ff", lw=1.6 if cut else 1.1,
                             ls="-" if cut else "--", alpha=max(0.35, 1.0 - 0.8 * off)))
        ax.annotate(str(i + 1), (x + v["rh"], v["depth"]), color="white", fontsize=8, ha="left", va="bottom")


def make_case_figure(case: dict, row: dict):
    plt = setup_mpl()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    import matplotlib.gridspec as gridspec
    d = np.load(CASES_DIR / f"{case['id']}.npz")
    t_us = d["times"] * 1e6
    tau_us = row["tau_p_s"] * 1e6
    c_tr = COPPER_C_TR_PROBE
    voids = case["voids"]

    fig = plt.figure(figsize=(16, 14.5))
    gs = gridspec.GridSpec(3, 2, height_ratios=[5.2, 4.0, 2.6], width_ratios=[1.35, 1.0], hspace=0.32, wspace=0.18)
    fig.suptitle(f"{case['id']}  —  {case['label']}", fontsize=15, fontweight="bold", y=0.995)

    # ---- MAP: temperature field at the frame nearest the void-signal peak
    ax = fig.add_subplot(gs[0, 0])
    k = int(np.argmin(np.abs(d["frame_times"] - row["t_peak_s"])))
    F = d["frames"][k]
    cmap = LinearSegmentedColormap.from_list("ttr", TEMP_COLORS)
    pm = ax.pcolormesh(d["x_um"], d["z_um"], F, cmap=cmap, vmin=0.0, vmax=max(float(F.max()), 1e-12), shading="nearest")
    ax.set_xlim(-ROD_R_UM, ROD_R_UM)
    ax.set_ylim(min(500.0, max(row["depth_max_um"] + 80.0, 220.0)), 0.0)
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("z [μm]  (깊이)")
    ax.set_title(f"MAP: ΔT(x, z) at t = {d['frame_times'][k] * 1e6:.0f} μs  (void 신호 피크 {row['t_peak_s'] * 1e6:.0f} μs 근방 프레임)", fontsize=11)
    fig.colorbar(pm, ax=ax, pad=0.02, label="ΔT [K]")
    draw_voids_xz(ax, voids)
    ax.axvspan(-PUMP_W_UM, PUMP_W_UM, ymin=0.985, ymax=1.0, color="orange", alpha=0.9)
    ax.text(0, 9, "펌프 스팟 (w = 10 μm)", color="white", ha="center", va="top", fontsize=8)

    # ---- plan view (x, y): all voids with their depths
    ax = fig.add_subplot(gs[0, 1])
    depths = [v["depth"] for v in voids]
    norm = plt.Normalize(min(60.0, min(depths)), max(220.0, max(depths)))
    cm2 = plt.get_cmap("viridis_r")
    ax.add_patch(Circle((0, 0), ROD_R_UM, fill=False, color="k", lw=1.5))
    ax.add_patch(Circle((0, 0), PUMP_W_UM, fill=False, color="darkorange", ls="--", lw=1.2))
    ax.add_patch(Circle((0, 0), PROBE_W_UM, fill=False, color="red", ls=":", lw=1.2))
    for i, v in enumerate(voids):
        x = v["rc"] * math.cos(math.radians(v["theta"]))
        y = v["rc"] * math.sin(math.radians(v["theta"]))
        ax.add_patch(Circle((x, y), v["rh"], color=cm2(norm(v["depth"])), alpha=0.75, ec="k", lw=0.6))
        ax.annotate(f"{i + 1}\n{v['depth']:g}", (x, y), ha="center", va="center", fontsize=7, color="w" if norm(v["depth"]) > 0.5 else "k")
    ax.set_aspect("equal")
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title("평면도: void 위치 (숫자 = 번호 / 윗면 깊이 μm)  ─ 주황 점선 펌프, 빨강 점선 프로브", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cm2, norm=norm)
    fig.colorbar(sm, ax=ax, pad=0.02, label="윗면 깊이 [μm]")

    # ---- signal charts (the two main charts of the GUI's 신호 page)
    ax = fig.add_subplot(gs[1, 0])
    rr_b, rr_v, rr_d = c_tr * d["dT_base"] * 1e4, c_tr * d["dT_void"] * 1e4, c_tr * d["dT_diff"] * 1e4
    ax.plot(t_us, rr_b, color="#1f77b4", label="baseline (void 없음)")
    ax.plot(t_us, rr_v, color="#d62728", label=f"void (k_void = {K_VOID:g})")
    ax.fill_between(t_us, rr_b, rr_v, color="#d62728", alpha=0.12)
    ax.plot(t_us, rr_d, color="#2ca02c", ls=":", label="void - baseline (같은 축)")
    ax.axhline(0, color="gray", lw=1)
    ax.axvspan(0, tau_us, color="orange", alpha=0.12)
    ax.text(tau_us, ax.get_ylim()[0], " 펄스", color="darkorange", fontsize=8, va="bottom")
    ax.set_xlim(0, WINDOW_US)
    ax.set_xlabel("t [μs]")
    ax.set_ylabel("ΔR/R [×10$^{-4}$]")
    ax.set_title(f"열반사 신호 ΔR/R  (프로브 632.8 nm, (dR/dT)/R = {c_tr:.2e} /K)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_us, d["dT_base"], color="#1f77b4", label="baseline (void 없음)")
    ax.plot(t_us, d["dT_void"], color="#d62728", label=f"void (k_void = {K_VOID:g})")
    if d["analytic"].size:
        ax.plot(d["base_times"] * 1e6, d["analytic"], color="gray", ls="--", label="해석해 (semi-infinite Cu)")
    ax.axvspan(0, tau_us, color="orange", alpha=0.12)
    ax.set_xlim(0, WINDOW_US)
    ax.set_xlabel("t [μs]")
    ax.set_ylabel("ΔT [K]")
    ax.set_title("프로브 가중 표면 온도 상승", fontsize=11)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    pk = float(np.max(d["pulse"])) or 1.0
    ax2.plot(t_us, d["pulse"] / pk, color="orange", ls=":", lw=1.2, label="펄스 파형")
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_ylabel("펄스 파형 (피크 = 1)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    # ---- headline metrics table + void info (as under the GUI charts)
    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    rr = lambda dT: f"{c_tr * dT * 1e4:+.3g} ×10$^{-4}$"      # noqa: E731
    cells = [
        ["baseline 피크", f"{row['dT_base_max_K']:+.3g} K", rr(row["dT_base_max_K"]), "—", f"{row['t_base_peak_s'] * 1e6:.3g} μs"],
        ["void 신호 피크 (void - baseline)", f"{row['void_minus_base_K']:+.3g} K", rr(row["void_minus_base_K"]),
         f"{row['ratio'] * 100:+.2f} %", f"{row['t_peak_s'] * 1e6:.3g} μs"],
        ["상대 대비 최대", f"{row['dT_at_contrast_max_K']:+.3g} K", rr(row["dT_at_contrast_max_K"]),
         f"{row['contrast_max'] * 100:+.2f} %", f"{row['t_contrast_max_s'] * 1e6:.3g} μs"],
    ]
    tb = ax.table(cellText=cells, colLabels=["", "ΔT", "ΔR/R", "피크 시점 대비", "시각"], loc="upper center",
                  cellLoc="center", colLoc="center", colWidths=[0.34, 0.15, 0.17, 0.17, 0.15])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    tb.scale(1.0, 1.45)
    verdict = "검출 가능" if row["detectable"] else f"검출 불가 (한계 조건: {row['limiting']})"
    ax.text(0.0, 0.22,
            f"판정 (baseline 피크를 {CRITERIA_DEFAULT['dT_allowed']:g} K 로 스케일, σ = {CRITERIA_DEFAULT['sigma']:g}, "
            f"C_min = {CRITERIA_DEFAULT['c_min'] * 100:g} %):  {verdict}\n"
            f"  스케일된 |ΔR/R| = {row['scaled_dRR_diff']:.3g} (기준 {N_SIGMA * CRITERIA_DEFAULT['sigma']:.1g}, 여유 ×{row['noise_margin']:.2f}),  "
            f"|대비| = {abs(row['ratio']) * 100:.2f} % (기준 {CRITERIA_DEFAULT['c_min'] * 100:g} %, 여유 ×{row['ratio_margin']:.2f})\n"
            f"  2·d²/D (가장 얕은 void {row['depth_min_um']:g} μm) = {2 * (row['depth_min_um'] * UM) ** 2 / COPPER.alpha * 1e6:.3g} μs · "
            f"τp = {tau_us:.3g} μs · 관측창 {WINDOW_US:g} μs",
            transform=ax.transAxes, fontsize=9, va="top", family="Malgun Gothic")

    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    lines = [f"k_void = {K_VOID:g} W/m·K (공기)   ·   void {len(voids)}개   ·   막힌 단면적 비율 {row['blocked_area_frac'] * 100:.1f} %   ·   "
             f"void 총부피 {row['void_volume_um3']:,.0f} μm³",
             f"계산: {'3D (r, θ, z) ' + ('전체 원주' if row['full_circle'] else '반원') + f', nθ = {row['n_theta']}' if row['is_3d'] else '2D 축대칭'}, "
             f"미지수 {row['n_unknowns']:,}, 스텝 {row['n_steps']}, void 해상도 1/{row['void_cells']}, 에너지 오차 {row['energy_error']:.1e}, "
             f"{row['wall_s'] / 60:.1f} min",
             "번호: 윗면 깊이 / 축 거리 r / 방위각 θ / 반폭 / 두께 [μm, °]"]
    for i, v in enumerate(voids[:12]):
        lines.append(f"  {i + 1:2d}: {v['depth']:6.1f} / {v['rc']:5.1f} / {v['theta']:6.1f} / {v['rh']:4.1f} / {v['thick']:4.1f}")
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, fontsize=8.6, va="top", family="Malgun Gothic")

    fig.savefig(FIG_DIR / f"{case['id']}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=str, default="", help="comma-separated series letters or case ids")
    ap.add_argument("--refigure", action="store_true", help="redraw the per-case figures from the cache only")
    args = ap.parse_args()
    for p in (OUT, CASES_DIR, FIG_DIR):
        p.mkdir(parents=True, exist_ok=True)
    cases = all_cases()
    if args.only:
        keys = set(args.only.split(","))
        cases = [c for c in cases if c["series"] in keys or c["id"] in keys]
    if args.refigure:
        for c in cases:
            jp = CASES_DIR / f"{c['id']}.json"
            if jp.exists():
                make_case_figure(c, json.loads(jp.read_text(encoding="utf-8")))
                print("figure", c["id"], flush=True)
        return
    todo = [c for c in cases if not (CASES_DIR / f"{c['id']}.json").exists()]
    print(f"{len(cases)} cases, {len(todo)} to run, {args.workers} workers", flush=True)
    todo.sort(key=cost, reverse=True)
    for c in todo:
        print(f"  {c['id']:14s} {c['label']}", flush=True)
    t0 = time.time()
    if args.workers <= 1:
        for c in todo:
            run_case(c)
    else:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for _ in pool.imap_unordered(run_case, todo):
                pass
    print(f"all done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
