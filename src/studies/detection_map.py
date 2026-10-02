"""ns-TTR void detection-limit study: depth x half-width sweep, thickness sweep, pulse-width sweep.

Runs the simulator core (ttr_sim) with the same defaults as the GUI, applies the detection criteria as
post-processing (the model is linear, so the pulse energy is scaled afterwards) and writes

    results/detection_map/cases.csv            one row per case
    results/detection_map/summary.md           minimum detectable half-width per depth
    results/detection_map/summary.json         the same numbers for the report builder

Figures and the Word report are made from these files by detection_map_report.py.

Run from the src directory:   .venv\\Scripts\\python.exe studies\\detection_map.py
Nothing in the simulator or the GUI is modified by this script.
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
sys.path.insert(0, str(SRC))

from ttr_sim import BottomStack, DEPTH_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec  # noqa: E402
from ttr_sim.materials import AIR, COPPER, COPPER_C_TR_PROBE, COPPER_REFLECTIVITY_DEFAULT  # noqa: E402
from ttr_sim.presets import D_TH_PRESET  # noqa: E402
from ttr_sim.solver import build_grid  # noqa: E402
from ttr_sim.validation import run_pair  # noqa: E402

OUT = SRC / "results" / "detection_map"
UM = 1e-6

# ----------------------------------------------------------------------------- sweep definition
DEPTHS = [2, 5, 10, 20, 40, 100, 200, 400]                      # void top depth, um
HALF_WIDTHS_MAIN = [2, 5, 10, 20, 30, 40, 80]                   # um; 80 > rod radius = cross-section fully blocked
HALF_WIDTHS_EXTRA = [3, 7, 14, 25, 35]                          # extra points used only to locate the boundary
THICKNESS = 10.0                                                # um, main grid
ROD_RADIUS_UM = 40.0

THICK_SWEEP = dict(depths=[20, 100, 400], half_widths=[20, 40], thicknesses=[1, 10, 40])
PULSE_SWEEP = dict(depths=[20, 100, 400], half_widths=[20, 40], factors=[0.25, 0.5, 1.0, 2.0, 4.0])

# ----------------------------------------------------------------------------- detection criteria (easy to change)
CRITERIA_DEFAULT = dict(sigma=1e-6, c_min=0.02, dT_allowed=10.0)
# sensitivity analysis: one parameter changed at a time
CRITERIA_VARIANTS = {
    "기본 (σ=1e-6, C=2 %, 10 K)": CRITERIA_DEFAULT,
    "σ = 1e-5": dict(CRITERIA_DEFAULT, sigma=1e-5),
    "σ = 1e-7": dict(CRITERIA_DEFAULT, sigma=1e-7),
    "C_min = 1 %": dict(CRITERIA_DEFAULT, c_min=0.01),
    "C_min = 5 %": dict(CRITERIA_DEFAULT, c_min=0.05),
    "ΔT 허용 = 1 K": dict(CRITERIA_DEFAULT, dT_allowed=1.0),
    "ΔT 허용 = 50 K": dict(CRITERIA_DEFAULT, dT_allowed=50.0),
}
N_SIGMA = 3.0

PUMP_W_UM, PROBE_W_UM = 10.0, 5.0


# ----------------------------------------------------------------------------- GUI-equivalent setup
def default_energy_nJ(tau_p: float) -> float:
    """Same as app.default_energy_nJ (pulse energy giving roughly a 1 K peak rise)."""
    e = 2.3 * math.sqrt(tau_p / 17.2e-9)
    mag = 10 ** math.floor(math.log10(e))
    return float(round(e / mag) * mag)


def default_window_us(p, d_um: float) -> float:
    """Same as app.default_window_us: max(5 tau_p, 4 d^2/D), 3 significant digits."""
    t = max(5.0 * p.tau_p, 4.0 * (d_um * UM) ** 2 / COPPER.alpha) * 1e6
    mag = 10 ** math.floor(math.log10(t)) / 100
    return float(round(t / mag) * mag)


def preset_index(depth_um: float) -> int:
    """Depth range containing the depth: lower bound <= depth < upper bound (existing rule: the pulse width
    is the one of the range whose LOWER bound is at or below the void depth)."""
    for i, p in enumerate(DEPTH_PRESETS):
        lo, hi = [float(x) for x in p.depth_range.replace(" μm", "").split("~")]
        if lo <= depth_um < hi or (i == len(DEPTH_PRESETS) - 1 and depth_um <= hi):
            return i
    raise ValueError(f"depth {depth_um} um outside the preset table")


def make_config(depth_um, rh_um, thick_um, tau_factor=1.0, window_factor=1.0) -> tuple[int, SimConfig]:
    idx = preset_index(depth_um)
    p = DEPTH_PRESETS[idx]
    tau_p, dz = p.tau_p, p.dz
    if tau_factor != 1.0:                    # same rule as the GUI's pulse-width field
        tau_p = p.tau_p * tau_factor
        dz = math.sqrt(D_TH_PRESET * tau_p) / 10.0
    window_us = max(default_window_us(p, p.d_rep * 1e6), 5.0 * tau_p * 1e6) * window_factor
    cfg = SimConfig(
        geometry=Geometry(),
        void=VoidSpec(enabled=True, depth=depth_um * UM, thickness=thick_um * UM, r_center=0.0, r_half=rh_um * UM,
                      k=0.026, rho_cp=AIR.rho_cp, shape="ellipse"),
        laser=Laser(tau_p=tau_p, profile="square", energy=default_energy_nJ(p.tau_p) * 1e-9, w=PUMP_W_UM * UM,
                    reflectivity=COPPER_REFLECTIVITY_DEFAULT, probe_w=PROBE_W_UM * UM, c_tr=COPPER_C_TR_PROBE),
        numerics=Numerics(dz=dz, fo=0.5, t_end_mode="absolute", t_end_abs=window_us * 1e-6, dt_growth=1.05,
                          n_snapshots=12),
        bottom=BottomStack(),
    )
    return idx, cfg


def run_case(depth_um, rh_um, thick_um, tau_factor=1.0, group="main") -> dict:
    """One baseline + void pair.  If the signal peak sits at the end of the observation window the window is
    lengthened (x4, up to x64) until the peak is inside it."""
    t0 = time.time()
    window_factor = 1.0
    while True:
        idx, cfg = make_config(depth_um, rh_um, thick_um, tau_factor, window_factor)
        base, void, sig = run_pair(cfg, store_fields=False)
        at_end = sig["t_peak"] >= 0.98 * cfg.t_end
        if not at_end or window_factor >= 64.0:
            break
        window_factor *= 4.0
    T0 = cfg.numerics.T0
    i_pk = int(np.argmax(np.abs(sig["dT"])))
    base_at_peak = float(sig["Tb"][i_pk] - T0)
    rise_b = base.T_probe - T0
    return dict(
        group=group, depth_um=depth_um, half_width_um=rh_um, thickness_um=thick_um,
        blocks_rod=rh_um > ROD_RADIUS_UM, preset=DEPTH_PRESETS[idx].depth_range.replace(" μm", ""),
        tau_factor=tau_factor, tau_p_s=cfg.laser.tau_p, energy_J=cfg.laser.energy,
        window_s=cfg.t_end, window_factor=window_factor, peak_at_window_end=bool(at_end),
        dT_base_max_K=sig["peak_rise_base"], t_base_peak_s=float(base.times[int(np.argmax(rise_b))]),
        dT_base_center_max_K=float(base.T_surface[:, 0].max() - T0),
        void_minus_base_K=sig["peak_dT"], t_peak_s=sig["t_peak"], base_at_peak_K=base_at_peak,
        dRR_diff=COPPER_C_TR_PROBE * sig["peak_dT"], ratio=sig["contrast_at_peak"],
        n_cells=build_grid(cfg).n_cells, n_steps=len(base.times) - 1, wall_s=time.time() - t0,
    )


# ----------------------------------------------------------------------------- post-processing
def judge(row: dict, sigma: float, c_min: float, dT_allowed: float) -> dict:
    """Scale the pulse energy so that the baseline peak rise equals dT_allowed (linear model), then apply
    the noise condition (scaled |dR/R difference| >= 3 sigma) and the ratio condition (|ratio| >= c_min)."""
    s = dT_allowed / row["dT_base_max_K"]
    scaled_dT = s * row["void_minus_base_K"]
    scaled_dRR = abs(COPPER_C_TR_PROBE) * abs(scaled_dT)
    noise_ok = scaled_dRR >= N_SIGMA * sigma
    ratio_ok = abs(row["ratio"]) >= c_min
    if noise_ok and ratio_ok:
        limiting = ""
    elif not noise_ok and not ratio_ok:
        limiting = "잡음+비율"
    else:
        limiting = "잡음" if not noise_ok else "비율"
    return dict(scale=s, scaled_void_minus_base_K=scaled_dT, scaled_dRR_diff=scaled_dRR,
                noise_margin=scaled_dRR / (N_SIGMA * sigma), ratio_margin=abs(row["ratio"]) / c_min,
                noise_ok=bool(noise_ok), ratio_ok=bool(ratio_ok), detectable=bool(noise_ok and ratio_ok), limiting=limiting)


def min_detectable_half_width(rows: list[dict], crit: dict) -> dict:
    """Smallest detectable half-width at one depth (rows sorted by half-width, thickness fixed).

    Both margins (noise, ratio) grow with the half-width; the boundary is where the smaller of the two
    crosses 1.  It is located by log-log interpolation between the neighbouring computed half-widths.
    Half-widths above the rod radius count as 'the cross-section is fully blocked'."""
    rows = sorted(rows, key=lambda r: r["half_width_um"])
    w = np.array([r["half_width_um"] for r in rows], dtype=float)
    j = [judge(r, **crit) for r in rows]
    margin = np.array([min(x["noise_margin"], x["ratio_margin"]) for x in j])
    ok = margin >= 1.0
    if ok.all():
        return dict(min_half_width_um=float(w[0]), bound="<=", limiting="", grid_min=float(w[0]))
    if not ok.any():
        return dict(min_half_width_um=None, bound="none", limiting=j[-1]["limiting"], grid_min=None)
    k = int(np.argmax(ok))                                      # first detectable grid point
    lo, hi = k - 1, k
    x = math.log(w[lo]) + (0.0 - math.log(margin[lo])) / (math.log(margin[hi]) - math.log(margin[lo])) * (
        math.log(w[hi]) - math.log(w[lo]))
    return dict(min_half_width_um=float(math.exp(x)), bound="=", limiting=j[lo]["limiting"], grid_min=float(w[hi]))


FIELDS = ["group", "depth_um", "half_width_um", "thickness_um", "blocks_rod", "preset", "tau_factor", "tau_p_s",
          "energy_J", "window_s", "window_factor", "peak_at_window_end", "dT_base_max_K", "dT_base_center_max_K",
          "t_base_peak_s", "void_minus_base_K", "t_peak_s", "base_at_peak_K", "dRR_diff", "ratio",
          "scale", "scaled_void_minus_base_K", "scaled_dRR_diff", "noise_ok", "ratio_ok", "detectable", "limiting",
          "n_cells", "n_steps", "wall_s"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rows = []

    def add(r):
        r.update({k: v for k, v in judge(r, **CRITERIA_DEFAULT).items() if k in FIELDS})
        rows.append(r)
        print(f"  [{r['group']:9s}] depth {r['depth_um']:>3g}  half-width {r['half_width_um']:>3g}  thick {r['thickness_um']:>3g}"
              f"  tau x{r['tau_factor']:<4g}  void-base {r['void_minus_base_K'] * 1e3:10.4g} mK  ratio {r['ratio'] * 100:8.2f} %"
              f"  {'가능' if r['detectable'] else '불가(' + r['limiting'] + ')':10s}"
              f"{'  window x%g' % r['window_factor'] if r['window_factor'] > 1 else ''}", flush=True)

    print("main grid (depth x half-width, thickness %g um)" % THICKNESS)
    for d in DEPTHS:
        for w in sorted(HALF_WIDTHS_MAIN + HALF_WIDTHS_EXTRA):
            add(run_case(d, w, THICKNESS, group="main" if w in HALF_WIDTHS_MAIN else "boundary"))
    print("thickness sweep")
    for d in THICK_SWEEP["depths"]:
        for w in THICK_SWEEP["half_widths"]:
            for th in THICK_SWEEP["thicknesses"]:
                add(run_case(d, w, th, group="thickness"))
    print("pulse-width sweep")
    for d in PULSE_SWEEP["depths"]:
        for w in PULSE_SWEEP["half_widths"]:
            for f in PULSE_SWEEP["factors"]:
                add(run_case(d, w, THICKNESS, tau_factor=f, group="pulse"))

    with open(OUT / "cases.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in FIELDS})

    # ---- minimum detectable half-width per depth, for every criteria variant
    grid_rows = [r for r in rows if r["group"] in ("main", "boundary")]
    boundary = {}
    for name, crit in CRITERIA_VARIANTS.items():
        boundary[name] = {str(d): min_detectable_half_width([r for r in grid_rows if r["depth_um"] == d], crit) for d in DEPTHS}

    def fmt(b):
        if b["bound"] == "none":
            return "검출 불가"
        if b["min_half_width_um"] > ROD_RADIUS_UM:
            return "단면 전체 차단 필요 (> 40 µm)"
        return ("≤ " if b["bound"] == "<=" else "") + f"{b['min_half_width_um']:.1f} µm"

    lines = ["# 깊이별 검출 가능한 최소 반폭 (두께 10 µm)", "",
             f"기본 기준: σ = {CRITERIA_DEFAULT['sigma']:g}, 3σ, C_min = {CRITERIA_DEFAULT['c_min'] * 100:g} %, "
             f"ΔT_max_allowed = {CRITERIA_DEFAULT['dT_allowed']:g} K", "",
             "| 깊이 (µm) | 펄스폭 | 최소 반폭 (기본 기준) | 판정을 결정한 조건 | " + " | ".join(list(CRITERIA_VARIANTS)[1:]) + " |",
             "|---|---|---|---|" + "---|" * (len(CRITERIA_VARIANTS) - 1)]
    name0 = list(CRITERIA_VARIANTS)[0]
    for d in DEPTHS:
        b = boundary[name0][str(d)]
        tau = next(r["tau_p_s"] for r in grid_rows if r["depth_um"] == d)
        lines.append(f"| {d} | {tau * 1e6:.4g} µs | {fmt(b)} | {b['limiting'] or '—'} | "
                     + " | ".join(fmt(boundary[n][str(d)]) for n in list(CRITERIA_VARIANTS)[1:]) + " |")
    lines += ["", "- 최소 반폭은 계산한 반폭들 사이를 로그 보간한 값입니다. '≤' 는 계산한 가장 작은 반폭(2 µm)에서도 검출 가능하다는 뜻입니다.",
              "- 반폭이 구리 기둥 반경(40 µm)보다 크면 void 가 단면을 완전히 막는 경우입니다."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = dict(
        criteria_default=CRITERIA_DEFAULT, criteria_variants=CRITERIA_VARIANTS, n_sigma=N_SIGMA,
        depths=DEPTHS, half_widths_main=HALF_WIDTHS_MAIN, half_widths_extra=HALF_WIDTHS_EXTRA, thickness=THICKNESS,
        thick_sweep=THICK_SWEEP, pulse_sweep=PULSE_SWEEP, boundary=boundary, rod_radius_um=ROD_RADIUS_UM,
        pump_w_um=PUMP_W_UM, probe_w_um=PROBE_W_UM, reflectivity=COPPER_REFLECTIVITY_DEFAULT, c_tr=COPPER_C_TR_PROBE,
        n_cases=len(rows), wall_s=time.time() - t_start,
        presets=[dict(range=p.depth_range.replace(" μm", ""), tau_p_s=p.tau_p, energy_J=default_energy_nJ(p.tau_p) * 1e-9)
                 for p in DEPTH_PRESETS],
    )
    (OUT / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} cases in {time.time() - t_start:.0f} s -> {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
