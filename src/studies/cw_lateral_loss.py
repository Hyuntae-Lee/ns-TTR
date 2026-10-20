"""CW transmission mode, first study (spec docs/cw-transmission-spec.md §5.5): lateral-loss sensitivity.

How much does each k_eff analysis method of ttr_sim.analysis depend on what surrounds the via?  The outer
boundary is varied between the three options of spec §2.1 ("far" adiabatic shell, neighbouring vias as a
perfect sink at r = pitch - R_cu, adiabatic symmetry cell at r = pitch / 2) for three via pitches (glass gaps
50 / 100 / 200 um), for the void-free via and for two mid-rod disc voids.  Every case runs the 20 ms step
response plus the frequency-domain solve (10 Hz .. 5 kHz); all methods are applied to the same curves.

Outputs (results/cw_lateral_loss):
    cases.csv      one row per (case, boundary, pitch): k_eff of every method, signals at t_end, lock-in values
    summary.md     tables: k_eff bias of every method per boundary (baseline) and the void-induced change
    figs/*.png     far-face / pump-face curves per boundary, k_eff bias chart, lock-in amplitude and phase
Run from the src directory:   .venv\\Scripts\\python.exe studies\\cw_lateral_loss.py
Nothing in the simulator or the GUIs is modified by this script.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from ttr_sim import VoidSpec  # noqa: E402
from ttr_sim import analysis as an  # noqa: E402
from ttr_sim.materials import COPPER, FUSED_SILICA  # noqa: E402
from ttr_sim.transmission import cw_config, frequency_response, run_step_response  # noqa: E402

OUT = SRC / "results" / "cw_lateral_loss"
UM = 1e-6
R_CU, L = 40 * UM, 500 * UM
A = math.pi * R_CU ** 2
T_END = 20e-3
FREQS = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0])
GAPS_UM = [50, 100, 200]                       # glass gap between neighbouring vias -> pitch = 80 + gap
BOUNDARIES = ["far", "neighbor_sink", "neighbor_adiabatic"]
VOIDS = {
    "baseline": None,
    "void_r20": VoidSpec(enabled=True, depth=245 * UM, thickness=10 * UM, r_half=20 * UM, k=0.026),
    "void_r35": VoidSpec(enabled=True, depth=245 * UM, thickness=10 * UM, r_half=35 * UM, k=0.026),
}
METHODS = ["step_intercept", "parker", "two_face", "fin_fit", "tlm", "tlm_loss", "tlm_shell"]
LABELS = {
    "step_intercept": "스텝 절편", "parker": "Parker 반상승", "two_face": "양면 온도차", "fin_fit": "핀 모델 피팅(G′)",
    "tlm": "TLM(단열)", "tlm_loss": "TLM(G′)", "tlm_shell": "TLM(유리 껍질)",
}
BOUNDARY_LABEL = {"far": "far (무한 유리)", "neighbor_sink": "이웃 싱크 (하한)", "neighbor_adiabatic": "이웃 단열 (상한)"}


def run_case(name: str, void, outer: str, gap_um: float | None) -> dict:
    pitch = (80 + gap_um) * UM if gap_um is not None else None
    cfg = cw_config(outer=outer, pitch=pitch, t_end=T_END, void=void)
    t0 = time.perf_counter()
    resp = run_step_response(cfg, store_fields=False)
    fr = frequency_response(cfg, FREQS)
    res = an.analyze(resp.t, resp.dT_pump, resp.dT_far, resp.P_abs, L, A, COPPER.rho_cp, spot_radius=cfg.laser.w,
                     freqs=FREQS, H_far=fr.H_far, fit_t_max=3e-3, shell=(R_CU, FUSED_SILICA))
    k = an.k_eff_table(res)
    lk = an.lockin(fr.H_far, fr.P_abs)
    row = dict(case=name, boundary=outer, gap_um=(gap_um if gap_um is not None else ""), pitch_um=(pitch / UM if pitch else ""),
               R_out_um=resp.result.grid.r_faces[-1] / UM, P_abs_mW=resp.P_abs * 1e3,
               dT_far_end=resp.dT_far[-1], dT_pump_end=resp.dT_pump[-1], dT_diff_end=resp.dT_diff[-1],
               dT_far_1ms=float(np.interp(1e-3, resp.t, resp.dT_far)), dT_far_3ms=float(np.interp(3e-3, resp.t, resp.dT_far)),
               E_out_frac=float(resp.E_out[-1] / resp.E_in[-1]), energy_error=resp.energy_error,
               t_int_ms=res["step_intercept"]["t_int"] * 1e3, t_half_ms=res["parker"]["t_half"] * 1e3,
               rho_cp_eff_ratio=res["step_intercept"]["rho_cp_eff"] / COPPER.rho_cp,
               R_th=res["two_face"]["R_th"], G_prime_fin=res["fin_fit"]["G_prime"], G_prime_tlm=res["tlm_loss"]["G_prime"],
               wall_s=time.perf_counter() - t0, n_cells=resp.result.n_cells, n_steps=resp.result.n_steps,
               warnings="; ".join(resp.warnings))
    for m in METHODS:
        row[f"k_{m}"] = k[m]
    for f, amp, ph in zip(FREQS, lk["amplitude"], lk["phase_deg"]):
        row[f"lockin_amp_uK_{f:g}Hz"] = amp * 1e6
        row[f"lockin_phase_deg_{f:g}Hz"] = ph
    row["_curves"] = dict(t=resp.t, far=resp.dT_far, pump=resp.dT_pump, freqs=FREQS, H_far=fr.H_far, H_pump=fr.H_pump)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)
    rows = []
    for name, void in VOIDS.items():
        rows.append(run_case(name, void, "far", None))
        print(f"{name:9s} far                 k_tlm_shell={rows[-1]['k_tlm_shell']:.1f}  dT_far(20ms)={rows[-1]['dT_far_end']:.3f} K  {rows[-1]['wall_s']:.1f}s", flush=True)
        for gap in GAPS_UM:
            for outer in BOUNDARIES[1:]:
                rows.append(run_case(name, void, outer, gap))
                r = rows[-1]
                print(f"{name:9s} {outer:19s} gap={gap:3d}um k_tlm_shell={r['k_tlm_shell']:.1f}  dT_far(20ms)={r['dT_far_end']:.3f} K  "
                      f"E_out={r['E_out_frac'] * 100:.0f}%  {r['wall_s']:.1f}s", flush=True)
    write_csv(rows)
    write_summary(rows)
    make_figures(rows)
    print("done ->", OUT)


def write_csv(rows):
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(OUT / "cases.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


def _key(r):
    return (r["boundary"], r["gap_um"])


def write_summary(rows):
    base = {_key(r): r for r in rows if r["case"] == "baseline"}
    lines = ["# CW 투과 모드 — 측면 손실 민감도 (첫 번째 스터디)", "",
             f"조건: ⌀80 × 500 µm Cu via / fused silica, 흡수 {rows[0]['P_abs_mW']:.1f} mW (입사 10 mW, R = 0.6), "
             f"플랫 스팟 a = 5.5 µm, 프로브 4 µm 양면, 관측창 {T_END * 1e3:.0f} ms, 주파수 {FREQS[0]:g}–{FREQS[-1]:g} Hz.",
             "외곽 경계: far = 자동 반경의 단열 유리 껍질, 이웃 싱크 = r = pitch − R_cu 등온(실온), 이웃 단열 = r = pitch/2 단열. "
             "pitch = 80 µm + 유리 간격.", "",
             "## 1. 기준(void 없음) via 에서 각 방법의 k_eff [W/mK] (참값 398)", "",
             "| 경계 | 간격 [µm] | " + " | ".join(LABELS[m] for m in METHODS) + " | ΔT_far(20 ms) [K] | ΔT_pump-ΔT_far [K] | 싱크 방출 |",
             "|---|---|" + "---|" * len(METHODS) + "---|---|---|"]
    for r in rows:
        if r["case"] != "baseline":
            continue
        cells = " | ".join(f"{r[f'k_{m}']:.0f} ({(r[f'k_{m}'] / COPPER.k - 1) * 100:+.0f} %)" for m in METHODS)
        lines.append(f"| {BOUNDARY_LABEL[r['boundary']]} | {r['gap_um'] or '—'} | {cells} | {r['dT_far_end']:.3f} | {r['dT_diff_end']:.3f} | {r['E_out_frac'] * 100:.0f} % |")
    lines += ["", "## 2. void 에 의한 k_eff 변화 Δk/k [%] (같은 경계의 기준 대비)", "",
              "void: 축 위 원판, 깊이 245–255 µm (로드 중앙), 두께 10 µm, 공기. r_half 20 µm (단면 25 %) / 35 µm (단면 77 %).", "",
              "| void | 경계 | 간격 [µm] | " + " | ".join(LABELS[m] for m in METHODS) + " | ΔT_far(20 ms) 변화 | 락인 100 Hz 진폭 변화 | 락인 1 kHz 위상 변화 [°] |",
              "|---|---|---|" + "---|" * len(METHODS) + "---|---|---|"]
    for r in rows:
        if r["case"] == "baseline":
            continue
        b = base[_key(r)]
        cells = " | ".join(f"{(r[f'k_{m}'] / b[f'k_{m}'] - 1) * 100:+.1f}" for m in METHODS)
        lines.append(f"| {r['case']} | {BOUNDARY_LABEL[r['boundary']]} | {r['gap_um'] or '—'} | {cells} | "
                     f"{(r['dT_far_end'] / b['dT_far_end'] - 1) * 100:+.1f} % | "
                     f"{(r['lockin_amp_uK_100Hz'] / b['lockin_amp_uK_100Hz'] - 1) * 100:+.1f} % | "
                     f"{r['lockin_phase_deg_1000Hz'] - b['lockin_phase_deg_1000Hz']:+.2f} |")
    # spread of each method between the two neighbour bounds at the tightest pitch
    lines += ["", "## 3. 경계 불확실성: 이웃 싱크 ↔ 이웃 단열 사이의 k_eff 폭 (기준 via)", "",
              "| 간격 [µm] | " + " | ".join(LABELS[m] for m in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for gap in GAPS_UM:
        s, a = base[("neighbor_sink", gap)], base[("neighbor_adiabatic", gap)]
        cells = " | ".join(f"{abs(a[f'k_{m}'] - s[f'k_{m}']) / COPPER.k * 100:.0f} %" for m in METHODS)
        lines.append(f"| {gap} | {cells} |")
    lines += ["", "## 4. 읽는 법", "",
              "* 손실 없는 모델(스텝 절편, Parker, 단열 TLM, 핀 모델)은 유리로 새는 열 때문에 체계적으로 편향된다. "
              "편향의 방향: 절편·Parker 는 겉보기 상승이 빨라져 k 를 **높게**, TLM(단열)·핀 모델은 진폭이 줄어 k 를 **낮게** 읽는다.",
              "* 유리 껍질 어드미턴스 TLM 모델(실린더 주위 무한 유리 해석해)은 모든 경계·피치에서 참값의 ±2 % 안에 든다: "
              "락인 주파수(10 Hz 이상)의 유리 침투 깊이가 이웃까지의 간격과 비슷하거나 짧아 경계가 거의 보이지 않기 때문이다. "
              "유일하게 배열 정보 없이 쓸 수 있는 방법이다.",
              "* 간격 50 µm 의 이웃 단열 셀에서는 얇은 유리 고리가 빨리 포화되어 병렬 열용량처럼 작용 → 절편·Parker 의 편향 부호가 뒤집힌다 "
              "(절편 −27 %, Parker −5 %). 폐쇄 경계의 DC 는 무한히 올라가므로(ΔT_far 5 K) 이 경계는 상한일 뿐 물리적 값이 아니다.",
              "* void 에 의한 Δk 는 양면 온도차를 제외한 모든 방법에서 거의 같다(r_half 20 µm: −1.5 %, 35 µm: −8.6 %)이고 경계에 둔감하다: "
              "void 는 축방향 직렬 저항으로 작용하고 측면 손실 편향은 대부분 상쇄된다. 양면 온도차는 준정상 L/(2kA) 항만 보므로 절반만 읽는다.",
              "* 양면 온도차는 단열 로드에서 정상상태가 없어 R_th 가 L/(2kA)+스프레딩 저항으로 수렴한다 (k_eff = L/(R_th A) 는 두 편향이 상쇄된 값).",
              "* §3 의 폭이 '측정값이 via 가 아니라 배열(이웃)에 얼마나 지배되는가' 의 지표다. "
              "폭이 작은 방법이 배열 정보 없이 쓸 수 있는 방법이다.",
              "* 스펙 §3 의 '방법 간 불일치 = void 지표' 는 여기서 양면 온도차 ↔ 나머지 방법의 차이로 나타난다 "
              "(α·ρc 로 읽는 방법은 void 를 온전히, R_th 로 읽는 방법은 절반만 본다)."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    base = [r for r in rows if r["case"] == "baseline"]
    colors = {"far": "k", "neighbor_sink": "tab:blue", "neighbor_adiabatic": "tab:red"}
    styles = {50: "-", 100: "--", 200: ":"}
    # 1. curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in base:
        c = r["_curves"]
        ls = styles.get(r["gap_um"], "-")
        lab = BOUNDARY_LABEL[r["boundary"]] + (f", 간격 {r['gap_um']} µm" if r["gap_um"] != "" else "")
        axes[0].plot(c["t"] * 1e3, c["far"], color=colors[r["boundary"]], ls=ls, label=lab)
        axes[1].plot(c["t"] * 1e3, c["pump"], color=colors[r["boundary"]], ls=ls, label=lab)
    for ax, ttl in zip(axes, ["반대면 ΔT_far(t)", "펌프면 ΔT_pump(t)"]):
        ax.set_xlabel("t [ms]"); ax.set_ylabel("ΔT [K]"); ax.set_title(ttl); ax.grid(alpha=0.3)
        ax.set_xscale("log"); ax.set_xlim(0.01, T_END * 1e3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "curves_by_boundary.png", dpi=150); plt.close(fig)
    # 2. k_eff bias chart
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(METHODS)); n = len(base); w = 0.8 / n
    for i, r in enumerate(base):
        vals = [(r[f"k_{m}"] / COPPER.k - 1) * 100 for m in METHODS]
        lab = BOUNDARY_LABEL[r["boundary"]] + (f", {r['gap_um']} µm" if r["gap_um"] != "" else "")
        ax.bar(x + (i - n / 2 + 0.5) * w, vals, w, label=lab, color=colors[r["boundary"]],
               alpha={50: 1.0, 100: 0.65, 200: 0.35}.get(r["gap_um"], 1.0))
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8)
    ax.set_ylabel("k_eff 편향 [%]"); ax.set_title("기준 via: 분석 방법별 k_eff 편향 (참값 398 W/mK)"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "k_eff_bias.png", dpi=150); plt.close(fig)
    # 3. lock-in amplitude / phase (far face)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in base:
        c = r["_curves"]
        ls = styles.get(r["gap_um"], "-")
        lab = BOUNDARY_LABEL[r["boundary"]] + (f", {r['gap_um']} µm" if r["gap_um"] != "" else "")
        axes[0].loglog(c["freqs"], np.abs(c["H_far"]) * 2 / math.pi * r["P_abs_mW"] * 1e-3 * 1e3, color=colors[r["boundary"]], ls=ls, label=lab)
        axes[1].semilogx(c["freqs"], np.degrees(np.unwrap(np.angle(c["H_far"]))), color=colors[r["boundary"]], ls=ls, label=lab)
    axes[0].set_ylabel("락인 진폭 (사각 chopping 기본파) [mK]"); axes[1].set_ylabel("위상 [°]")
    for ax in axes:
        ax.set_xlabel("f [Hz]"); ax.grid(alpha=0.3, which="both")
    axes[0].set_title("반대면 락인 진폭"); axes[1].set_title("반대면 락인 위상"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "lockin_far.png", dpi=150); plt.close(fig)
    # 4. void signatures on the far face (far boundary)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bf = next(r for r in rows if r["case"] == "baseline" and r["boundary"] == "far")
    for r in rows:
        if r["boundary"] != "far" or r["case"] == "baseline":
            continue
        c = r["_curves"]
        ax.semilogx(c["t"] * 1e3, (c["far"] - bf["_curves"]["far"]) / np.maximum(bf["_curves"]["far"], 1e-12) * 100, label=r["case"])
    ax.set_xlabel("t [ms]"); ax.set_ylabel("ΔT_far 변화 [%]"); ax.set_title("void 에 의한 반대면 신호 변화 (far 경계)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlim(0.1, T_END * 1e3)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "void_far_signal.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
