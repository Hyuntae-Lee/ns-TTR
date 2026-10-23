"""CW transmission mode: lock-in phase-noise requirement for the k_eff deliverable.

The deliverable is the phase-based glass-shell TLM inversion (ttr_sim.analysis.invert_tlm with shell=...).
This study adds independent Gaussian phase noise to the far-face lock-in phases of the void-free via (far
boundary, direct frequency-domain solve) and inverts many noisy realisations: the spread and bias of k_eff
versus the phase noise (0.05 .. 2 deg) and versus the frequency set (number and range of lock-in points).
It also shows the same inversion on the step-transient FFT route, so both acquisition modes are covered.

Outputs (results/cw_phase_noise): summary.md, noise.csv, figs/k_vs_phase_noise.png
Run from the src directory:   .venv\\Scripts\\python.exe studies\\cw_phase_noise.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")          # Windows console (cp949) cannot print the labels
sys.path.insert(0, str(SRC))

from ttr_sim import analysis as an  # noqa: E402
from ttr_sim.materials import COPPER, FUSED_SILICA  # noqa: E402
from ttr_sim.transmission import cw_config, frequency_response, lockin_dataset, run_step_response  # noqa: E402

OUT = SRC / "results" / "cw_phase_noise"
UM = 1e-6
R_CU, L = 40 * UM, 500 * UM
A = math.pi * R_CU ** 2
SHELL = (R_CU, FUSED_SILICA)
SIGMAS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
FREQ_SETS = {
    "9점 50 Hz–2 kHz (GUI 기본)": np.logspace(math.log10(50), math.log10(2000), 9),
    "21점 50 Hz–2 kHz": np.logspace(math.log10(50), math.log10(2000), 21),
    "9점 10 Hz–5 kHz": np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000.0]),
    "5점 50 Hz–2 kHz": np.array([50, 100, 200, 500, 2000.0]),
    "3점 100 Hz–1 kHz": np.array([100, 300, 1000.0]),
    "1점 500 Hz": np.array([500.0]),
    "1점 100 Hz": np.array([100.0]),
    "17점 10 Hz–5 kHz": np.logspace(1, math.log10(5000), 17),
}
N_TRIALS = 400


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)
    cfg = cw_config(f_min=10.0)
    resp = run_step_response(cfg, store_fields=False)
    rows, curves = [], {}
    for name, freqs in FREQ_SETS.items():
        fr = frequency_response(cfg, freqs)
        mc = an.phase_noise_sensitivity(freqs, fr.H_far, COPPER.rho_cp, L, A, SHELL, SIGMAS, n_trials=N_TRIALS, seed=1,
                                        k_true=COPPER.k)
        curves[name] = mc
        for r in mc["rows"]:
            rows.append(dict(freq_set=name, n_freq=len(freqs), sigma_deg=r["sigma_deg"], k_noise_free=mc["k_noise_free"],
                             k_mean=r["k_mean"], k_std=r["k_std"], bias_pct=r["bias_rel"] * 100, std_pct=r["std_rel"] * 100,
                             p05=r["p05"], p95=r["p95"]))
        print(f"{name:24s} k0={mc['k_noise_free']:.1f}  " + "  ".join(f"σ={r['sigma_deg']:g}°: ±{r['std_rel'] * 100:.2f}%" for r in mc["rows"]), flush=True)
    # the transient route: same inversion on the step FFT (no added noise) and on the mixed dataset
    freqs = FREQ_SETS["9점 50 Hz–2 kHz (GUI 기본)"]
    fr = frequency_response(cfg, freqs)
    ds = lockin_dataset(resp, fr)
    k_direct = an.invert_tlm(freqs, fr.H_far, COPPER.rho_cp, L, A, shell=SHELL)["k_eff"]
    k_mixed = an.invert_tlm(ds.freqs, ds.H_far, COPPER.rho_cp, L, A, shell=SHELL)["k_eff"]
    k_step = an.invert_tlm(ds.freqs, lockin_dataset(resp, None, freqs).H_far, COPPER.rho_cp, L, A, shell=SHELL)["k_eff"]
    with open(OUT / "noise.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# CW 투과 모드 — 락인 위상 잡음에 대한 k_eff 민감도", "",
             f"조건: 기준 via (void 없음), far 경계, 흡수 {resp.P_abs * 1e3:.1f} mW, 반대면 프로브. 주파수 영역 직접 풀이의 위상에 "
             f"독립 가우시안 잡음 σ 를 더해 {N_TRIALS} 회 역산 (위상만 사용, 진폭 잡음은 무관).", "",
             "## 1. k_eff 의 표준편차 [%] (참값 398 W/mK 기준) — 주파수 세트 × 위상 잡음", "",
             "| 주파수 세트 | n | 잡음 없는 k | " + " | ".join(f"σ = {s:g}°" for s in SIGMAS) + " |",
             "|---|---|---|" + "---|" * len(SIGMAS)]
    for name, mc in curves.items():
        cells = " | ".join(f"±{r['std_rel'] * 100:.2f} ({r['bias_rel'] * 100:+.2f})" for r in mc["rows"])
        lines.append(f"| {name} | {len(FREQ_SETS[name])} | {mc['k_noise_free']:.1f} | {cells} |")
    lines += ["", "괄호 안은 평균의 편향 [%]. 1 σ 값이므로 95 % 구간은 약 2배.", "",
              "## 2. 취득 경로", "",
              f"| 경로 | k_eff [W/mK] |", "|---|---|",
              f"| 락인 (주파수 영역 직접 풀이) | {k_direct:.1f} |",
              f"| 스텝 과도응답 → FFT (전 주파수) | {k_step:.1f} |",
              f"| 혼합 (1/창 미만 직접, 이상 FFT) | {k_mixed:.1f} |", "",
              "## 3. 읽는 법", "",
              "* 위상 기반이므로 반사율·c_tr 의 불확실성은 k_eff 에 들어오지 않는다. 남는 것은 위상 잡음과 모델(유리 껍질) 오차(~1–2 %)뿐이다.",
              "* 1 % 의 k_eff 재현성을 원하면 위 표에서 ±1 % 가 되는 σ 가 락인의 위상 요구 사양이다. "
              "주파수 점을 늘리면 √n 으로 좋아지고, 민감도가 큰 중간 주파수(100 Hz–1 kHz: 위상이 k 에 가장 가파르게 의존)를 포함하는 것이 중요하다.",
              "* 1점 측정도 가능하나(위상 하나 → k 하나) 모델 오차를 가려낼 수 없다. 여러 주파수의 잔차가 모델 검증이 된다."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, mc in curves.items():
        ax.loglog([r["sigma_deg"] for r in mc["rows"]], [r["std_rel"] * 100 for r in mc["rows"]], "o-", label=name)
    ax.axhline(1.0, color="k", lw=0.8, ls="--"); ax.text(SIGMAS[0], 1.05, "1 %", fontsize=8)
    ax.set_xlabel("락인 위상 잡음 σ [°]"); ax.set_ylabel("k_eff 표준편차 [%]"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    ax.set_title("위상 잡음 → k_eff 재현성 (유리 껍질 TLM, 위상 역산)")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "k_vs_phase_noise.png", dpi=150); plt.close(fig)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
