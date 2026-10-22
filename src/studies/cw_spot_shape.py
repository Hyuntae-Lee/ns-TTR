"""CW transmission mode, spec step 7: effect of the true 12 x 10 um flat elliptical pump spot and of an alignment
offset, computed with the half-cylinder 3-D solver, against the equivalent circular spot (radius sqrt(ab)) of
the axisymmetric solver.

Cases: circular spot forced through the 3-D solver (quadrature / solver check), ellipse 12 x 10 um, ellipse
rotated (10 x 12: same thing by symmetry, a check), circular spot offset by 2 and 5 um, ellipse + 3 um offset.
Window 3 ms (the pump-face signal has reached its spreading plateau, the far face its arrival), 5 um grid and
12 azimuthal sectors (24 give the same numbers) to keep the 3-D runs at tens of seconds.  Outputs (results/cw_spot_shape): summary.md, cases.csv, figs/spot_shape.png.
Run from the src directory:   .venv\\Scripts\\python.exe studies\\cw_spot_shape.py
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(SRC))

from ttr_sim.solver3d import count_unknowns, needs_3d  # noqa: E402
from ttr_sim.transmission import cw_config, run_step_response  # noqa: E402

OUT = SRC / "results" / "cw_spot_shape"
UM = 1e-6
T_END = 3e-3
DZ = 5e-6            # 5 um grid: the 3-D runs stay under direct LU (~40 s each); 12 sectors give the same result as 24
CASES = {
    "원형 등가 (2-D 축대칭)": dict(),
    "원형 등가, 3-D 솔버": dict(spot_offset=1e-12),
    "타원 12 × 10 µm": dict(spot_axes=(12 * UM, 10 * UM)),
    "타원 10 × 12 µm (회전 대칭 검사)": dict(spot_axes=(10 * UM, 12 * UM)),
    "원형, 정렬 오차 2 µm": dict(spot_offset=2 * UM),
    "원형, 정렬 오차 5 µm": dict(spot_offset=5 * UM),
    "타원 12 × 10 + 정렬 오차 3 µm": dict(spot_axes=(12 * UM, 10 * UM), spot_offset=3 * UM),
}
T_MARKS = [10e-6, 100e-6, 1e-3, 3e-3]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)
    rows, curves = [], {}
    ref = None
    for name, kw in CASES.items():
        cfg = cw_config(t_end=T_END, dz=DZ, spot_theta_cells=12, **kw)
        t0 = time.perf_counter()
        resp = run_step_response(cfg, store_fields=False)
        wall = time.perf_counter() - t0
        curves[name] = resp
        if ref is None:
            ref = resp
        row = dict(case=name, solver="3-D" if resp.result.is_3d else "2-D", unknowns=(count_unknowns(cfg)[2] if needs_3d(cfg) else resp.result.n_cells),
                   wall_s=wall, P_abs_mW=resp.P_abs * 1e3, energy_error=resp.energy_error)
        for tm in T_MARKS:
            i = int(np.searchsorted(resp.t, tm))
            row[f"pump_{tm * 1e6:g}us_K"] = resp.dT_pump[i]
            row[f"pump_rel_{tm * 1e6:g}us_pct"] = (resp.dT_pump[i] / ref.dT_pump[i] - 1) * 100
            row[f"far_{tm * 1e6:g}us_K"] = resp.dT_far[i]
            row[f"far_rel_{tm * 1e6:g}us_pct"] = (resp.dT_far[i] / ref.dT_far[i] - 1) * 100 if ref.dT_far[i] > 1e-9 else float("nan")
        rows.append(row)
        print(f"{name:32s} {row['solver']} {row['unknowns']:>8,} {wall:6.1f}s  pump(3ms) {resp.dT_pump[-1]:.4f} K ({row['pump_rel_3000us_pct']:+.2f} %)  "
              f"far(3ms) {resp.dT_far[-1]:.5f} K ({row['far_rel_3000us_pct']:+.3f} %)  E={resp.energy_error:.1e}", flush=True)
    with open(OUT / "cases.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# CW 투과 모드 — 타원 스팟 · 정렬 오차 (스펙 7단계, 3-D)", "",
             f"조건: 기준 via, far 경계, 흡수 {ref.P_abs * 1e3:.1f} mW, 관측창 {T_END * 1e3:.0f} ms. 등가 원형 반경 √(ab) = 5.48 µm 의 2-D 축대칭 결과를 기준(0 %)으로 한 상대 차이.", "",
             "| 케이스 | 솔버 | 미지수 | 시간 [s] | " + " | ".join(f"펌프면 {tm * 1e6:g} µs" for tm in T_MARKS) + " | " + " | ".join(f"반대면 {tm * 1e6:g} µs" for tm in T_MARKS) + " |",
             "|---|---|---|---|" + "---|" * (2 * len(T_MARKS))]
    for r in rows:
        pc = " | ".join(f"{r[f'pump_rel_{tm * 1e6:g}us_pct']:+.2f} %" for tm in T_MARKS)
        fc = " | ".join(("—" if math.isnan(r[f'far_rel_{tm * 1e6:g}us_pct']) else f"{r[f'far_rel_{tm * 1e6:g}us_pct']:+.3f} %") for tm in T_MARKS)
        lines.append(f"| {r['case']} | {r['solver']} | {r['unknowns']:,} | {r['wall_s']:.1f} | {pc} | {fc} |")
    lines += ["", "## 읽는 법", "",
              "* '원형 등가, 3-D 솔버' 행은 같은 스팟을 3-D 로 푼 것: 2-D 와의 차이가 3-D 격자·구적의 오차 수준이다.",
              "* 타원 12 × 10 µm 은 등가 원형과 면적·흡수 출력이 같다. 펌프면 프로브(지름 8 µm, 스팟 중심)는 스팟 형상을 보지만 수 % 수준이고, "
              "반대면은 Cu 내 반경 평형(R²/α ≈ 14 µs) 뒤라 거의 보지 않는다 — 스펙 §1 표의 예측대로 납품값(반대면 위상)에는 영향이 없다.",
              "* 정렬 오차는 펌프면 프로브와 스팟이 어긋나는 효과(프로브가 스팟 가장자리/밖을 봄)이며 반대면에는 역시 거의 영향이 없다.",
              "* 10 × 12 행은 12 × 10 과 같아야 한다 (반원 격자의 대칭 검사)."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, resp in curves.items():
        if name.startswith("원형 등가 (2-D"):
            continue
        axes[0].semilogx(resp.t[1:] * 1e3, (resp.dT_pump[1:] / ref.dT_pump[1:] - 1) * 100, label=name)
        sel = ref.dT_far > 1e-3 * ref.dT_far[-1]
        axes[1].semilogx(resp.t[sel] * 1e3, (resp.dT_far[sel] / ref.dT_far[sel] - 1) * 100, label=name)
    axes[0].set_title("펌프면 ΔT 상대 차이 (등가 원형 2-D 기준)"); axes[1].set_title("반대면 ΔT 상대 차이")
    for ax in axes:
        ax.set_xlabel("t [ms]"); ax.set_ylabel("[%]"); ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "spot_shape.png", dpi=150); plt.close(fig)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
