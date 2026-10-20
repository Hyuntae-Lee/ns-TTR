"""One-off repair of cached multi-void cases computed before baseline_config kept the extra voids' grid zones:
the 2-D baseline is recomputed on the void run's grid (seconds), the difference curves and metrics are
rebuilt from the cached void-run probe curve, and the per-case figure is redrawn.  Also renames the
whole-rod random cases F_seed* -> F0_seed* (series F0) made before series F was restricted to depths >= 70 um.

Run from the src directory:   .venv\\Scripts\\python.exe studies\\multi_void_repair.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "studies"))

from detection_map import CRITERIA_DEFAULT, judge  # noqa: E402
from multi_void import CASES_DIR, FIG_DIR, all_cases, build_cfg, make_case_figure  # noqa: E402
from ttr_sim.materials import COPPER_C_TR_PROBE  # noqa: E402
from ttr_sim.solver import baseline_config, build_grid, run_simulation  # noqa: E402


def rename_f0():
    for seed in range(1, 6):
        for ext in (".json", ".npz"):
            old = CASES_DIR / f"F_seed{seed}{ext}"
            if old.exists():
                old.rename(CASES_DIR / f"F0_seed{seed}{ext}")
        old = FIG_DIR / f"F_seed{seed}.png"
        if old.exists():
            old.rename(FIG_DIR / f"F0_seed{seed}.png")
        jp = CASES_DIR / f"F0_seed{seed}.json"
        if jp.exists():
            r = json.loads(jp.read_text(encoding="utf-8"))
            if r["id"] != f"F0_seed{seed}":
                r["id"], r["series"], r["label"] = f"F0_seed{seed}", "F0", f"무작위 10개, 로드 전체 (시드 {seed})"
                jp.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


def repair(case: dict) -> None:
    cid = case["id"]
    jp, npz = CASES_DIR / f"{cid}.json", CASES_DIR / f"{cid}.npz"
    if not jp.exists():
        return
    row = json.loads(jp.read_text(encoding="utf-8"))
    if row.get("baseline_grid_fixed"):
        return
    cfg = build_cfg(case)
    assert build_grid(baseline_config(cfg)).n_cells == build_grid(cfg).n_cells
    base = run_simulation(baseline_config(cfg), store_fields=False)
    d = dict(np.load(npz))
    t = d["times"]
    dT_b = np.interp(t, base.times, base.dT_probe)
    dT_v = d["dT_void"]
    dT = dT_v - dT_b
    valid = dT_b > 1e-3 * dT_b.max()
    contrast = np.zeros_like(dT)
    contrast[valid] = dT[valid] / dT_b[valid]
    i_pk = int(np.argmax(np.abs(dT)))
    i_c = int(np.argmax(np.abs(contrast)))
    row.update(
        dT_base_max_K=float(dT_b.max()), t_base_peak_s=float(t[int(np.argmax(dT_b))]),
        void_minus_base_K=float(dT[i_pk]), t_peak_s=float(t[i_pk]),
        ratio=float(dT[i_pk] / dT_b[i_pk]) if dT_b[i_pk] > 0 else float("nan"), base_at_peak_K=float(dT_b[i_pk]),
        dRR_diff=COPPER_C_TR_PROBE * float(dT[i_pk]),
        contrast_max=float(contrast[i_c]), t_contrast_max_s=float(t[i_c]), dT_at_contrast_max_K=float(dT[i_c]),
        peak_at_window_end=bool(t[i_pk] >= 0.98 * cfg.t_end), baseline_grid_fixed=True,
    )
    row.update(judge(row, **CRITERIA_DEFAULT))
    d["dT_base"], d["dT_diff"], d["contrast"] = dT_b, dT, contrast
    d["base_times"] = base.times
    np.savez_compressed(npz, **d)
    jp.write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
    make_case_figure(case, row)
    print(f"{cid:12s} peak {row['void_minus_base_K'] * 1e3:8.4f} mK at {row['t_peak_s'] * 1e6:6.1f} us  ratio {row['ratio'] * 100:6.3f} %  "
          f"contrast max {row['contrast_max'] * 100:6.3f} %  {'가능' if row['detectable'] else '불가(' + row['limiting'] + ')'}", flush=True)


def main():
    rename_f0()
    for c in all_cases():
        if len(c["voids"]) > 1:
            repair(c)


if __name__ == "__main__":
    main()
