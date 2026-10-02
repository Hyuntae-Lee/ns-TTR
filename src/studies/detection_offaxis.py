"""Detection-limit study, part 2: shallow voids displaced from the rod axis (3-D runs).

The pump and the probe stay on the rod axis; the void is moved sideways by `offset`.  Off-axis voids need the
3-D (r, theta, z) solver, which takes several minutes per case, so the cases run in parallel processes.

    results/detection_map/offaxis_cases.csv

Run from the src directory:   .venv\\Scripts\\python.exe studies\\detection_offaxis.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "studies"))

from detection_map import CRITERIA_DEFAULT, OUT, UM, judge, make_config  # noqa: E402
from ttr_sim.materials import COPPER_C_TR_PROBE  # noqa: E402
from ttr_sim.validation import run_pair  # noqa: E402

# (set name, depth, half-width, thickness, offsets, laser overrides)
SETS = [
    # the case tried by hand: 5 um preset default void (thickness 2.5 um, half-width 10 um)
    ("depth5", 5.0, 10.0, 2.5, [0, 5, 10, 15, 20, 25, 30], {}),
    # a deeper void of the main grid (thickness 10 um)
    ("depth20", 20.0, 10.0, 10.0, [0, 5, 10, 15, 25], {}),
    # same hand-tried void at 25 um off axis, pump widened to a flat-top beam covering the whole rod:
    # probe unchanged (5 um, on axis) / probe as wide as the rod
    ("flat40_probe5", 5.0, 10.0, 2.5, [0, 25], dict(beam="flat", w=40.0 * UM)),
    ("flat40_probe40", 5.0, 10.0, 2.5, [0, 25], dict(beam="flat", w=40.0 * UM, probe_w=40.0 * UM)),
]
WORKERS = 4


def run_one(job):
    name, depth, rh, th, offset, laser_kw = job
    t0 = time.time()
    idx, cfg = make_config(depth, rh, th)
    cfg = replace(cfg, void=replace(cfg.void, r_center=offset * UM), laser=replace(cfg.laser, **laser_kw))
    base, void, sig = run_pair(cfg, store_fields=False)
    T0 = cfg.numerics.T0
    i_pk = int(np.argmax(np.abs(sig["dT"])))
    row = dict(
        set=name, depth_um=depth, half_width_um=rh, thickness_um=th, offset_um=offset,
        beam=cfg.laser.beam, pump_w_um=cfg.laser.w / UM, probe_w_um=cfg.laser.probe_w / UM,
        tau_p_s=cfg.laser.tau_p, energy_J=cfg.laser.energy, window_s=cfg.t_end,
        peak_at_window_end=bool(sig["t_peak"] >= 0.98 * cfg.t_end),
        dT_base_max_K=sig["peak_rise_base"], void_minus_base_K=sig["peak_dT"], t_peak_s=sig["t_peak"],
        base_at_peak_K=float(sig["Tb"][i_pk] - T0), dRR_diff=COPPER_C_TR_PROBE * sig["peak_dT"], ratio=sig["contrast_at_peak"],
        is_3d=bool(void.is_3d), n_cells=int(void.diagnostics.get("n_cells", 0)), wall_s=time.time() - t0,
    )
    j = judge(row, **CRITERIA_DEFAULT)
    row.update(scale=j["scale"], scaled_void_minus_base_K=j["scaled_void_minus_base_K"], scaled_dRR_diff=j["scaled_dRR_diff"],
               noise_ok=j["noise_ok"], ratio_ok=j["ratio_ok"], detectable=j["detectable"], limiting=j["limiting"])
    print("  [%-14s] depth %g  half-width %g  thick %g  offset %2g  void-base %10.4g mK  ratio %7.2f %%  %s  (%.0f s)"
          % (name, depth, rh, th, offset, row["void_minus_base_K"] * 1e3, row["ratio"] * 100,
             "가능" if row["detectable"] else "불가(" + row["limiting"] + ")", row["wall_s"]), flush=True)
    return row


def main():
    """Cases already in offaxis_cache.jsonl are not recomputed, so an interrupted run can simply be restarted."""
    import json
    OUT.mkdir(parents=True, exist_ok=True)
    cache_file = OUT / "offaxis_cache.jsonl"
    rows = [json.loads(ln) for ln in cache_file.read_text(encoding="utf-8").splitlines()] if cache_file.exists() else []
    done = {(r["set"], r["offset_um"]) for r in rows}
    jobs = [(name, d, rh, th, off, kw) for name, d, rh, th, offs, kw in SETS for off in offs if (name, off) not in done]
    jobs.sort(key=lambda j: j[4] == 0)                    # the slow 3-D cases first
    print("%d cases cached, %d to run" % (len(rows), len(jobs)), flush=True)
    t0 = time.time()
    if jobs:
        with Pool(WORKERS) as pool, open(cache_file, "a", encoding="utf-8") as fh:
            for row in pool.imap_unordered(run_one, jobs, chunksize=1):
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
    order = {name: i for i, (name, *_rest) in enumerate(SETS)}
    rows.sort(key=lambda r: (order[r["set"]], r["offset_um"]))
    with open(OUT / "offaxis_cases.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("%d cases, %.0f s -> %s" % (len(rows), time.time() - t0, OUT / "offaxis_cases.csv"), flush=True)
    (OUT / "offaxis_done.flag").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    main()
