"""Superposition study: how does the ns-TTR signal of several voids relate to the single-void responses?

Follow-up of studies/multi_void.py (results/multi_void, read only here).  Same physics and numerics
(multi_void.build_cfg: 40~100 um preset, tau_p = 27.6 us, GUI default energy, pump 10 / probe 5 um, window
1000 us, same detection criteria).  Default void: air, half-width 10 um, thickness 10 um (P4 excepted).

Series
  R    regression: A_d070_axis and A_d070_r25 recomputed, must match the cached curves to 1e-6
  P0   single void on the axis (2-D), top depth 10..480 um every 10 um plus 85/105/115 um: depth library,
       and the building blocks of P1 / C_stack3 sums
  P0'  single void off axis (3-D): depth 70 um at r = 10/15/20/28, 90 um at r = 10/15/20, 150 um at r = 10/20/28
       (r = 0 from P0, r = 25 from the cached A series): radial dependence of the single-void response
  P1   vertical pair on the axis (2-D): top void 70 um, lower void top at 80 + g um, gap g = 5..80 um
  P2   side-by-side pair at 70 um on the r = 25 um ring (3-D): centre spacing s = 22..50 um
  P3   diagonal pair: 70 um on the axis + 90 um at r = 10/15/20/25 um (r = 0 is P1_g10)
  P4   the six shapes of series E at 120 um on the axis (2-D), (10, 10) = S_axis_d120

Outputs (results/superposition): cases/<id>.json|npz, figs/<id>.png, run.log, geometry_check.json.
Geometry rule: every void inside r <= 38 um (2 um from the wall) and no two voids overlapping; violations
are reported and skipped.  2-D cases run first, then 3-D.  Finished cases are skipped on a restart.
Run from the src directory:   .venv\\Scripts\\python.exe studies\\superposition.py [--workers 4] [--only P1,P2]
Analysis: superposition_analysis.py; report: superposition_report.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "studies"))

import multi_void as mv  # noqa: E402
from multi_void import BASE_DEPTH, RH, RING_R, TH, V, build_cfg  # noqa: E402
from ttr_sim.solver3d import count_unknowns, needs_3d  # noqa: E402

OUT = SRC / "results" / "superposition"
CASES_DIR, FIG_DIR, LOG = OUT / "cases", OUT / "figs", OUT / "run.log"
MV_CASES = SRC / "results" / "multi_void" / "cases"
# the shared runner / figure code of multi_void writes to its module globals: point them at this study
mv.CASES_DIR, mv.FIG_DIR, mv.LOG = CASES_DIR, FIG_DIR, LOG

WALL_MARGIN_UM = 2.0
R_MAX_UM = 40.0 - WALL_MARGIN_UM

P0_DEPTHS = sorted(set(range(10, 481, 10)) | {85, 105, 115})
P0P = [(70.0, 10.0), (70.0, 15.0), (70.0, 20.0), (70.0, 28.0), (90.0, 10.0), (90.0, 15.0), (90.0, 20.0),
       (150.0, 10.0), (150.0, 20.0), (150.0, 28.0)]
P1_GAPS = [5.0, 10.0, 20.0, 35.0, 50.0, 80.0]
P2_SPACINGS = [22.0, 25.0, 30.0, 35.0, 42.0, 50.0]
P3_RADII = [10.0, 15.0, 20.0, 25.0]
P4_SHAPES = [(7.1, 20.0), (10.0, 2.0), (10.0, 5.0), (10.0, 20.0), (14.1, 5.0)]


def all_cases() -> list[dict]:
    base = V(BASE_DEPTH)
    C = [dict(id="R_d070_axis", series="R", label="회귀: 단독 70 μm, 축 위 (= A_d070_axis)", voids=[base], ref="A_d070_axis"),
         dict(id="R_d070_r25", series="R", label="회귀: 단독 70 μm, r = 25 μm (= A_d070_r25)", voids=[V(BASE_DEPTH, RING_R)], ref="A_d070_r25")]
    for d in P0_DEPTHS:
        C.append(dict(id=f"S_axis_d{d:03d}", series="P0", label=f"단독 {d} μm, 축 위", voids=[V(float(d))]))
    for d, r in P0P:
        C.append(dict(id=f"S_d{int(d):03d}_r{int(r):02d}", series="P0p", label=f"단독 {d:g} μm, r = {r:g} μm", voids=[V(d, r)]))
    for g in P1_GAPS:
        C.append(dict(id=f"P1_g{int(g):02d}", series="P1", label=f"위아래 쌍 (축 위): 70 μm + 윗면 {80 + g:g} μm, 틈 {g:g} μm",
                      voids=[base, V(80.0 + g)], gap=g))
    for s in P2_SPACINGS:
        dth = math.degrees(2.0 * math.asin(s / (2.0 * RING_R)))
        C.append(dict(id=f"P2_s{int(s):02d}", series="P2", label=f"나란히 쌍 (70 μm, r = 25 μm): 중심 간격 {s:g} μm (Δθ = {dth:.1f}°)",
                      voids=[V(BASE_DEPTH, RING_R, 0.0), V(BASE_DEPTH, RING_R, dth)], spacing=s))
    for r in P3_RADII:
        C.append(dict(id=f"P3_r{int(r):02d}", series="P3", label=f"대각선 쌍: 70 μm 축 위 + 90 μm r = {r:g} μm",
                      voids=[base, V(90.0, r)], rc_lower=r))
    for rh, th in P4_SHAPES:
        C.append(dict(id=f"P4_rh{rh:g}_th{th:g}", series="P4", label=f"단독 120 μm, 반폭 {rh:g} μm, 두께 {th:g} μm",
                      voids=[V(120.0, rh=rh, thick=th)]))
    return C


# ----------------------------------------------------------------------------- geometry check
def geometry_problems(voids: list[dict]) -> list[str]:
    out = []
    for i, v in enumerate(voids):
        if v["rc"] + v["rh"] > R_MAX_UM + 1e-9:
            out.append(f"void {i + 1}: rc + rh = {v['rc'] + v['rh']:.1f} μm > {R_MAX_UM:g} μm")
        if v["depth"] < 0 or v["depth"] + v["thick"] > 500.0:
            out.append(f"void {i + 1}: outside 0..500 μm in z")
    for i in range(len(voids)):
        for j in range(i + 1, len(voids)):
            a, b = voids[i], voids[j]
            dx = a["rc"] * math.cos(math.radians(a["theta"])) - b["rc"] * math.cos(math.radians(b["theta"]))
            dy = a["rc"] * math.sin(math.radians(a["theta"])) - b["rc"] * math.sin(math.radians(b["theta"]))
            dh = math.hypot(dx, dy)
            za = (a["depth"], a["depth"] + a["thick"])
            zb = (b["depth"], b["depth"] + b["thick"])
            z_overlap = min(za[1], zb[1]) - max(za[0], zb[0])
            if dh < a["rh"] + b["rh"] - 1e-9 and z_overlap > 1e-9:
                # bounding boxes overlap: check the spheroids themselves (normalised centre distance along the
                # line of centres, conservative for ellipsoids)
                dz = (a["depth"] + a["thick"] / 2) - (b["depth"] + b["thick"] / 2)
                if (dh / (a["rh"] + b["rh"])) ** 2 + (dz / (a["thick"] / 2 + b["thick"] / 2)) ** 2 < 1.0:
                    out.append(f"void {i + 1} and {j + 1} overlap (horizontal {dh:.1f} μm, vertical {abs(dz):.1f} μm)")
    return out


def check_geometry(cases: list[dict]) -> tuple[list[dict], dict]:
    ok, report = [], {}
    for c in cases:
        probs = geometry_problems(c["voids"])
        report[c["id"]] = probs
        if probs:
            print(f"  SKIP {c['id']}: " + "; ".join(probs), flush=True)
        else:
            ok.append(c)
    return ok, report


# ----------------------------------------------------------------------------- regression
def regression_check(case: dict) -> dict:
    new = np.load(CASES_DIR / f"{case['id']}.npz")
    old = np.load(MV_CASES / f"{case['ref']}.npz")
    same_t = np.array_equal(new["times"], old["times"])
    scale = float(np.max(np.abs(old["dT_diff"])))
    err = float(np.max(np.abs(new["dT_diff"] - old["dT_diff"])) / scale)
    err_base = float(np.max(np.abs(new["dT_base"] - old["dT_base"])) / float(np.max(np.abs(old["dT_base"]))))
    return dict(id=case["id"], ref=case["ref"], same_time_axis=bool(same_t), rel_err_dT_diff=err, rel_err_dT_base=err_base,
                passed=bool(same_t and err < 1e-6 and err_base < 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--refigure", action="store_true")
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
                mv.make_case_figure(c, json.loads(jp.read_text(encoding="utf-8")))
        return
    cases, geo = check_geometry(cases)
    (OUT / "geometry_check.json").write_text(json.dumps(geo, ensure_ascii=False, indent=1), encoding="utf-8")
    t0 = time.time()
    # ---- regression first
    regs = [c for c in cases if c["series"] == "R"]
    for c in regs:
        mv.run_case(c)
    checks = [regression_check(c) for c in regs]
    (OUT / "regression_check.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")
    for ch in checks:
        print(f"  regression {ch['id']}: same axis {ch['same_time_axis']}, rel err dT_diff {ch['rel_err_dT_diff']:.2e}, "
              f"dT_base {ch['rel_err_dT_base']:.2e} -> {'PASS' if ch['passed'] else 'FAIL'}", flush=True)
    if regs and not all(ch["passed"] for ch in checks):
        print("regression failed: stopping before the other cases", flush=True)
        sys.exit(2)
    # ---- 2-D first, then 3-D (largest first within 3-D)
    todo = [c for c in cases if c["series"] != "R" and not (CASES_DIR / f"{c['id']}.json").exists()]
    two_d = [c for c in todo if not needs_3d(build_cfg(c))]
    three_d = [c for c in todo if needs_3d(build_cfg(c))]
    three_d.sort(key=lambda c: count_unknowns(build_cfg(c))[2], reverse=True)
    print(f"{len(cases)} cases: {len(two_d)} 2-D and {len(three_d)} 3-D to run ({len(todo)} total), {args.workers} workers", flush=True)
    for c in two_d:
        mv.run_case(c)
    if three_d:
        if args.workers <= 1:
            for c in three_d:
                mv.run_case(c)
        else:
            from multiprocessing import Pool
            with Pool(args.workers) as pool:
                for _ in pool.imap_unordered(mv.run_case, three_d):
                    pass
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
