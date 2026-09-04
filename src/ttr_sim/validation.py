"""Verification utilities exposed in the GUI (spec §7):

* energy conservation (computed inside the solver, summarised here),
* grid convergence (spatial refinement at fixed Fourier number, plus a time-step halving),
* analytic baseline comparison (Carslaw & Jaeger semi-infinite solid, Gaussian spot),
* k_void sensitivity sweep over the discrete preset values.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Optional, Sequence

import numpy as np

from .analytic import analytic_probe_signal
from .presets import KVOID_PRESETS
from .solver import SimConfig, SimResult, baseline_config, probe_weights, refined_config, run_simulation, void_signal

Progress = Optional[Callable[[float, str], None]]


def _sub_progress(progress: Progress, i: int, n: int, label: str):
    """Map the progress of sub-job i of n onto the parent progress bar. Nestable: the returned
    callback accepts an optional inner label (as passed by run_pair inside grid_convergence)."""
    if progress is None:
        return None

    def cb(frac: float, inner: str = ""):
        progress((i + frac) / n, f"{label} · {inner}" if inner else label)

    return cb


def run_pair(cfg: SimConfig, progress: Progress = None) -> tuple[SimResult, SimResult, dict]:
    """Run baseline (no void) and void case; return both plus the void-signal metrics."""
    base = run_simulation(baseline_config(cfg), progress=_sub_progress(progress, 0, 2, "baseline (void 없음)"))
    if cfg.void.enabled:
        void = run_simulation(cfg, progress=_sub_progress(progress, 1, 2, "void 포함"))
    else:
        void = base
        if progress:
            progress(1.0, "완료")
    return base, void, void_signal(base, void)


def analytic_comparison(base: SimResult) -> dict:
    """Compare the numerical void-free baseline with the semi-infinite analytic solution."""
    cfg = base.config
    pw = probe_weights(cfg, base.grid)
    ana = analytic_probe_signal(base.grid.r_c, pw, base.times, cfg.laser, cfg.copper)
    num = base.dT_probe
    scale = float(ana.max()) if ana.max() > 0 else 1.0
    rms_rel = float(np.sqrt(np.mean((num - ana) ** 2)) / scale)
    max_rel = float(np.max(np.abs(num - ana)) / scale)
    peak_rel = float((num.max() - ana.max()) / scale)
    notes = []
    d = cfg.diagnostics()
    if not cfg.geometry.homogeneous_copper and d["interface_reached"]:
        notes.append(
            "열 침투 깊이가 구리/실리카 계면(r=40 μm)에 도달했습니다. 해석해는 무한 구리 매질을 가정하므로 "
            "편차의 일부는 물리적(실리카의 낮은 전도율)입니다. '균질 구리 검증 모드'로 이산화 오차만 분리해 볼 수 있습니다."
        )
    if d["L_diff"] > 0.8 * cfg.geometry.L:
        notes.append("열이 후면(z=L)에 도달하여 semi-infinite 가정이 성립하지 않습니다.")
    return dict(t=base.times, analytic=ana, numeric=num, rms_rel=rms_rel, max_rel=max_rel, peak_rel=peak_rel, notes=notes)


def _row(label: str, cfg: SimConfig, base: SimResult, void: SimResult, sig: dict) -> dict:
    return dict(
        label=label, dz=cfg.numerics.dz, dr=cfg.dr, dt=cfg.dt, fo=cfg.numerics.fo,
        n_cells=void.n_cells, n_steps=void.n_steps,
        peak_rise=sig["peak_rise_base"], peak_dT=sig["peak_dT"], peak_contrast=sig["peak_contrast"],
        t_peak=sig["t_peak"], energy_error=max(abs(base.energy_error), abs(void.energy_error)),
        wall=base.wall_time + void.wall_time, t=void.times, dT=sig["dT"], T_probe=void.T_probe,
        T_base=sig["Tb"],
    )


def grid_convergence(
    cfg: SimConfig,
    base: SimResult,
    void: SimResult,
    sig: dict,
    factor: float = 2.0,
    include_coarse: bool = True,
    include_dt_half: bool = True,
    progress: Progress = None,
) -> dict:
    """Re-run with refined (and optionally coarsened) grids and a halved time step.

    Returns rows for each level plus relative changes of the key metrics with respect to the
    finest level and, when three spatial levels are available, an observed order of accuracy.
    """
    jobs = []
    if include_coarse:
        jobs.append(("coarse", refined_config(cfg, 1.0 / factor)))
    jobs.append(("fine", refined_config(cfg, factor)))
    if include_dt_half:
        jobs.append(("dt_half", replace(cfg, numerics=replace(cfg.numerics, fo=cfg.numerics.fo / 2))))

    rows = {"current": _row("현재 격자", cfg, base, void, sig)}
    names = {"coarse": f"거친 격자 (Δ×{factor:g})", "fine": f"조밀 격자 (Δ/{factor:g}, Δt/{factor ** 2:g})",
             "dt_half": "시간 간격 절반 (Fo/2)"}
    for i, (key, c) in enumerate(jobs):
        b, v, s = run_pair(c, progress=_sub_progress(progress, i, len(jobs), names[key]))
        rows[key] = _row(names[key], c, b, v, s)

    ref = rows["fine"]
    comparisons = {}
    for key, r in rows.items():
        if key == "fine":
            continue
        dT_ref = np.interp(r["t"], ref["t"], ref["dT"])
        rms = float(np.sqrt(np.mean((r["dT"] - dT_ref) ** 2)))
        comparisons[key] = dict(
            d_peak_rise=(r["peak_rise"] - ref["peak_rise"]) / ref["peak_rise"],
            d_peak_dT=(r["peak_dT"] - ref["peak_dT"]) / ref["peak_dT"] if ref["peak_dT"] != 0 else float("nan"),
            d_peak_contrast=(r["peak_contrast"] - ref["peak_contrast"]),
            rms_dT_rel=rms / max(abs(ref["peak_dT"]), 1e-300),
        )
    order = None
    if include_coarse:
        e1 = abs(rows["coarse"]["peak_dT"] - rows["current"]["peak_dT"])
        e2 = abs(rows["current"]["peak_dT"] - rows["fine"]["peak_dT"])
        if e1 > 0 and e2 > 0:
            order = math.log(e1 / e2) / math.log(factor)
    return dict(rows=rows, comparisons=comparisons, order=order, factor=factor)


def kvoid_sensitivity(
    cfg: SimConfig,
    base: SimResult,
    ks: Sequence[float] = tuple(p.k for p in KVOID_PRESETS),
    progress: Progress = None,
) -> list[dict]:
    """Run the void case for each k_void candidate (baseline reused) and collect metrics."""
    rows = []
    for i, k in enumerate(ks):
        c = replace(cfg, void=replace(cfg.void, k=k))
        v = run_simulation(c, progress=_sub_progress(progress, i, len(ks), f"k_void = {k:g} W/m·K"))
        s = void_signal(base, v)
        rows.append(dict(
            k=k, peak_dT=s["peak_dT"], peak_contrast=s["peak_contrast"], t_peak=s["t_peak"],
            energy_error=v.energy_error, wall=v.wall_time, t=v.times, dT=s["dT"], T_probe=v.T_probe,
        ))
    ref = rows[0]  # the physically most accurate (smallest k) is the reference
    for r in rows:
        r["d_peak_dT_vs_min"] = (r["peak_dT"] - ref["peak_dT"]) / ref["peak_dT"] if ref["peak_dT"] != 0 else float("nan")
    return rows
