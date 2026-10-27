"""Stage-2 timing probe.  argv[1] = "base": same-grid 3-D void-free via (the void filled with copper, so the
grid, the azimuthal sectors and the solver are those of the void run); "void": the air void (run concurrently
with "base" to measure the parallel slowdown).  Case: r_off 20 um, sphere 44.5 um, centre depth 250 um,
default settings (20 ms window, dz 2.5 um).  Saves the curves and a JSON line for the Δ evaluation."""
import json, math, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
import numpy as np
from ttr_sim import VoidSpec
from ttr_sim import analysis as an
from ttr_sim.materials import COPPER, FUSED_SILICA
from ttr_sim.solver3d import needs_3d, count_unknowns
from ttr_sim.transmission import cw_config, run_step_response, lockin_dataset

OUTDIR = Path(__file__).resolve().parent
UM = 1e-6; L = 500 * UM; R = 40 * UM; A = math.pi * R ** 2
FREQS = np.logspace(math.log10(50), math.log10(2000), 9)
mode = sys.argv[1]
d = 44.5
kw = dict(k=0.026) if mode == "void" else dict(k=COPPER.k, rho_cp=COPPER.rho_cp)
v = VoidSpec(enabled=True, depth=(250 - d / 2) * UM, thickness=d * UM, r_half=d / 2 * UM, r_center=20 * UM, theta=0.0, shape="ellipse", **kw)
cfg = cw_config(void=v, f_min=50.0)
print(mode, "needs_3d", needs_3d(cfg), "unknowns", count_unknowns(cfg), time.strftime("%H:%M:%S"), flush=True)
t0 = time.perf_counter()
r = run_step_response(cfg, store_fields=False)
w = time.perf_counter() - t0
ds = lockin_dataset(r, None, FREQS)
k = an.invert_tlm(FREQS, ds.H_far, COPPER.rho_cp, L, A, shell=(R, FUSED_SILICA))["k_eff"]
np.savez(OUTDIR / f"pair_{mode}.npz", t=r.t, dT_pump=r.dT_pump, dT_far=r.dT_far, H_far=ds.H_far, H_pump=ds.H_pump, P_abs=r.P_abs, freqs=FREQS)
res = dict(mode=mode, is_3d=bool(r.result.is_3d), wall_min=w / 60, k_eff=k, steps=int(r.result.n_steps), Eerr=float(r.energy_error))
(OUTDIR / f"pair_{mode}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1), flush=True)
