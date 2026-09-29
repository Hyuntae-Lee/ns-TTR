# A/E curves for the report figures (re-run of the v1 / v6 cases with the curves saved)
import numpy as np, math
from ttr_sim import DEPTH_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, run_simulation
from ttr_sim.materials import COPPER, Material
from refml import pulse_response
Cu = (COPPER.k, COPPER.rho_cp); out = {}
for pi in (0, 2):
    p = DEPTH_PRESETS[pi]; la = Laser(tau_p=p.tau_p, profile="gaussian", energy=1e-9, w=10e-6, probe_w=5e-6)
    cfg = SimConfig(geometry=Geometry(homogeneous_copper=True),
                    void=VoidSpec(enabled=False, depth=p.d_rep, thickness=.5 * p.d_rep, r_half=2 * p.d_rep),
                    laser=la, numerics=Numerics(dz=p.dz, t_end_factor=20))
    r = run_simulation(cfg, store_fields=False)
    ref = pulse_response(r.times[1:], la.energy * (1 - la.reflectivity), [(*Cu, 1)], [], la.w, la.probe_w, la)
    out[f"A{pi}_t"] = r.times[1:]; out[f"A{pi}_sim"] = r.dT_probe[1:]; out[f"A{pi}_ref"] = ref
    print("A", pi, "done", flush=True)
INS = Material("insulator", k=1e-12, rho=1.0, cp=1.0); k, rc, a = COPPER.k, COPPER.rho_cp, COPPER.alpha; L = 500e-6
def slab_step(t, q):
    Fo = a * np.clip(t, 0, None) / L ** 2; n = np.arange(1, 400)[:, None]
    return q * L / k * (Fo + 1 / 3 - (2 / np.pi ** 2) * (np.exp(-n ** 2 * np.pi ** 2 * Fo[None, :]) / n ** 2).sum(0)) * (t > 0)
for pi in (6, 8):
    p = DEPTH_PRESETS[pi]; la = Laser(tau_p=p.tau_p, profile="square", energy=1.0, w=1.0, probe_w=5e-6, reflectivity=0.6)
    cfg = SimConfig(geometry=Geometry(), void=VoidSpec(enabled=False, depth=p.d_rep, thickness=.5 * p.d_rep, r_half=10e-6),
                    laser=la, numerics=Numerics(dz=p.dz, t_end_factor=3), silica=INS)
    r = run_simulation(cfg, store_fields=False); q = la.I0 * (1 - la.reflectivity)
    ref = slab_step(r.times, q) - slab_step(r.times - la.tau_p, q)
    out[f"E{pi}_t"] = r.times; out[f"E{pi}_sim"] = r.dT_probe; out[f"E{pi}_ref"] = ref
    print("E", pi, "done", flush=True)
np.savez("vA.npz", **out)
