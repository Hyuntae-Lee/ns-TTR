# C extension: r_half sweep of a disk void, d = 5 um, h = 2.5 um -> convergence to the layered (Cu/air/Cu) analytic upper bound
import numpy as np, math
from ttr_sim import DEPTH_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, run_simulation
from ttr_sim.materials import COPPER, AIR
from ttr_sim.solver import baseline_config, void_signal
from refml import pulse_response
Cu = (COPPER.k, COPPER.rho_cp); Air = (AIR.k, AIR.rho_cp)
p = DEPTH_PRESETS[2]; d = p.d_rep; h = 0.5 * d
la = Laser(tau_p=p.tau_p, profile="gaussian", energy=1e-9, w=10e-6, probe_w=5e-6)
E = la.energy * (1 - la.reflectivity)
res = {}
for shape in ("ellipse",):
    for rh in (40e-6, 20e-6, 10e-6, 5e-6, 2.5e-6):
        cfg = SimConfig(geometry=Geometry(homogeneous_copper=True),
                        void=VoidSpec(enabled=True, depth=d, thickness=h, r_center=0.0, r_half=rh, k=AIR.k, shape=shape),
                        laser=la, numerics=Numerics(dz=p.dz, t_end_factor=6))
        rv = run_simulation(cfg, store_fields=False); rb = run_simulation(baseline_config(cfg), store_fields=False)
        s = void_signal(rb, rv)
        sig = rv.dT_probe - np.interp(rv.times, rb.times, rb.dT_probe)
        print(f"C2 {shape} r_half={rh*1e6:g} um: peak {s['peak_dT']*1e3:.3f} mK at {s['t_peak']*1e9:.1f} ns, "
              f"max contrast {s['peak_contrast']:.4f}, cells {rv.n_cells}", flush=True)
        res[f"{shape}_{rh*1e6:g}"] = np.vstack((rv.times, sig))
t = res["ellipse_40"][0]
ref_v = pulse_response(t[1:], E, [(*Cu, d), (*Air, h), (*Cu, 1)], [1e15, 1e15], la.w, la.probe_w, la)
ref_b = pulse_response(t[1:], E, [(*Cu, 1)], [], la.w, la.probe_w, la)
sref = ref_v - ref_b
print(f"C2 layered analytic: peak {sref.max()*1e3:.3f} mK at {t[1:][sref.argmax()]*1e9:.1f} ns", flush=True)
np.savez("vC2e.npz", t_ref=t[1:], sref=sref, **res)
