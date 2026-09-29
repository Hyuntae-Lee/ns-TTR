# F: literature scenarios. F1 Cu 220 ns square (Dillmann 2016), F2 Si 27 ns gate 1-D (Darif & Semmar 2008)
import numpy as np, math
from ttr_sim import Geometry, Laser, Numerics, SimConfig, VoidSpec, run_simulation
from ttr_sim.materials import COPPER, Material
from ttr_sim.analytic import center_step_response
D = COPPER.alpha
out = {}
for key, label, w in (("r", "w=27.5um (radius)", 27.5e-6), ("d", "w=13.75um (27.5 = diameter)", 13.75e-6)):
    tau = 220e-9
    la = Laser(tau_p=tau, profile="square", energy=0.286e-3, w=w, probe_w=0.0, reflectivity=0.9)
    dz = math.sqrt(D * tau) / 10
    cfg = SimConfig(geometry=Geometry(homogeneous_copper=True),
                    void=VoidSpec(enabled=False, depth=5e-6, thickness=2.5e-6, r_half=10e-6),
                    laser=la, numerics=Numerics(dz=dz, t_end_factor=3))
    r = run_simulation(cfg, store_fields=False)
    t = r.times; dT = r.dT_probe
    q0 = la.I0 * (1 - la.reflectivity)
    ref = center_step_response(t, q0, w, COPPER) - center_step_response(t - tau, q0, w, COPPER)
    i = int(np.argmax(dT))
    tm = float(np.interp(1065.0, dT[:i + 1], t[:i + 1])) if dT.max() > 1065 else float("nan")
    print(f"F1 {label}: q0={q0:.3e} W/m2  sim peak {dT.max():.1f} K at {t[i]*1e9:.1f} ns | analytic peak {ref.max():.1f} K "
          f"| 1-D limit {2*q0*math.sqrt(D*tau/math.pi)/COPPER.k:.1f} K | dT=1065 K at {tm*1e9:.1f} ns | cells {r.n_cells} steps {r.n_steps}", flush=True)
    out["t" + key] = t; out["sim" + key] = dT; out["ref" + key] = ref
np.savez("vF1.npz", **out)

# F2 silicon, 1-D (w = 1 m), 27 ns gate pulse, R = 0.59, fluence 1 mJ/cm^2 at centre
SI = Material("Si", k=148.0, rho=2320.0, cp=710.0)
a = SI.alpha; tau = 27e-9; w = 1.0
F = 10.0  # J/m^2 = 1 mJ/cm^2
E = F * math.pi * w ** 2 / 2
la = Laser(tau_p=tau, profile="square", energy=E, w=w, probe_w=5e-6, reflectivity=0.59)
dz = math.sqrt(a * tau) / 10
cfg = SimConfig(geometry=Geometry(homogeneous_copper=True),
                void=VoidSpec(enabled=False, depth=5e-6, thickness=2.5e-6, r_half=10e-6),
                laser=la, numerics=Numerics(dz=dz, t_end_factor=3), copper=SI)
r = run_simulation(cfg, store_fields=False)
t = r.times; dT = r.dT_probe; q0 = la.I0 * (1 - la.reflectivity)
ref = 2 * q0 * (np.sqrt(a * np.clip(t, 0, None) / math.pi) - np.sqrt(a * np.clip(t - tau, 0, None) / math.pi)) / SI.k
peak = dT.max(); Tm = 1687.0 - 293.15
print(f"F2 Si: q0={q0:.4e} W/m2, sim peak {peak:.4f} K per mJ/cm2 (1-D analytic {ref.max():.4f}) "
      f"-> melt threshold (dT={Tm:.0f} K) {Tm/peak:.0f} mJ/cm2 (analytic {Tm/ref.max():.0f}); cells {r.n_cells} steps {r.n_steps}", flush=True)
np.savez("vF2.npz", t=t, sim=dT, ref=ref)
