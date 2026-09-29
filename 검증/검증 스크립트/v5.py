import numpy as np,math
from ttr_sim import DEPTH_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, run_simulation
from ttr_sim.materials import COPPER, AIR, Material
from ttr_sim.solver import baseline_config, void_signal
D=COPPER.alpha
print('--- V5a self-similar sweep (w=3d, probe=1.5d, void r_half=2d, thick=0.5d, tau_p=2d^2/D, ellipsoid, k=0.026)')
for d in (1e-6,2e-6,4e-6,8e-6):
    tau=2*d*d/D; dz=math.sqrt(D*tau)/10
    la=Laser(tau_p=tau,profile="gaussian",energy=1e-9,w=3*d,probe_w=1.5*d)
    cfg=SimConfig(geometry=Geometry(homogeneous_copper=True),void=VoidSpec(enabled=True,depth=d,thickness=.5*d,r_half=2*d,k=AIR.k),laser=la,numerics=Numerics(dz=dz,t_end_factor=6))
    v=run_simulation(cfg,store_fields=False); b=run_simulation(baseline_config(cfg),store_fields=False); s=void_signal(b,v)
    print('d=%g um: (t_peak-t_c)*D/d^2=%.4f  peak_dT/peak_rise=%.5f  contrast_max=%.4f  cells=%d'%(d*1e6,(s['t_peak']-la.t_center)*D/d**2,s['peak_dT']/s['peak_rise_base'],s['peak_contrast'],v.n_cells),flush=True)
print('--- V5b GUI-like sweep (w=10um, probe=5um fixed; real Cu-in-silica geometry)')
for pi in (0,1,2,3,4,5):
    p=DEPTH_PRESETS[pi]; d=p.d_rep
    la=Laser(tau_p=p.tau_p,profile="gaussian",energy=1e-9,w=10e-6,probe_w=5e-6)
    cfg=SimConfig(geometry=Geometry(),void=VoidSpec(enabled=True,depth=d,thickness=.5*d,r_half=min(2*d,35e-6),k=AIR.k),laser=la,numerics=Numerics(dz=p.dz,t_end_mode="void",t_end_factor=8))
    v=run_simulation(cfg,store_fields=False); b=run_simulation(baseline_config(cfg),store_fields=False); s=void_signal(b,v)
    print('d=%g um tau_p=%.3g s: (t_peak-t_c)*D/d^2=%.3f  peak_dT=%.4g K per nJ  peak_dT/peak_rise=%.4f'%(d*1e6,p.tau_p,(s['t_peak']-la.t_center)*D/d**2,s['peak_dT'],s['peak_dT']/s['peak_rise_base']),flush=True)
