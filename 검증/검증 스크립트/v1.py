import numpy as np,time
from ttr_sim import DEPTH_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, run_simulation
from ttr_sim.materials import COPPER, FUSED_SILICA, AIR, Material
from refml import pulse_response
Cu=(COPPER.k,COPPER.rho_cp)
def rel(num,ref): s=ref.max(); return float(np.sqrt(np.mean((num-ref)**2))/s), float(np.max(np.abs(num-ref))/s), float((num.max()-ref.max())/s)
out={}
# V1: homogeneous copper, Gaussian pulse, presets 0 and 2
for pi in (0,2):
    p=DEPTH_PRESETS[pi]
    la=Laser(tau_p=p.tau_p,profile="gaussian",energy=1e-9,w=10e-6,probe_w=5e-6)
    cfg=SimConfig(geometry=Geometry(homogeneous_copper=True),void=VoidSpec(enabled=False,depth=p.d_rep,thickness=.5*p.d_rep,r_half=2*p.d_rep),laser=la,numerics=Numerics(dz=p.dz,t_end_factor=20))
    r=run_simulation(cfg,store_fields=False)
    ref=pulse_response(r.times[1:],la.energy*(1-la.reflectivity),[(*Cu,1)],[],la.w,la.probe_w,la)
    out[f'V1 preset{pi}']=rel(r.dT_probe[1:],ref)+(r.n_cells,r.n_steps)
    print('V1',pi,out[f'V1 preset{pi}'],flush=True)
# V3: w0 invariance
p=DEPTH_PRESETS[2]; res=[]
for w,pw in ((10e-6,5e-6),(5e-6,10e-6),(7.90569e-6,7.90569e-6)):
    la=Laser(tau_p=p.tau_p,profile="gaussian",energy=1e-9,w=w,probe_w=pw)
    cfg=SimConfig(geometry=Geometry(homogeneous_copper=True),void=VoidSpec(enabled=False,depth=p.d_rep,thickness=.5*p.d_rep,r_half=2*p.d_rep),laser=la,numerics=Numerics(dz=p.dz,t_end_factor=10))
    r=run_simulation(cfg,store_fields=False); res.append((r.times,r.dT_probe))
for i in (1,2):
    b=np.interp(res[0][0],res[i][0],res[i][1]); print('V3 rel diff vs (10,5):',rel(b,res[0][1]),flush=True)
