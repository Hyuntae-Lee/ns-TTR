import numpy as np,math
from dataclasses import replace
from ttr_sim import DEPTH_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, run_simulation
from ttr_sim.materials import COPPER, FUSED_SILICA, AIR, Material
from ttr_sim.solver import baseline_config, void_signal
from refml import pulse_response
Cu=(COPPER.k,COPPER.rho_cp); Air=(AIR.k,AIR.rho_cp)
def rel(num,ref): s=ref.max(); return float(np.sqrt(np.mean((num-ref)**2))/s), float(np.max(np.abs(num-ref))/s), float((num.max()-ref.max())/s)
# V4: wide disk void (box, r_half = R_cu) vs layered Cu(d)/air(h)/Cu analytic
p=DEPTH_PRESETS[2]; d=p.d_rep; h=0.5*d
la=Laser(tau_p=p.tau_p,profile="gaussian",energy=1e-9,w=10e-6,probe_w=5e-6)
cfg=SimConfig(geometry=Geometry(homogeneous_copper=True),void=VoidSpec(enabled=True,depth=d,thickness=h,r_center=0.0,r_half=40e-6,k=AIR.k,shape="box"),laser=la,numerics=Numerics(dz=p.dz,t_end_factor=6))
rv=run_simulation(cfg,store_fields=False); rb=run_simulation(baseline_config(cfg),store_fields=False)
E=la.energy*(1-la.reflectivity)
ref_v=pulse_response(rv.times[1:],E,[(*Cu,d),(*Air,h),(*Cu,1)],[1e15,1e15],la.w,la.probe_w,la)
ref_b=pulse_response(rv.times[1:],E,[(*Cu,1)],[],la.w,la.probe_w,la)
print('V4 void-run vs layered analytic (rms,max,peak):',rel(rv.dT_probe[1:],ref_v),'cells',rv.n_cells)
sig=rv.dT_probe[1:]-np.interp(rv.times[1:],rb.times,rb.dT_probe); sref=ref_v-ref_b
print('V4 void SIGNAL: sim peak %.5f K at %.1f ns ; analytic peak %.5f K at %.1f ns ; rms/peak %.4f'%(sig.max(),rv.times[1:][sig.argmax()]*1e9,sref.max(),rv.times[1:][sref.argmax()]*1e9,np.sqrt(np.mean((sig-sref)**2))/sref.max()))
print('   baseline peak',rb.dT_probe.max(),' contrast at end sim/analytic', sig[-1]/ (rb.dT_probe[-1]), sref[-1]/ref_b[-1], 't_end[ns]',rv.times[-1]*1e9)
np.savez('v4.npz',t=rv.times[1:],sim_v=rv.dT_probe[1:],ref_v=ref_v,ref_b=ref_b,sim_b=np.interp(rv.times[1:],rb.times,rb.dT_probe))
