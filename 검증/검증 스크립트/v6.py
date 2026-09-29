import numpy as np,math
from ttr_sim import DEPTH_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec, run_simulation
from ttr_sim.materials import COPPER, Material
INS=Material("insulator",k=1e-12,rho=1.0,cp=1.0)
k,rc,a=COPPER.k,COPPER.rho_cp,COPPER.alpha; L=500e-6
def slab_step(t,q):   # surface temp rise of insulated slab, step flux q at z=0 (Carslaw & Jaeger)
    Fo=a*np.clip(t,0,None)/L**2; n=np.arange(1,400)[:,None]
    return q*L/k*(Fo+1/3-(2/np.pi**2)*(np.exp(-n**2*np.pi**2*Fo[None,:])/n**2).sum(0))*(t>0)
for pi,label in ((6,'tau_p=172us (L_diff<<L)'),(8,'tau_p=2.76ms (back face reached)')):
    p=DEPTH_PRESETS[pi]; w=1.0
    la=Laser(tau_p=p.tau_p,profile="square",energy=1.0,w=w,probe_w=5e-6,reflectivity=0.6)
    cfg=SimConfig(geometry=Geometry(),void=VoidSpec(enabled=False,depth=p.d_rep,thickness=.5*p.d_rep,r_half=10e-6),laser=la,numerics=Numerics(dz=p.dz,t_end_factor=3),silica=INS)
    r=run_simulation(cfg,store_fields=False); q=la.I0*(1-la.reflectivity); t=r.times
    ref=slab_step(t,q)-slab_step(t-la.tau_p,q)
    s=ref.max(); print(label,': rms %.2e max %.2e peak %.2e | final rise sim %.6g ref(E/(rho c L)) %.6g | cells %d steps %d | energy err %.1e'%(np.sqrt(np.mean((r.dT_probe-ref)**2))/s,np.max(np.abs(r.dT_probe-ref))/s,(r.dT_probe.max()-s)/s,r.dT_probe[-1],q*la.tau_p/(rc*L),r.n_cells,r.n_steps,r.energy_error),flush=True)
