import numpy as np,sys
sys.path.insert(0,'..')
from refml import step_response
from scipy.signal import fftconvolve
d=np.load('dig.npz')
def model(layers,G,w1,w2,fwhm=3.2e-9,bw=None,tend=2.2e-6):
    tl=np.logspace(-12,np.log10(tend),240);S=step_response(tl,layers,G,w1,w2)
    dt=2e-11;tf=np.arange(0,tend,dt);Sf=np.interp(tf,np.r_[0,tl],np.r_[0,S])
    tm=3*fwhm;sig=fwhm/2.3548;n=int(2*tm/dt);tk=np.arange(n)*dt;f=np.exp(-(tk-tm)**2/(2*sig**2));f/=f.sum()*dt
    R=fftconvolve(np.gradient(f,dt),Sf)[:len(tf)]*dt
    if bw:
        for b in bw:
            tau=1/(2*np.pi*b);m=int(10*tau/dt);h=np.exp(-np.arange(m)*dt/tau);h/=h.sum();R=fftconvolve(R,h)[:len(tf)]
    return tf-tm,R   # time relative to pulse centre
cases={'a':([(120,2700*897,80e-9),(29.3,3980*778,1)],[90e6],65e-6,8.7e-6,d['fa']),
       'b':([(120,19300*129,100e-9),(349.9,3260*690,1)],[73.6e6],62e-6,3.5e-6,d['fb'])}
res={}
for k,(L,G,w1,w2,fit) in cases.items():
    for tag,bw in (('ideal',None),('bw',(400e6,500e6))):
        t,R=model(L,G,w1,w2,bw=bw);ip=R.argmax();Rn=R/R[ip]
        # paper curve: peak time on its axis
        pf=fit[fit[:,0]>12]      # compare beyond 12 ns on paper axis
        best=None
        for t0 in np.arange(0,12,0.05):   # paper-axis time of pulse centre
            m=np.interp((pf[:,0]-t0)*1e-9,t,Rn);e=np.sqrt(np.mean((np.log(m/pf[:,1]))**2))
            if best is None or e<best[0]:best=(e,t0)
        e,t0=best;m=np.interp((pf[:,0]-t0)*1e-9,t,Rn)
        print(k,tag,'peak delay %.2f ns'%(t[ip]*1e9),'t0=%.2f'%t0,'rms log err %.3f'%e,'ratio paper/model @20,50,100,300,1000:',[round(float(np.interp(x,pf[:,0],pf[:,1]/m)),3) for x in (20,50,100,300,1000)])
        res[k+tag]=(t,Rn,t0)
np.savez('cmp.npz',**{k+'_t':v[0][::5] for k,v in res.items()},**{k+'_R':v[1][::5] for k,v in res.items()},**{k+'_t0':v[2] for k,v in res.items()})
