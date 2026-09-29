"""Reference: laterally infinite multilayer stack, Gaussian pump/probe, surface flux (impedance recursion + fixed Talbot)."""
import numpy as np
from scipy.signal import fftconvolve
_xg,_wg=np.polynomial.legendre.leggauss(240)
def Ztop(k,s,layers,Gs,back='semi'):
    K,C,L=layers[-1]; lam=np.sqrt(4*np.pi**2*k**2+s*C/K)
    Z=1/(K*lam) if back=='semi' else 1/(K*lam*np.tanh(lam*L))
    for i in range(len(layers)-2,-1,-1):
        Z=Z+1/Gs[i]; K,C,L=layers[i]; lam=np.sqrt(4*np.pi**2*k**2+s*C/K); g=K*lam; th=np.tanh(lam*L); Z=(Z+th/g)/(1+g*Z*th)
    return Z
def _F(s,layers,Gs,w0sq,back):
    kmax=6/(np.pi*np.sqrt(w0sq)); k=0.5*kmax*(_xg+1); wk=0.5*kmax*_wg; s=np.atleast_1d(s)[:,None]
    return (np.exp(-np.pi**2*k**2*w0sq)*Ztop(k[None,:],s,layers,Gs,back)*2*np.pi*k*wk).sum(1)/s[:,0]
def talbot(F,t,M=24):
    kk=np.arange(1,M); d=np.empty(M,complex); d[0]=2*M/5; th=kk*np.pi/M
    d[1:]=2*kk*np.pi/5*(1/np.tan(th)+1j); g=np.empty(M,complex); g[0]=0.5*np.exp(d[0])
    g[1:]=(1+1j*th*(1+1/np.tan(th)**2)-1j/np.tan(th))*np.exp(d[1:]); return 0.4/t*np.real((g*F(d/t)).sum())
def step_response(t,layers,Gs,w1,w2,back='semi'):
    w0sq=(w1**2+w2**2)/2; return np.array([talbot(lambda s:_F(s,layers,Gs,w0sq,back),ti) for ti in np.atleast_1d(t)])
def pulse_response(t,E_abs,layers,Gs,w1,w2,laser,back='semi',n_fine=400001):
    """dT(t) for the simulator's Laser object (uses its f, fprime, tau_p: integral f dt = tau_p)."""
    t=np.asarray(t); tend=t.max(); tl=np.logspace(-12,np.log10(tend),220); S=step_response(tl,layers,Gs,w1,w2,back)
    tf=np.linspace(0,tend,n_fine); dt=tf[1]-tf[0]; Sf=np.interp(tf,np.r_[0,tl],np.r_[0,S])
    P=E_abs/laser.tau_p   # absorbed power at f=1
    if laser.profile=='square':
        return P*(np.interp(t,tf,Sf)-np.interp(np.clip(t-laser.tau_p,0,None),tf,Sf)*(t>laser.tau_p))
    fp=laser.fprime(tf); R=float(laser.f(0.0))*Sf+fftconvolve(fp,Sf)[:n_fine]*dt
    return P*np.interp(t,tf,R)
