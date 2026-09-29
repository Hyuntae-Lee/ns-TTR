# G: reference curves (multilayer analytic, refml) for Mao et al. 2024 Fig. 6 conditions.
# The simulator cannot run these (needs z-layer stack + interface conductance G); the curves are for the
# user's own comparison with the measured transients in Fig. 6(a), (b).
import numpy as np
from ttr_sim import Laser
from refml import pulse_response
la = Laser(tau_p=3.2e-9, profile="gaussian", energy=1.0, w=1.0, probe_w=1.0, reflectivity=0.0)
tq = np.array([5, 10, 20, 50, 100, 200, 500, 1000]) * 1e-9
t = np.concatenate([np.linspace(1e-10, 20e-9, 400), np.logspace(np.log10(20e-9), np.log10(1.2e-6), 400)[1:]])
cases = {
    "al": ("Al 80 nm / sapphire (G=90 MW/m2K, K=29.3; w1=65, w2=8.7 um)", [(120.0, 2700 * 897, 80e-9), (29.3, 3980 * 778, 1)], [90e6], 65e-6, 8.7e-6),
    "au": ("Au 100 nm / SiC (G=73.6 MW/m2K, K=349.9; w1=62, w2=3.5 um)", [(120.0, 19300 * 129, 100e-9), (349.9, 3260 * 690, 1)], [73.6e6], 62e-6, 3.5e-6),
}
out = {"t": t}
for key, (name, layers, Gs, w1, w2) in cases.items():
    y = pulse_response(t + la.t_center, 1.0, layers, Gs, w1, w2, la)   # t measured from the pulse centre
    y = y / y.max(); out[key] = y
    print("G", name, " ".join(f"{v:.3f}" for v in np.interp(tq, t, y)), flush=True)
np.savez("vG.npz", **out)
