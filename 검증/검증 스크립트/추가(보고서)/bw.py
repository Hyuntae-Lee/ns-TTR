# Detector/oscilloscope bandwidth effect (1st-order low-pass 400 MHz + 500 MHz in cascade) on the reference solution.
# Reproduces the plan's 2026-09-18 §3-G numbers (my calculation, refml semi-infinite / multilayer).
import numpy as np
from ttr_sim import Laser
from ttr_sim.materials import COPPER
from refml import pulse_response
def lowpass(y, dt, fc):
    tau = 1 / (2 * np.pi * fc); a = 1 - np.exp(-dt / tau); out = np.empty_like(y); acc = 0.0
    for i, v in enumerate(y):
        acc += a * (v - acc); out[i] = acc
    return out
def analyse(name, layers, Gs, w1, w2, fwhm, t_end):
    la = Laser(tau_p=fwhm, profile="gaussian", energy=1.0, w=1.0, probe_w=1.0, reflectivity=0.0)
    dt = fwhm / 400; t = np.arange(dt, t_end, dt)
    y = pulse_response(t, 1.0, layers, Gs, w1, w2, la); tc = la.t_center
    yf = lowpass(lowpass(y, dt, 400e6), dt, 500e6)
    i0, i1 = y.argmax(), yf.argmax(); t5 = tc + 5 * fwhm; j = np.searchsorted(t, t5)
    print(f"{name}: peak {y[i0]:.4g} @ {(t[i0]-tc)*1e9:+.2f} ns -> filtered {yf[i1]:.4g} @ {(t[i1]-tc)*1e9:+.2f} ns : "
          f"peak change {(yf[i1]/y[i0]-1)*100:+.2f} %, shift {(t[i1]-t[i0])*1e9:+.2f} ns ; at 5 FWHM after centre: {(yf[j]/y[j]-1)*100:+.2f} % (unnormalised), "
          f"{(yf[j]/yf[i1])/(y[j]/y[i0])*100-100:+.2f} % (normalised)", flush=True)
    return t - tc, y / y.max(), yf / yf.max()
Cu = (COPPER.k, COPPER.rho_cp)
analyse("preset 0 (Cu semi-inf, 10/5 um, FWHM 17.24 ns)", [(*Cu, 1)], [], 10e-6, 5e-6, 17.24e-9, 400e-9)
analyse("Cu semi-inf, 10/5 um, FWHM 3.2 ns", [(*Cu, 1)], [], 10e-6, 5e-6, 3.2e-9, 100e-9)
out = {}
for key, name, layers, Gs, w1, w2 in (("al", "Al80/sapphire", [(120.0, 2700 * 897, 80e-9), (29.3, 3980 * 778, 1)], [90e6], 65e-6, 8.7e-6),
                                      ("au", "Au100/SiC", [(120.0, 19300 * 129, 100e-9), (349.9, 3260 * 690, 1)], [73.6e6], 62e-6, 3.5e-6)):
    t, y, yf = analyse(name + " (Mao 2024, FWHM 3.2 ns)", layers, Gs, w1, w2, 3.2e-9, 1.3e-6)
    out[key + "_t"] = t; out[key] = y; out[key + "_f"] = yf
np.savez("vG_bw.npz", **out)
