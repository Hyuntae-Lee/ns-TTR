"""Carslaw & Jaeger type analytic solution: semi-infinite homogeneous solid heated by a
Gaussian surface flux  q(r, t) = q0 exp(-2 r^2 / w^2) f(t).

Step response (f = 1 for t > 0), surface temperature rise at radius r:

    dT(r, 0, t) = q0 w^2 sqrt(a) / (k sqrt(pi)) * Int_0^{sqrt(t)} 2 exp(-2 r^2 / (w^2 + 8 a u^2)) / (w^2 + 8 a u^2) du

which at r = 0 reduces to the closed form  q0 w / (k sqrt(2 pi)) * arctan( sqrt(8 a t) / w ),
with the 1-D limit 2 q0 sqrt(a t / pi) / k for t << w^2 / (8 a) and the Lax steady state
q0 w sqrt(pi/8) / k for t -> infinity.

A square pulse is obtained by superposition, a Gaussian pulse by convolving the step response
with df/dt.  The solution is valid for the void-free baseline as long as heat has not reached
the copper/silica interface (r = 40 um) or the back face.
"""
from __future__ import annotations

import math
import numpy as np

from .materials import Material
from .solver import Laser


def surface_step_response(r, tau, q0: float, w: float, mat: Material, n_quad: int = 64) -> np.ndarray:
    """dT(r, z=0, tau) for a unit step of the Gaussian flux, shape (len(tau), len(r)). tau<=0 -> 0."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    alpha, k = mat.alpha, mat.k
    x, wgt = np.polynomial.legendre.leggauss(n_quad)
    s = np.sqrt(np.clip(tau, 0.0, None))                              # (nt,)
    u = 0.5 * s[:, None] * (x[None, :] + 1.0)                          # (nt, nq)
    du = 0.5 * s[:, None] * wgt[None, :]                               # (nt, nq)
    den = w ** 2 + 8.0 * alpha * u ** 2                                # (nt, nq)
    integrand = np.exp(-2.0 * r[None, None, :] ** 2 / den[:, :, None]) / den[:, :, None]   # (nt, nq, nr)
    integral = 2.0 * np.einsum("tq,tqr->tr", du, integrand)
    return q0 * w ** 2 * math.sqrt(alpha) / (k * math.sqrt(math.pi)) * integral


def center_step_response(tau, q0: float, w: float, mat: Material) -> np.ndarray:
    """Closed form on-axis step response (used as a cross-check of the quadrature)."""
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, None)
    return q0 * w / (mat.k * math.sqrt(2.0 * math.pi)) * np.arctan(np.sqrt(8.0 * mat.alpha * tau) / w)


def analytic_surface_signal(r, t, laser: Laser, mat: Material, n_stairs: int = 200) -> np.ndarray:
    """dT(r, 0, t) for the configured pulse shape, shape (len(t), len(r)).  `t` may be non-uniform."""
    t = np.asarray(t, dtype=float)
    q0 = laser.I0 * (1.0 - laser.reflectivity)
    if laser.profile == "square":
        return (surface_step_response(r, t, q0, laser.w, mat)
                - surface_step_response(r, t - laser.tau_p, q0, laser.w, mat))
    # Gaussian: represent f(t) as a staircase of n_stairs steps over its support and superpose the
    # step responses (Duhamel).  Works for any output time grid.
    t_max = laser.t_center + 4.0 * laser.sigma
    edges = np.linspace(0.0, t_max, n_stairs + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    levels = np.concatenate([[0.0], laser.f(mids), [0.0]])       # f = 0 before 0 and after t_max
    jumps = np.diff(levels)                                        # step heights at the edges
    out = np.zeros((len(t), len(np.atleast_1d(r))))
    for tk, dfk in zip(edges, jumps):
        if dfk != 0.0:
            out += dfk * surface_step_response(r, t - tk, q0, laser.w, mat)
    return out


def analytic_probe_signal(r_centers, probe_wts, t, laser: Laser, mat: Material) -> np.ndarray:
    """Probe-weighted analytic surface temperature rise, evaluated on the numerical grid's radial cells."""
    r_centers = np.asarray(r_centers)
    probe_wts = np.asarray(probe_wts)
    sel = probe_wts > 1e-12 * probe_wts.max()
    Tr = analytic_surface_signal(r_centers[sel], t, laser, mat)
    return Tr @ probe_wts[sel]
