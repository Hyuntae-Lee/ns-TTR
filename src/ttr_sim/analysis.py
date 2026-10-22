"""Analysis methods for the CW transmission measurement (docs/cw-transmission-spec.md §3).

Pure functions of curves: (t, dT_far, dT_pump, P_abs, L, A, rho_cp, ...).  Nothing here touches the solver,
so the same functions apply to simulated and to measured curves.

Conventions
-----------
* t [s] strictly increasing, t[0] = 0; dT [K] temperature rise of the probe spot; P_abs [W] absorbed power.
* L rod length, A = pi R_cu^2 rod cross-section, rho_cp volumetric heat capacity of the rod material.
* Frequency responses H(omega) are complex temperature amplitudes per unit absorbed power amplitude [K/W],
  convention dT(t) = Re(H e^{i omega t}) for P(t) = Re(e^{i omega t}): phase = arg H <= 0 (lag).

Deliverable (absolute k_eff)
----------------------------
invert_tlm(..., shell=(R_cu, silica))   phase-based least squares of the rod-in-glass transmission line
    (`tlm_response` with `lateral_admittance`) to the far-face frequency response.  Phase only, so the
    result does not depend on the reflectivity (absorbed power) or on the thermoreflectance coefficient.
    The frequency response may come from lock-in data or from a recorded step transient converted with
    `frequency_response_from_step`; both give the same k_eff (tests/test_cw.py).
phase_noise_sensitivity   Monte-Carlo spread / bias of that k_eff for a given phase noise (lock-in requirement).

Void indicators (relative to a reference via in the same neighbourhood, never an absolute k)
-------------------------------------------------------------------------------------------
step_intercept        linear asymptote of dT_far -> t_int (-> apparent alpha = L^2 / (6 t_int))
parker                impulse response d dT_far/dt -> half-rise time (-> apparent alpha = 0.1388 L^2 / t_half)
two_face_resistance   R_th(t) = (dT_pump - dT_far) / P_abs; a void changes R_th(t*).  No k_eff: the adiabatic rod
                      has no steady state (quasi-steady difference P L / (2 k A), not P L / (k A)) and the
                      indicator's sensitivity falls linearly from the pump face to the far face (flux ~ 1 - z/L).
Their lateral-loss bias (results/cw_lateral_loss) is systematic and cancels in the comparison with the reference.

Helpers: frequency_response_from_step, lockin (chopped-pump fundamental), tlm_response, lateral_admittance,
analyze (everything on one curve set), compare / void_indicators (baseline-vs-void differences).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.special import kv as _Kv

from .materials import COPPER, Material

PARKER_CONSTANT = 0.1388        # alpha = 0.1388 L^2 / t_half (Parker 1961, insulated slab)


# ----------------------------------------------------------------------------- helpers
def _window(t: np.ndarray, t_from: Optional[float], t_to: Optional[float], frac: tuple) -> np.ndarray:
    """Boolean mask of the fit window: absolute [t_from, t_to] when given, else fractions of the last time."""
    t_end = float(t[-1])
    lo = t_from if t_from is not None else frac[0] * t_end
    hi = t_to if t_to is not None else frac[1] * t_end
    m = (t >= lo) & (t <= hi)
    if m.sum() < 3:
        raise ValueError("fit window holds fewer than 3 samples")
    return m


def linear_fit(t, y, t_from=None, t_to=None, frac=(0.6, 1.0)) -> dict:
    """Least-squares line y = slope t + offset over the window; also the window mask and RMS residual."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    m = _window(t, t_from, t_to, frac)
    slope, offset = np.polyfit(t[m], y[m], 1)
    resid = y[m] - (slope * t[m] + offset)
    return dict(slope=float(slope), offset=float(offset), mask=m, rms=float(np.sqrt(np.mean(resid ** 2))),
                t_from=float(t[m][0]), t_to=float(t[m][-1]))


def impulse_response(t, dT_step) -> np.ndarray:
    """d dT/dt on the (non-uniform) time grid: the response to a unit-area impulse of the same absorbed power."""
    t, s = np.asarray(t, float), np.asarray(dT_step, float)
    return np.gradient(s, t, edge_order=1)


def spreading_resistance(k: float, a: float) -> float:
    """Spreading resistance of a flat circular spot of radius a on a half-space, ~ 1 / (4 k a) (spec §1)."""
    return 1.0 / (4.0 * k * a)


# ----------------------------------------------------------------------------- 1. step intercept
def step_intercept(t, dT_far, L: float, rho_cp: Optional[float] = None, P_abs: Optional[float] = None,
                   A: Optional[float] = None, t_from=None, t_to=None, frac=None) -> dict:
    """Time intercept of the linear asymptote of the far-face step response -> alpha = L^2 / (6 t_int).

    Without a window the asymptote is the tangent at the point of maximum slope (exact for a lossless rod,
    whose slope saturates; for a rod losing heat sideways the curve bends over afterwards and this tangent is
    the "linear part" one would draw).  With `t_from`/`t_to` (absolute) or `frac` (fractions of t[-1]) a
    straight line is fitted over that window instead.

    Returns t_int, slope, alpha, k_eff (= alpha rho_cp when rho_cp is given) and, when P_abs and A are given,
    the effective heat capacity per unit volume implied by the slope (P_abs / (slope A L); equals rho_cp for a
    lossless rod, larger when the surroundings share the heat)."""
    if t_from is None and t_to is None and frac is None:
        t, y = np.asarray(t, float), np.asarray(dT_far, float)
        h = impulse_response(t, y)
        i = int(np.argmax(h))
        slope = float(h[i])
        offset = float(y[i] - slope * t[i])
        fit = dict(rms=0.0, t_from=float(t[i]), t_to=float(t[i]))
    else:
        fit = linear_fit(t, dT_far, t_from, t_to, frac if frac is not None else (0.6, 1.0))
        slope, offset = fit["slope"], fit["offset"]
    t_int = -offset / slope if slope > 0 else float("nan")
    alpha = L ** 2 / (6.0 * t_int) if t_int > 0 else float("nan")
    out = dict(t_int=float(t_int), slope=slope, offset=offset, alpha=float(alpha), fit_rms=fit["rms"],
               t_from=fit["t_from"], t_to=fit["t_to"])
    if rho_cp is not None:
        out["k_eff"] = alpha * rho_cp
    if P_abs is not None and A is not None and slope > 0:
        out["rho_cp_eff"] = P_abs / (slope * A * L)
    return out


# ----------------------------------------------------------------------------- 2. Parker half-rise
def parker(t, dT_far, L: float, rho_cp: Optional[float] = None) -> dict:
    """Parker's method on the impulse response h = d dT_far/dt: t_half is the time at which h reaches half of
    its maximum in the window; alpha = 0.1388 L^2 / t_half.  With lateral losses h peaks and decays, so the
    maximum is the peak; for a lossless rod it is the plateau P_abs / (rho c A L)."""
    t = np.asarray(t, float)
    h = impulse_response(t, dT_far)
    i_max = int(np.argmax(h))
    h_max = float(h[i_max])
    if h_max <= 0:
        return dict(t_half=float("nan"), alpha=float("nan"), h=h, t_peak=float("nan"), h_max=h_max)
    # first crossing of h_max / 2 before the maximum, linearly interpolated
    k = int(np.argmax(h[:i_max + 1] >= 0.5 * h_max))
    if k == 0:
        t_half = float(t[0])
    else:
        t_half = float(np.interp(0.5 * h_max, [h[k - 1], h[k]], [t[k - 1], t[k]]))
    alpha = PARKER_CONSTANT * L ** 2 / t_half if t_half > 0 else float("nan")
    out = dict(t_half=t_half, alpha=float(alpha), h=h, t_peak=float(t[i_max]), h_max=h_max)
    if rho_cp is not None:
        out["k_eff"] = alpha * rho_cp
    return out


# ----------------------------------------------------------------------------- 3. two-face temperature difference
def two_face_resistance(t, dT_pump, dT_far, P_abs: float, L: float, A: float, t_star: Optional[float] = None,
                        k_ref: float = COPPER.k, spot_radius: Optional[float] = None) -> dict:
    """Void indicator: R_th(t) = (dT_pump - dT_far) / P_abs and its value at t_star (default: the last time).

    Not a k_eff method.  The reference levels returned for the plot are the through-flow series resistance
    L / (k_ref A) (heat leaving through the far face) and the adiabatic quasi-steady value L / (2 k_ref A)
    (uniform heating rate, the floating via), plus the pump-spot spreading resistance 1 / (4 k_ref a) when the
    spot radius is given.  A void raises R_th in proportion to the flux through its plane, which falls
    linearly from the pump face to the far face: voids near the far face are nearly invisible here."""
    t = np.asarray(t, float)
    R = (np.asarray(dT_pump, float) - np.asarray(dT_far, float)) / P_abs
    if t_star is None:
        t_star = float(t[-1])
    R_star = float(np.interp(t_star, t, R))
    out = dict(t_star=t_star, R_th=R_star, R_th_curve=R, R_through=L / (k_ref * A), R_quasi_steady=L / (2.0 * k_ref * A))
    if spot_radius is not None:
        out["R_spread"] = spreading_resistance(k_ref, spot_radius)
    return out


# ----------------------------------------------------------------------------- 5. lock-in: frequency responses
def frequency_response_from_step(t, dT_step, freqs, P_abs: float = 1.0, tail: str = "constant") -> np.ndarray:
    """H(omega) = (1 / P_abs) int_0^inf h(t) e^{-i omega t} dt with h = d s/dt the impulse response of the step
    response s(t).  h is taken at the nodes (second-order gradient on the non-uniform grid) and integrated as a
    piecewise-linear function, exactly per segment: this keeps the slope of s continuous, so the long late-time
    steps of a geometric time grid do not inject spurious high-frequency content (a piecewise-linear s has slope
    kinks at every node whose spectra only cancel when the slope is constant).
    `tail`: "constant" continues the final h beyond t[-1] (exact for a lossless rod, an upper bound of the
    low-frequency content otherwise), "none" truncates.  Returns complex H [K/W] per frequency in `freqs` [Hz]."""
    t, s = np.asarray(t, float), np.asarray(dT_step, float)
    h = np.gradient(s, t, edge_order=1)
    w = 2.0 * math.pi * np.asarray(freqs, float)[:, None]
    dt = np.diff(t)
    e0, e1 = np.exp(-1j * w * t[:-1]), np.exp(-1j * w * t[1:])
    I0 = (e0 - e1) / (1j * w)                                       # int e^{-iwt} over the segment
    I1 = -dt * e1 / (1j * w) + (e0 - e1) / (1j * w) ** 2            # int (t - t_k) e^{-iwt} over the segment
    m = (h[1:] - h[:-1]) / dt
    H = np.sum(h[:-1] * I0 + m * I1, axis=1)
    if tail == "constant":
        H = H + h[-1] * np.exp(-1j * w[:, 0] * t[-1]) / (1j * w[:, 0])
    elif tail != "none":
        raise ValueError("tail must be 'constant' or 'none'")
    return H / P_abs


def lockin(H, P_abs: float, chop: str = "square") -> dict:
    """Lock-in reading at the fundamental for a pump chopped between 0 and P_abs: square chopping has a
    fundamental of amplitude 2 P_abs / pi, sinusoidal modulation (0..P_abs) one of P_abs / 2.
    Returns amplitude [K], phase [deg, lag negative] and the complex fundamental."""
    H = np.asarray(H, complex)
    fund = (2.0 / math.pi if chop == "square" else 0.5) * P_abs * H
    return dict(amplitude=np.abs(fund), phase_deg=np.degrees(np.angle(fund)), complex=fund,
                H_amplitude=np.abs(H), H_phase_deg=np.degrees(np.angle(H)))


def lateral_admittance(freqs, R: float, shell: Material) -> np.ndarray:
    """Thermal admittance per unit length [W/(m K)] of an infinite medium (`shell`) surrounding a cylinder of
    radius R whose surface temperature oscillates at `freqs`: Y = 2 pi R k_s q K1(q R) / K0(q R), q = sqrt(i w / a_s)
    (Carslaw & Jaeger, cylinder in an infinite medium; the glass heat capacity makes Y grow with frequency and
    vanish at DC)."""
    w = 2.0 * math.pi * np.asarray(freqs, float)
    q = np.sqrt(1j * w / shell.alpha).astype(complex)
    x = q * R
    return 2.0 * math.pi * R * shell.k * q * _Kv(1, x) / _Kv(0, x)


def tlm_response(freqs, k: float, rho_cp: float, L: float, A: float, G_prime: float = 0.0, shell=None) -> tuple:
    """1-D transmission line of a rod (section A, length L) heated at z = 0 with a sinusoidal absorbed power,
    insulated at z = L, uniform lateral loss G' per unit length: gamma = sqrt((i omega rho c A + G') / (k A)),
        H_pump = coth(gamma L) / (k A gamma),   H_far = 1 / (k A gamma sinh(gamma L))      [K/W].
    `shell` = (R, Material) adds the frequency-dependent admittance of a surrounding infinite medium
    (`lateral_admittance`) to G': the rod-in-glass model of the CW via.  Returns (H_pump, H_far)."""
    w = 2.0 * math.pi * np.asarray(freqs, float)
    Y = G_prime + (lateral_admittance(freqs, shell[0], shell[1]) if shell is not None else 0.0)
    gamma = np.sqrt((1j * w * rho_cp * A + Y) / (k * A)).astype(complex)
    gL = gamma * L
    # numerically safe coth / 1/sinh for large |gamma L|
    with np.errstate(over="ignore", invalid="ignore"):
        sinh, cosh = np.sinh(gL), np.cosh(gL)
        H_pump = np.where(np.abs(gL) < 300, cosh / sinh, 1.0) / (k * A * gamma)
        H_far = np.where(np.abs(gL) < 300, 1.0 / sinh, 0.0) / (k * A * gamma)
    return H_pump, H_far


def _tlm_features(H, use: str) -> np.ndarray:
    H = np.asarray(H, complex)
    if use == "phase":
        return np.unwrap(np.angle(H))
    if use == "amplitude":
        return np.log(np.abs(H))
    if use == "both":
        return np.concatenate([np.log(np.abs(H)), np.unwrap(np.angle(H))])
    raise ValueError("use must be 'phase', 'amplitude' or 'both'")


def invert_tlm(freqs, H_far, rho_cp: float, L: float, A: float, fit_loss: bool = False, k0: float = COPPER.k,
               shell=None, use: str = "phase") -> dict:
    """k_eff (and G' when fit_loss) from the far-face frequency response by least squares of the TLM model over
    all frequencies.  `use` = "phase" (default: independent of the absorbed power and of the thermoreflectance
    coefficient, so of R and c_tr), "amplitude" or "both".  `shell` = (R, Material): include the surrounding
    glass through `lateral_admittance` - the deliverable configuration (then G' only carries what the glass
    model misses).  The input may be lock-in data or `frequency_response_from_step` of a recorded transient."""
    freqs = np.asarray(freqs, float)
    H = np.asarray(H_far, complex)
    target = _tlm_features(H, use)

    def resid(p):
        k = math.exp(p[0])
        G = math.exp(p[1]) if fit_loss else 0.0
        _, Hm = tlm_response(freqs, k, rho_cp, L, A, G, shell)
        return _tlm_features(Hm, use) - target

    p0 = [math.log(k0)] + ([math.log(1e-3)] if fit_loss else [])
    res = least_squares(resid, p0, method="trf")
    k = math.exp(res.x[0])
    G = math.exp(res.x[1]) if fit_loss else 0.0
    return dict(k_eff=k, G_prime=G, alpha=k / rho_cp, rms=float(np.sqrt(np.mean(res.fun ** 2))), success=bool(res.success),
                use=use, n_freq=int(len(freqs)))


def phase_noise_sensitivity(freqs, H_far, rho_cp: float, L: float, A: float, shell, sigma_deg, n_trials: int = 200,
                            seed: int = 0, k_true: Optional[float] = None) -> dict:
    """Monte-Carlo spread of the phase-based glass-shell inversion when every phase carries independent Gaussian
    noise of `sigma_deg` (one value or a list).  Returns, per sigma: mean k, standard deviation, bias relative to
    `k_true` (default: the noise-free inversion), and the 5-95 % range.  Amplitude noise does not enter."""
    rng = np.random.default_rng(seed)
    freqs = np.asarray(freqs, float)
    H = np.asarray(H_far, complex)
    k0 = invert_tlm(freqs, H, rho_cp, L, A, shell=shell)["k_eff"]
    k_ref = k0 if k_true is None else k_true
    rows = []
    for sd in np.atleast_1d(np.asarray(sigma_deg, float)):
        ks = np.empty(n_trials)
        for i in range(n_trials):
            Hn = H * np.exp(1j * np.radians(rng.normal(0.0, sd, size=len(freqs))))
            ks[i] = invert_tlm(freqs, Hn, rho_cp, L, A, shell=shell, k0=k0)["k_eff"]
        rows.append(dict(sigma_deg=float(sd), k_mean=float(ks.mean()), k_std=float(ks.std(ddof=1)),
                         bias_rel=float(ks.mean() / k_ref - 1.0), std_rel=float(ks.std(ddof=1) / k_ref),
                         p05=float(np.percentile(ks, 5)), p95=float(np.percentile(ks, 95))))
    return dict(k_noise_free=k0, k_ref=k_ref, n_trials=n_trials, rows=rows)


# ----------------------------------------------------------------------------- all methods / comparison
DELIVERABLE = "tlm_shell"
METHOD_LABELS = {
    "tlm_shell": "k_eff: 위상 기반 TLM 역산 (유리 껍질 어드미턴스 모델) — 절대값, R·c_tr 무관",
    "step_intercept": "void 지표: 스텝 절편 t_int (측면 손실로 겉보기 α 편향, 기준 via 와 비교)",
    "parker": "void 지표: Parker 반상승 t½ (측면 손실로 겉보기 α 편향, 기준 via 와 비교)",
    "two_face": "void 지표: 양면 온도차 R_th(t*) (펌프면 쪽 void 에 민감, 반대면 쪽은 거의 안 보임)",
}
INDICATOR_KEYS = {"step_intercept": "t_int", "parker": "t_half", "two_face": "R_th", "tlm_shell": "k_eff"}


def analyze(t, dT_pump, dT_far, P_abs: float, L: float, A: float, rho_cp: float, spot_radius: Optional[float] = None,
            freqs=None, H_far=None, t_star: Optional[float] = None, intercept_window=None, shell=None,
            k_ref: float = COPPER.k) -> dict:
    """The deliverable and the void indicators on one curve set.  `freqs`/`H_far` (lock-in data, or
    `frequency_response_from_step` of the transient) with `shell` = (R, Material) give the phase-based
    glass-shell inversion `tlm_shell`.  `intercept_window` = (t_from, t_to) fits a line there instead of the
    maximum-slope tangent.  Returns {method: result dict}."""
    t = np.asarray(t, float)
    win = dict(t_from=intercept_window[0], t_to=intercept_window[1]) if intercept_window else {}
    y_far = np.asarray(dT_far, float)
    out = dict(
        step_intercept=step_intercept(t, y_far, L, rho_cp, P_abs, A, **win),
        parker=parker(t, y_far, L, rho_cp),
        two_face=two_face_resistance(t, dT_pump, y_far, P_abs, L, A, t_star, k_ref, spot_radius),
    )
    if freqs is not None and H_far is not None and shell is not None and len(np.atleast_1d(freqs)) > 0:
        out["tlm_shell"] = invert_tlm(freqs, H_far, rho_cp, L, A, shell=shell, use="phase")
    return out


def void_indicators(base: dict, void: dict) -> dict:
    """Relative change of each method's indicator (INDICATOR_KEYS) from the reference via to the void via:
    {method: (base, void, delta, rel)}."""
    out = {}
    for m, key in INDICATOR_KEYS.items():
        if m in base and m in void and key in base[m] and key in void[m]:
            b, v = float(base[m][key]), float(void[m][key])
            out[m] = (b, v, v - b, (v - b) / b if b != 0 else float("nan"))
    return out


def k_eff_table(results: dict) -> dict:
    """{method: k or apparent k} from `analyze` output: the deliverable's k_eff, and for the transient indicators
    the apparent alpha * rho_cp (biased by the lateral loss, only meaningful relative to a reference); nan
    where a method gives none (two-face)."""
    return {m: float(r.get("k_eff", float("nan"))) for m, r in results.items()}


def compare(base: dict, void: dict) -> dict:
    """Baseline-vs-void differences of every scalar in `analyze` outputs: {method: {key: (base, void, delta, rel)}}."""
    out = {}
    for m in base:
        if m not in void:
            continue
        rows = {}
        for key, vb in base[m].items():
            vv = void[m].get(key)
            if isinstance(vb, (int, float)) and not isinstance(vb, bool) and isinstance(vv, (int, float)) and not isinstance(vv, bool):
                d = vv - vb
                rows[key] = (float(vb), float(vv), float(d), float(d / vb) if vb not in (0.0,) and np.isfinite(vb) else float("nan"))
        out[m] = rows
    return out
