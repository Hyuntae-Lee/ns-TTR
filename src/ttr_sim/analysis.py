"""Analysis methods for the CW transmission measurement (docs/cw-transmission-spec.md §3).

Pure functions of curves: (t, dT_far, dT_pump, P_abs, L, A, rho_cp, ...).  Nothing here touches the solver,
so the same functions apply to simulated and to measured curves.

Conventions
-----------
* t [s] strictly increasing, t[0] = 0; dT [K] temperature rise of the probe spot; P_abs [W] absorbed power.
* L rod length, A = pi R_cu^2 rod cross-section, rho_cp volumetric heat capacity of the rod material.
* Frequency responses H(omega) are complex temperature amplitudes per unit absorbed power amplitude [K/W],
  convention dT(t) = Re(H e^{i omega t}) for P(t) = Re(e^{i omega t}): phase = arg H <= 0 (lag).

Methods
-------
step_intercept        linear asymptote of dT_far -> t_int -> alpha = L^2 / (6 t_int)   (CW-step Parker)
parker                impulse response d dT_far/dt -> half-rise time -> alpha = 0.1388 L^2 / t_half
two_face_resistance   (dT_pump - dT_far) / P_abs = R_th(t) -> k_eff = L / (R_th A)   (optionally minus the
                      pump-spot spreading resistance).  NB: an adiabatic rod has no steady state; its
                      quasi-steady face difference is P L / (2 k A), so this k_eff drifts with t (spec §3).
fit_fin_model         least squares of the 1-D fin model (k_eff, lateral loss G') to the far-face curve
frequency_response_from_step   H(omega) from the step response (LTI: H = FT{d s/dt})
lockin                amplitude / phase of the lock-in fundamental for a square-chopped pump
tlm_response          1-D transmission-line (ABCD) response of a rod with uniform lateral loss
invert_tlm            k_eff (and G') from measured H_far(omega) by least squares on the TLM
analyze / compare     all methods at once, and the baseline-vs-void differences
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
                        spot_radius: Optional[float] = None) -> dict:
    """R_th(t) = (dT_pump - dT_far) / P_abs and k_eff = L / (R_th A), at t_star (default: the last time).

    With `spot_radius` the pump-spot spreading resistance 1 / (4 k a) is subtracted self-consistently:
    R_th = L / (k A) + 1 / (4 k a)  ->  k = (L / A + 1 / (4 a)) / R_th."""
    t = np.asarray(t, float)
    R = (np.asarray(dT_pump, float) - np.asarray(dT_far, float)) / P_abs
    k_curve = L / (A * np.where(R > 0, R, np.nan))
    if t_star is None:
        t_star = float(t[-1])
    R_star = float(np.interp(t_star, t, R))
    k_eff = L / (R_star * A) if R_star > 0 else float("nan")
    out = dict(t_star=t_star, R_th=R_star, k_eff=float(k_eff), R_th_curve=R, k_eff_curve=k_curve)
    if spot_radius is not None and R_star > 0:
        k_corr = (L / A + 1.0 / (4.0 * spot_radius)) / R_star
        out.update(k_eff_spot_corrected=float(k_corr), R_spread=spreading_resistance(k_corr, spot_radius))
    return out


# ----------------------------------------------------------------------------- 4. 1-D fin model fit
def fin_step_response(t, k: float, rho_cp: float, L: float, A: float, P_abs: float, G_prime: float = 0.0,
                      n_terms: int = 200) -> np.ndarray:
    """Far-face rise of a 1-D rod (length L, section A) heated by the absorbed power P_abs at z = 0 from t = 0,
    insulated at z = L, losing heat laterally with a uniform conductance per unit length G' [W/(m K)] to the
    ambient (fin model):  rho c A dT/dt = k A d^2T/dz^2 - G' T.

    With m = G' / (rho c A) and lambda_n = n^2 pi^2 alpha / L^2:
        T_far = P_abs / (rho c A L) [ (1 - e^{-m t}) / m + 2 sum_n (-1)^n (1 - e^{-(lambda_n + m) t}) / (lambda_n + m) ]
    (G' = 0 gives the Carslaw-Jaeger slab solution)."""
    t = np.asarray(t, float)[..., None]
    alpha = k / rho_cp
    m = G_prime / (rho_cp * A)
    n = np.arange(1, n_terms + 1, dtype=float)
    lam = (n * math.pi) ** 2 * alpha / L ** 2
    # time-independent part 2 sum (-1)^n / (lambda_n + m) in closed form (sum (-1)^n / (n^2 + c^2) =
    # pi / (2 c sinh(pi c)) - 1 / (2 c^2); -> -pi^2 / 12 for c -> 0), so the truncation only affects t -> 0
    c2 = m * L ** 2 / (math.pi ** 2 * alpha)
    if c2 > 1e-6:
        c = math.sqrt(c2)
        const = 2.0 * L ** 2 / (math.pi ** 2 * alpha) * (math.pi / (2.0 * c * math.sinh(math.pi * c)) - 1.0 / (2.0 * c2))
    else:
        const = -L ** 2 / (6.0 * alpha) * (1.0 - c2 * math.pi ** 2 * 7.0 / 60.0)   # next term of the expansion
    series = const - 2.0 * np.sum((-1.0) ** n * np.exp(-(lam + m) * t) / (lam + m), axis=-1)
    tt = t[..., 0]
    first = (1.0 - np.exp(-m * tt)) / m if m > 0 else tt
    return np.where(tt > 0.0, P_abs / (rho_cp * A * L) * (first + series), 0.0)


def fit_fin_model(t, dT_far, P_abs: float, L: float, A: float, rho_cp: float, fit_loss: bool = True,
                  t_min: float = 0.0, k0: float = COPPER.k, n_terms: int = 100) -> dict:
    """Least squares of `fin_step_response` to the far-face curve for k_eff (and G' when fit_loss).
    Parameters are fitted in log space (k > 0, G' > 0).  Returns k_eff, G_prime, alpha, the fitted curve on t,
    the RMS residual and the residual relative to the RMS of the data."""
    t, y = np.asarray(t, float), np.asarray(dT_far, float)
    sel = t >= t_min
    tt, yy = t[sel], y[sel]
    t_end = float(t[-1])
    scale = rho_cp * A / t_end                 # G' = (m t_end) * scale: the loss is fitted as the dimensionless rate m t_end

    def unpack(p):
        k = math.exp(p[0])
        G = math.exp(p[1]) * scale if fit_loss else 0.0
        return k, G

    def resid(p):
        k, G = unpack(p)
        return fin_step_response(tt, k, rho_cp, L, A, P_abs, G, n_terms) - yy

    lo = [math.log(1e-2 * k0)] + ([-25.0] if fit_loss else [])
    hi = [math.log(1e2 * k0)] + ([5.0] if fit_loss else [])
    p0 = [math.log(k0)] + ([-3.0] if fit_loss else [])
    res = least_squares(resid, p0, bounds=(lo, hi), method="trf", x_scale="jac")
    k, G = unpack(res.x)
    curve = fin_step_response(t, k, rho_cp, L, A, P_abs, G, n_terms)
    resid = curve[sel] - yy
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return dict(k_eff=k, G_prime=G, alpha=k / rho_cp, curve=curve, rms=rms,
                rms_rel=rms / float(np.sqrt(np.mean(yy ** 2))), success=bool(res.success), t_min=t_min)


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


def invert_tlm(freqs, H_far, rho_cp: float, L: float, A: float, fit_loss: bool = False, k0: float = COPPER.k,
               shell=None) -> dict:
    """k_eff (and G' when fit_loss) from the far-face frequency response by least squares of log-amplitude
    and phase of the TLM model over all frequencies.  `shell` = (R, Material): include the surrounding glass
    through `lateral_admittance` (then G' only carries what the glass model misses)."""
    freqs = np.asarray(freqs, float)
    H = np.asarray(H_far, complex)
    target = np.concatenate([np.log(np.abs(H)), np.unwrap(np.angle(H))])

    def resid(p):
        k = math.exp(p[0])
        G = math.exp(p[1]) if fit_loss else 0.0
        _, Hm = tlm_response(freqs, k, rho_cp, L, A, G, shell)
        return np.concatenate([np.log(np.abs(Hm)), np.unwrap(np.angle(Hm))]) - target

    p0 = [math.log(k0)] + ([math.log(1e-3)] if fit_loss else [])
    res = least_squares(resid, p0, method="trf")
    k = math.exp(res.x[0])
    G = math.exp(res.x[1]) if fit_loss else 0.0
    return dict(k_eff=k, G_prime=G, alpha=k / rho_cp, rms=float(np.sqrt(np.mean(res.fun ** 2))), success=bool(res.success))


# ----------------------------------------------------------------------------- all methods / comparison
METHOD_LABELS = {
    "step_intercept": "스텝 절편법 (t 절편 → α = L²/6t)",
    "parker": "Parker 반상승 (임펄스 응답 → α = 0.1388 L²/t½)",
    "two_face": "양면 온도차 (R_th → k_eff = L/(R_th A))",
    "fin_fit": "곡선 피팅 (1-D 핀 모델, k_eff + G′)",
    "fin_fit_lossless": "곡선 피팅 (1-D 단열 막대, k_eff)",
    "tlm": "락인 TLM 역산 (단열 막대, k_eff)",
    "tlm_loss": "락인 TLM 역산 (균일 손실 G′ 추가, k_eff + G′)",
    "tlm_shell": "락인 TLM 역산 (유리 껍질 어드미턴스 모델, k_eff)",
}


def analyze(t, dT_pump, dT_far, P_abs: float, L: float, A: float, rho_cp: float, spot_radius: Optional[float] = None,
            freqs=None, H_far=None, t_star: Optional[float] = None, intercept_window=None, fit_t_max: Optional[float] = None,
            shell=None) -> dict:
    """Every method on one curve set.  `freqs`/`H_far` (lock-in data, e.g. from the frequency-domain solve or
    from `frequency_response_from_step`) enable the TLM inversions; `shell` = (R, Material) adds the glass-shell
    TLM model.  `intercept_window` = (t_from, t_to) fits a line there instead of the maximum-slope tangent;
    `fit_t_max` limits the fin-model fit to t <= fit_t_max.  Returns {method: result dict}."""
    t = np.asarray(t, float)
    win = dict(t_from=intercept_window[0], t_to=intercept_window[1]) if intercept_window else {}
    sel = t <= (fit_t_max if fit_t_max is not None else t[-1])
    y_far = np.asarray(dT_far, float)
    out = dict(
        step_intercept=step_intercept(t, y_far, L, rho_cp, P_abs, A, **win),
        parker=parker(t, y_far, L, rho_cp),
        two_face=two_face_resistance(t, dT_pump, y_far, P_abs, L, A, t_star, spot_radius),
        fin_fit=fit_fin_model(t[sel], y_far[sel], P_abs, L, A, rho_cp, fit_loss=True),
        fin_fit_lossless=fit_fin_model(t[sel], y_far[sel], P_abs, L, A, rho_cp, fit_loss=False),
    )
    if freqs is not None and H_far is not None and len(np.atleast_1d(freqs)) > 0:
        out["tlm"] = invert_tlm(freqs, H_far, rho_cp, L, A, fit_loss=False)
        if len(np.atleast_1d(freqs)) >= 2:
            out["tlm_loss"] = invert_tlm(freqs, H_far, rho_cp, L, A, fit_loss=True)
        if shell is not None:
            out["tlm_shell"] = invert_tlm(freqs, H_far, rho_cp, L, A, fit_loss=False, shell=shell)
    return out


def k_eff_table(results: dict) -> dict:
    """{method: k_eff} from `analyze` output (nan where a method gives no k)."""
    return {m: float(r.get("k_eff", r.get("k_eff_spot_corrected", float("nan")))) for m, r in results.items()}


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
