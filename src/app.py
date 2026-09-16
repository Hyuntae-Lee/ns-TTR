"""ns-TTR Void Detection Simulator - Streamlit GUI.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.interpolate import RegularGridInterpolator

from ttr_sim import DEPTH_PRESETS, KVOID_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec
from ttr_sim.analytic import center_step_response
from ttr_sim.materials import (
    AIR, COPPER, COPPER_ABSORPTION_DEPTH, COPPER_C_TR_PROBE, COPPER_REFLECTIVITY_DEFAULT, FUSED_SILICA,
    PROBE_WAVELENGTH_NM, PUMP_WAVELENGTH_NM,
)
from ttr_sim.presets import fmt_length, fmt_time
from ttr_sim.solver import MAT_VOID, build_grid, material_map
from ttr_sim.solver3d import theta_faces
from ttr_sim.validation import analytic_comparison, grid_convergence, run_pair

st.set_page_config(page_title="Nanosecond Transient Thermoreflectance Simulator", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

UM = 1e-6

# Temperature colour scale: navy (coldest) → blue → cyan → green → yellow → orange → pure bright red (hottest).
TEMP_COLORSCALE = [
    [0.00, "#141466"], [0.14, "#0000ff"], [0.32, "#00c8ff"], [0.50, "#00e000"],
    [0.68, "#ffff00"], [0.85, "#ff8c00"], [1.00, "#ff0000"],
]


# Browser-side behaviour for the temperature-field figure.  FIELD_PARENT_JS is injected ONCE into the
# main page as a <script> (so it runs in the page's own realm and survives Streamlit re-renders and view
# switches); FIELD_TOOLBAR_HTML is a small same-origin iframe with zoom / snapshot buttons that call it.
FIELD_PARENT_JS = r"""
(function () {
  if (window.__ttrField) return;
  const isFieldGd = g => !!(g && g._fullLayout && g._fullLayout.meta && g._fullLayout.meta.tag === 'ttr-field' &&
      g._transitionData && g._transitionData._frames && g._transitionData._frames.length > 1);
  const findGd = () => Array.from(document.querySelectorAll('.js-plotly-plot')).find(isFieldGd) || null;
  const gdOf = el => { const g = (el && el.closest) ? el.closest('.js-plotly-plot') : null; return isFieldGd(g) ? g : null; };
  const anim = { mode: 'immediate', frame: { duration: 0, redraw: true }, transition: { duration: 0 } };

  const api = {
    findGd: findGd,
    step: function (dir) {
      const gd = findGd(); if (!gd) return;
      const n = gd._transitionData._frames.length;
      const sl = gd._fullLayout.sliders && gd._fullLayout.sliders[0];
      const cur = sl ? sl.active : 0;
      const k = Math.min(n - 1, Math.max(0, cur + dir));
      if (k === cur) return;
      Plotly.relayout(gd, { 'sliders[0].active': k });
      Plotly.animate(gd, [String(k)], anim);
    },
    reset: function () {
      const gd = findGd(); if (!gd) return;
      const m = gd._fullLayout.meta;
      Plotly.relayout(gd, { 'xaxis.range': [-m.R, m.R], 'yaxis.range': [m.L, 0] });
    },
    focus: function () {
      const gd = findGd(); if (!gd) return;
      if (!gd.hasAttribute('tabindex')) gd.setAttribute('tabindex', '0');
      gd.style.outline = 'none';
      gd.focus({ preventScroll: true });
      gd.style.boxShadow = '0 0 0 2px #d62728';
    }
  };
  window.__ttrField = api;

  // Plain wheel over the figure = page scroll; Ctrl+wheel = Plotly zoom (only ctrl-wheel reaches Plotly's
  // scrollZoom handler, which itself prevents the browser's page-zoom default).
  document.addEventListener('wheel', function (e) {
    if (gdOf(e.target) && !e.ctrlKey) e.stopImmediatePropagation();
  }, { capture: true, passive: false });

  // Clicking on the figure gives it keyboard focus (red outline) so the arrow keys go straight to it.
  document.addEventListener('mousedown', function (e) {
    if (gdOf(e.target)) setTimeout(api.focus, 0);
  }, true);
  document.addEventListener('focusout', function (e) {
    const gd = gdOf(e.target); if (gd && e.target === gd) gd.style.boxShadow = '';
  }, true);

  // Left/Right arrows step the snapshot while the field figure is on screen (unless typing in a field).
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    if (!gdOf(e.target)) {
      const tag = (e.target && e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || (e.target && e.target.isContentEditable)) return;
      if (!findGd()) return;
    }
    e.preventDefault(); e.stopImmediatePropagation();
    api.step(e.key === 'ArrowRight' ? 1 : -1);
  }, true);
})();
"""

FIELD_TOOLBAR_HTML = """
<script>
(function () {
  const W = window.parent, D = W.document;
  if (!W.__ttrField) {
    const s = D.createElement('script');
    s.textContent = __PARENT_CODE__;
    D.head.appendChild(s);
  }
})();
</script>
""".replace("__PARENT_CODE__", json.dumps(FIELD_PARENT_JS))


# ----------------------------------------------------------------------------- helpers
def time_unit(t_end: float) -> tuple[float, str]:
    """Time axis unit for all plots: always μs, so axes read the same across presets (17 ns .. ms windows)."""
    return 1e6, "μs"


def default_energy_nJ(tau_p: float) -> float:
    """Pulse energy giving roughly a 1 K peak surface rise (peak T ∝ E/√τp for a 1-D transient)."""
    e = 2.3 * math.sqrt(tau_p / 17.2e-9)
    mag = 10 ** math.floor(math.log10(e))
    return float(round(e / mag) * mag)


def preset_defaults(idx: int) -> dict:
    p = DEPTH_PRESETS[idx]
    d_um = p.d_rep * 1e6
    return dict(
        void_depth_um=d_um,
        void_thickness_um=max(round(0.5 * d_um, 3), round(2 * p.dz * 1e6, 3)),
        void_r_um=min(round(2 * d_um, 3), 40.0),
        energy_nJ=default_energy_nJ(p.tau_p),
        t_end_us=default_window_us(p, d_um),
    )


def on_preset_change():
    """A preset change re-fills the void geometry and pulse-energy defaults for the new depth."""
    for key, val in preset_defaults(st.session_state.preset_idx).items():
        st.session_state[key] = val


def default_window_us(p, depth_um: float) -> float:
    """Default observation window: covers the pulse (5 τp) and the void response (4 d²/D), 3 significant digits."""
    t = max(5.0 * p.tau_p, 4.0 * (depth_um * UM) ** 2 / COPPER.alpha) * 1e6
    mag = 10 ** math.floor(math.log10(t)) / 100
    return float(round(t / mag) * mag)


def init_state():
    if "preset_idx" not in st.session_state:
        st.session_state.preset_idx = 2
        for key, val in preset_defaults(2).items():
            st.session_state[key] = val
    st.session_state.setdefault("kvoid_idx", 0)     # default: real air value 0.026 W/(m K)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("conv", None)


def build_config() -> SimConfig:
    s = st.session_state
    p = DEPTH_PRESETS[s.preset_idx]
    k_void = KVOID_PRESETS[s.kvoid_idx].k
    void = VoidSpec(
        enabled=True, depth=s.void_depth_um * UM, thickness=s.void_thickness_um * UM,
        r_center=s.void_rc_um * UM, r_half=s.void_r_um * UM, k=k_void, rho_cp=AIR.rho_cp,
        shape="ellipse",                                  # fixed: spheroidal void (bubble-like)
    )
    laser = Laser(
        tau_p=p.tau_p, profile="gaussian", energy=s.energy_nJ * 1e-9, w=s.spot_um * UM,
        reflectivity=s.reflectivity, probe_w=s.probe_um * UM, c_tr=float(s.get("c_tr", COPPER_C_TR_PROBE)),
    )
    numerics = Numerics(
        dz=p.dz,                                              # surface-layer spacing sqrt(D tau_p)/10 from the preset
        fo=100.0 / float(s.get("steps_per_pulse", 200)),      # dz = sqrt(D tau_p)/10  =>  dt = tau_p/N  <=>  Fo = 100/N
        t_end_mode="absolute", t_end_abs=float(s.get("t_end_us", default_window_us(p, s.void_depth_um))) * 1e-6,
        dt_growth=1.0 + float(s.get("dt_growth_pct", 5.0)) / 100.0, n_snapshots=int(s.get("n_snapshots", 12)),   # stretch: fixed 1.15
    )
    geometry = Geometry()      # real Cu-in-silica geometry; the homogeneous-copper validation mode lives in the tests only
    return SimConfig(geometry=geometry, void=void, laser=laser, numerics=numerics)


def progress_ui(container):
    bar = container.progress(0.0, text="준비 중...")

    def cb(frac: float, label: str = ""):
        bar.progress(min(max(frac, 0.0), 1.0), text=f"{label}  {frac * 100:.0f}%")

    return bar, cb


def pulse_shading(fig: go.Figure, laser: Laser, tscale: float):
    if laser.profile == "square":
        x0, x1 = 0.0, laser.tau_p
    else:
        x0, x1 = laser.t_center - 0.5 * laser.tau_p, laser.t_center + 0.5 * laser.tau_p
    fig.add_vrect(x0=x0 * tscale, x1=x1 * tscale, fillcolor="orange", opacity=0.12, line_width=0,
                  annotation_text="펄스 FWHM", annotation_position="top left")


def add_pulse_trace(fig: go.Figure, res: "SimResult", tscale: float, y2_title: str = "펄스 파형 (피크 = 1)"):
    """Overlay the normalised laser power f(t)/max on a secondary axis so the reader sees exactly when
    energy is being deposited (a Gaussian starts well before its FWHM band)."""
    pk = float(np.max(res.pulse)) or 1.0
    fig.add_trace(go.Scatter(x=res.times * tscale, y=res.pulse / pk, name="펄스 파형", yaxis="y2",
                             line=dict(color="orange", width=1.2, dash="dot"), opacity=0.9))
    fig.update_layout(yaxis2=dict(title=y2_title, overlaying="y", side="right", range=[0, 1.08],
                                  showgrid=False, tickvals=[0, 0.5, 1]))


def fmt_pct(x: float) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}%"


def fmt_sci(x: float) -> str:
    return f"{x:.2e}"


# ----------------------------------------------------------------------------- sidebar
init_state()
# Selectboxes (control and the dropdown list, which Plotly-independent BaseWeb renders in a portal)
# use a monospace font so the padded preset columns line up; white-space: pre keeps the padding.
st.markdown(
    """
    <style>
      div[data-testid="stSelectbox"] div[data-baseweb="select"],
      div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
      div[data-baseweb="popover"] li[role="option"], div[data-baseweb="popover"] li[role="option"] *,
      ul[role="listbox"] li, ul[role="listbox"] li * {
        font-family: Consolas, "Courier New", monospace !important;
        white-space: pre !important;
        font-size: 0.9rem !important;
      }
      /* no sidebar collapse control: the settings panel is always shown */
      [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"],
      section[data-testid="stSidebar"] button[kind="header"],
      section[data-testid="stSidebar"] button[kind="headerNoPadding"] { display: none !important; }
      /* the (now empty) sidebar header reserved ~60px; drop it and tighten the top of the content */
      [data-testid="stSidebarHeader"] { display: none !important; }
      [data-testid="stSidebarUserContent"] { padding-top: 0.75rem !important; }
      section[data-testid="stSidebar"] h1 {
        padding: 0.4rem 0 0.9rem 0.2rem !important;
        font-size: 1.35rem !important; line-height: 1.3 !important; font-weight: 700 !important;
        letter-spacing: -0.01em;
      }
      /* wider sidebar and dropdown so the padded preset labels are not truncated */
      section[data-testid="stSidebar"] { width: 30rem !important; min-width: 30rem !important; }
      section[data-testid="stSidebar"] > div { width: 30rem !important; }
      div[data-baseweb="popover"] ul[role="listbox"] { min-width: 28rem !important; }

      /* ---- card-style settings panel ---- */
      section[data-testid="stSidebar"] { background: #f0f2f6 !important; }
      /* bordered st.container(...) -> white card (the bordered block sits directly under stLayoutWrapper) */
      section[data-testid="stSidebar"] [data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"] {
        background: #ffffff !important; border: 1px solid #e3e6ec !important; border-radius: 14px !important;
        padding: 1.0rem 1.1rem 0.9rem 1.1rem !important; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
      }
      /* editable controls -> light grey fields on the white cards */
      section[data-testid="stSidebar"] div[data-baseweb="input"],
      section[data-testid="stSidebar"] div[data-baseweb="base-input"],
      section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
      section[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"],
      section[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"] button {
        background: #f0f2f6 !important; border-color: #e3e6ec !important;
      }
      section[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"] { border-radius: 8px; overflow: hidden; }
      section[data-testid="stSidebar"] div[data-baseweb="select"] > div { border-radius: 8px !important; }
      section[data-testid="stSidebar"] [data-testid="stExpander"] details {
        background: #ffffff; border: 1px solid #e3e6ec !important; border-radius: 14px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
      }
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary { font-weight: 700; font-size: 1.05rem; padding: 0.85rem 1.1rem; }
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { color: #262730; }
      /* temperature map: the single reset icon sits top-right inside the plot, translucent until hovered */
      .js-plotly-plot .modebar { opacity: 0.45; transition: opacity 0.15s; }
      .js-plotly-plot .modebar:hover { opacity: 1; }
      .js-plotly-plot .modebar-btn svg { width: 22px !important; height: 22px !important; }
      .js-plotly-plot .modebar-btn[data-title="Fullscreen"], .js-plotly-plot .modebar-btn[data-title="Close fullscreen"] { display: none !important; }
      .card-h { font-weight: 700; font-size: 1.1rem; color: #262730; margin: 0 0 0.6rem 0; line-height: 1.3; }
      .card-h .num { color: #d62728; font-weight: 600; font-size: 0.85rem; margin-right: 0.85rem; vertical-align: 0.05rem; }
      .card-body { color: #444; padding-left: 1.9rem; font-size: 1.0rem; }
      .card-body.muted { color: #555; margin: -0.2rem 0 0.7rem 0; }
      section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
        border-radius: 12px; font-weight: 700; font-size: 1.1rem; padding: 0.7rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
def card_header(num: int, title: str):
    """Card heading: small red number badge + bold title (see the sidebar CSS)."""
    st.markdown(f'<div class="card-h"><span class="num">{num}</span>{title}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.title("Nanosecond Transient Thermoreflectance Simulator")

    # ---- card 1: depth preset (outside the form so its on_change refills the defaults immediately)
    with st.container(border=True):
        card_header(1, "Void 깊이 프리셋")
        st.selectbox(
            "목표 void 깊이 → 펄스폭 + 격자 (짝으로 결정)", options=list(range(len(DEPTH_PRESETS))),
            format_func=lambda i: DEPTH_PRESETS[i].display, key="preset_idx", on_change=on_preset_change,
            label_visibility="collapsed",
            help="τp = 2·d²/D_th 로 펄스폭이 정해지고, 표면층 격자 Δz = √(D_th·τp)/10 이 자동으로 따라옵니다.",
        )
        preset = DEPTH_PRESETS[st.session_state.preset_idx]
        if preset.warning:
            st.warning(preset.warning)

    # ---- card 2: k_void (fixed)
    with st.container(border=True):
        card_header(2, "k_void (void 열전도율)")
        st.session_state.kvoid_idx = 0                       # fixed: real air value (no user control)
        st.markdown(f'<div class="card-body">{KVOID_PRESETS[0].k:g} W/m·K (공기)</div>', unsafe_allow_html=True)

    # All numeric inputs live in a form: typed values are collected when the 적용 button is
    # clicked (or Enter is pressed inside a field), so they never depend on a keypress being
    # delivered to the widget (an IME can swallow Enter) and never trigger intermediate reruns.
    with st.form("settings", border=False):
        # ---- card 3: void geometry
        with st.container(border=True):
            card_header(3, "Void 형상")
            c1, c2 = st.columns(2)
            c1.number_input("깊이 (윗면 z) [μm]", min_value=0.0, max_value=500.0, step=0.01, key="void_depth_um", format="%.3f")
            c2.number_input("두께 [μm]", min_value=0.001, max_value=500.0, step=0.01, key="void_thickness_um", format="%.3f")
            c1.number_input("축에서 벗어난 거리 [μm]", min_value=0.0, max_value=40.0, step=0.01, key="void_rc_um", value=0.0, format="%.3f",
                            help="void 중심이 원기둥 축에서 벗어난 거리. 0 이면 축대칭 2D 계산, 0 보다 크면 void 하나가 축 밖에 있는 "
                                 "3차원(r, θ, z) 계산을 합니다 (펌프·프로브는 축 중심 고정). 방향은 결과에 영향이 없으므로 거리만 지정합니다.")
            c2.number_input("반경 반폭 [μm]", min_value=0.001, max_value=40.0, step=0.01, key="void_r_um", format="%.3f")

        # ---- card 4: laser
        with st.container(border=True):
            card_header(4, "레이저")
            st.markdown(
                f'<div class="card-body muted">펌프: {PUMP_WAVELENGTH_NM:g} nm 펄스, Gaussian (FWHM = τp) · '
                f'프로브: {PROBE_WAVELENGTH_NM:g} nm CW</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            c1.number_input("펄스 에너지 [nJ]", min_value=1e-3, max_value=1e7, key="energy_nJ", format="%.4g")
            c2.number_input("펌프 1/e² 반경 [μm]", min_value=1.0, max_value=40.0, value=10.0, step=1.0, key="spot_um")
            c1.number_input("프로브 1/e² 반경 [μm]", min_value=0.0, max_value=40.0, value=5.0, step=0.5, key="probe_um",
                            help="0이면 중심 셀 온도")
            c2.number_input(f"구리 반사율 R (펌프 {PUMP_WAVELENGTH_NM:g} nm)", min_value=0.0, max_value=0.99,
                            value=COPPER_REFLECTIVITY_DEFAULT, step=0.05, key="reflectivity",
                            help=f"펌프 {PUMP_WAVELENGTH_NM:g} nm 에서의 구리 반사율 (깨끗한 표면 약 0.6). 흡수 flux = I₀(1−R). "
                                 f"광학 흡수깊이 {COPPER_ABSORPTION_DEPTH * 1e9:.0f} nm 도 이 파장 기준입니다.")
            st.number_input(
                f"열반사 계수 (dR/dT)/R [1/K] (프로브 {PROBE_WAVELENGTH_NM:g} nm)", min_value=-1e-2, max_value=1e-2,
                value=COPPER_C_TR_PROBE, step=1e-5, key="c_tr", format="%.2e",
                help=f"프로브 {PROBE_WAVELENGTH_NM:g} nm 에서 구리의 열반사 계수. ΔR/R = (dR/dT)/R × ΔT 로 신호 탭의 ΔR/R 지표에만 쓰입니다 "
                     "(열 계산에는 영향 없음). 구리는 630 nm 근처에서 약 −1 ~ −2 ×10⁻⁴ /K 로 보고되며 표면 상태에 따라 수십 % 달라지므로, "
                     "절대값이 중요하면 기준 시료로 보정한 값을 넣으세요.",
            )

        # ---- advanced numerics (collapsed card)
        with st.expander("고급 수치 설정"):
            st.select_slider("펄스 중 시간 스텝 수 N (Δt = τp / N)", options=[50, 100, 200, 400, 800], value=200, key="steps_per_pulse",
                             help="펄스가 진행되는 동안의 시간 간격. N = 200 이면 Δt = τp/200 이고, 이는 표면층 격자 기준 Fourier 수 "
                                  "Fo = D·Δt/Δz² = 0.5 에 해당합니다. Crank–Nicolson은 무조건 안정이지만 N 이 작으면(Δt 가 크면) "
                                  "급격한 transient 에서 정확도가 떨어집니다. 펄스가 끝난 뒤의 Δt 는 아래 증가율로 커집니다.")
            st.number_input(
                "관측 시간창 [μs]", min_value=1e-3, max_value=1e6, step=1.0, key="t_end_us", format="%.4g",
                help="그래프의 시간축 끝. 프리셋을 바꾸면 그 프리셋의 대표 깊이에 맞는 기본값 max(5·τp, 4·d²/D) 이 다시 채워지고, "
                     "그 외에는 바꾸기 전까지 고정되므로 void 크기·위치·에너지 등을 바꿔도 x축이 같아 결과를 나란히 비교할 수 있습니다. "
                     "void 깊이를 바꿔 신호 피크(≈2·d²/D)가 창을 넘으면 아래에 경고가 표시됩니다.",
            )
            st.number_input(
                "펄스 종료 후 Δt 증가율 [% / 스텝]", min_value=0.0, max_value=50.0, value=5.0, step=1.0, key="dt_growth_pct", format="%.0f",
                help="펄스가 끝나면 온도 변화가 점점 느려지므로 시간 간격을 스텝마다 이 비율만큼 늘려 갑니다. "
                     "5 % 면 Δt 가 항상 '지금까지 흐른 시간의 약 5 %' 로 유지되어, 1 ms 관측창도 약 300 스텝이면 충분합니다. "
                     "0 % 면 펄스 중의 촘촘한 Δt 를 끝까지 유지합니다 (긴 관측창에서는 스텝 수가 수만~수백만으로 급증). "
                     "정확도를 더 높이려면 2~3 %, 계산을 빨리 하려면 10 % 정도가 적당합니다. (내부적으로 8스텝 블록마다 적용)",
            )
            st.slider("온도장 스냅샷 개수", min_value=12, max_value=100, value=12, step=1, key="n_snapshots",
                      help="관측 시간창을 균등 분할하여 저장하는 온도장 개수. 많을수록 온도장 탭의 시간 간격이 촘촘해집니다 (메모리 사용 증가).")

        # ---- read-only summary of what the current settings imply (all widgets above exist in this run)
        cfg_preview = build_config()
        d = cfg_preview.diagnostics()
        try:
            grid_preview = build_grid(cfg_preview)
            n_cells = grid_preview.n_cells
            grid_err = None
        except ValueError as e:
            n_cells, grid_err = None, str(e)

        la_p = cfg_preview.laser
        f_peak = 1.0 if la_p.profile == "square" else float(la_p.f(la_p.t_center))          # Gaussian peak ≈ 0.94
        P_inc = la_p.energy / la_p.tau_p * f_peak                                             # W, incident
        P_abs = P_inc * (1.0 - la_p.reflectivity)
        q_abs = la_p.I0 * (1.0 - la_p.reflectivity) * f_peak
        cu = cfg_preview.copper
        # lower estimate: semi-infinite copper, Gaussian spot (3-D spreading);  upper estimate: add the
        # 1-D rod term that appears once the heat is confined radially by the silica (flux spread over pi R^2)
        dT_lo = float(center_step_response(la_p.tau_p, q_abs, la_p.w, cu))
        q_rod = P_abs / (math.pi * cfg_preview.geometry.R_cu ** 2)
        dT_hi = dT_lo + 2.0 * q_rod * math.sqrt(cu.alpha * la_p.tau_p / math.pi) / cu.k
        P_fmt = lambda p: f"{p * 1e3:.3g} mW" if p < 1 else f"{p:.3g} W"                     # noqa: E731

        with st.expander("계산된 격자 · 시간 정보  (읽기 전용)"):
            v = cfg_preview.void
            blocks_all = v.shape == "box" and v.r_inner == 0 and v.r_outer >= cfg_preview.geometry.R_cu * (1 - 1e-9)
            st.caption(
                f"void 범위: z = {v.depth * 1e6:.3f} ~ {v.z_bottom * 1e6:.3f} μm, "
                f"r = {v.r_inner * 1e6:.3f} ~ {min(v.r_outer, cfg_preview.geometry.R_cu) * 1e6:.3f} μm, "
                f"k_void = {v.k:g} W/m·K" + ("  — 단면 전체를 가로막음 (우회로 없음)" if blocks_all else "")
            )
            st.markdown(
                f"**τp** = {fmt_time(d['tau_p'])}  ·  **격자** 표면층 Δz = {fmt_length(d['dz'])}, Δr = {fmt_length(d['dr'])}; "
                f"void 주변 Δz = {fmt_length(d['dz_void'])}, Δr = {fmt_length(d['dr_void'])} (void 치수의 1/{d['void_cells']})  \n"
                f"**Δt** = {fmt_time(d['dt'])} (τp/{round(100 / d['fo'])}, Fo={d['fo']:g})"
                + (f" → 펄스 후 스텝당 +{(d['dt_growth'] - 1) * 100:.0f} % 로 증가, 최대 {fmt_time(d['dt_max'])}" if d['dt_growth'] > 1 and np.isfinite(d['dt_max']) else "")
                + f"  ·  **스텝** = {d['n_steps']:,}  ·  **셀** = {n_cells if n_cells else '—'}  \n"
                f"**관측창** = {fmt_time(d['t_end'])}  ·  **√(D·t_end)** = {fmt_length(d['L_diff'])}  \n"
                f"**피크 파워** 입사 {P_fmt(P_inc)} · 흡수 {P_fmt(P_abs)} (= 에너지 / τp{'' if f_peak == 1 else ' × 0.94'})  \n"
                f"**예상 표면 피크 상승** ≈ {dT_lo:.3g} ~ {dT_hi:.3g} K (반무한 구리 해석해 ~ 로드 갇힘 상한)"
            )
            if cfg_preview.void.r_center > 0 and n_cells:
                n_th = len(theta_faces(cfg_preview)) - 1
                n_unknowns = n_cells * n_th
                st.info(
                    f"축에서 벗어난 void → void 케이스는 3차원 (r, θ, z) 계산: 방위각 셀 {n_th}개 (반원, void 각폭을 {cfg_preview.numerics.void_cells}등분), "
                    f"미지수 ≈ {n_unknowns:,} ({'직접 LU' if n_unknowns <= 60_000 else 'ILU + 반복법'}). "
                    f"baseline 은 축대칭 2D. 계산 시간은 2D 보다 수십~수백 배 걸릴 수 있습니다."
                )
            st.markdown(
                """
**유효한 피크 상승 범위** — 이 시뮬레이터는 **선형 모델**입니다. 구리 반사율 R, 열전도율 k, 열용량 ρc 를 초기 온도의 값으로
고정하므로 모든 온도 상승과 ΔR/R 은 펄스 에너지에 정확히 비례합니다. 그 가정이 성립하는 범위는 표면 피크 상승으로 판단합니다.

| 표면 피크 상승 | 판정 |
|---|---|
| 10 K 이내 | 유효. 흡수율 변화 0.2 %, 구리 물성 변화 1 % 미만으로 격자 오차보다 작습니다. |
| 10 ~ 50 K | 주의. k 는 약 0.02 %/K 감소, ρc 는 약 0.1 %/K 증가하여 절대값에 수 % 오차. void 유/무 차이(대비)는 양쪽에 같은 오차가 들어가 영향이 작습니다. |
| 50 K 초과 | 무효. 온도 의존 물성·흡수율을 넣은 비선형 모델이 필요합니다 (현재 미구현). |

펄스가 w²/(8D) ≈ 0.1 μs 보다 길면 표면 온도는 에너지가 아니라 **피크 파워**로 정해지는 정상 상태값에 수렴합니다
(스팟 10 μm, R = 0.6 기준 입사 100 mW 당 약 4 K). 수백 μs 펄스에 실험 파워를 그대로 넣으면 이 범위를 쉽게 넘을 수 있으니,
펄스 에너지 = 파워 × 펄스폭 으로 입력한 뒤 위의 예상값과 계산 후의 "피크 ΔT (baseline)" 지표를 확인하세요.
예상 범위의 아래값은 반무한 균질 구리 해석해(3차원 확산), 위값은 여기에 실리카에 갇힌 로드의 1차원 가열 항
2·(P_abs/πR²)·√(D·τp/π)/k 를 더한 상한입니다. 경고 판정은 상한값으로 합니다.
                """
            )

        # warnings stay visible outside the collapsed info card
        if dT_hi > 50:
            st.error(f"예상 피크 상승 최대 {dT_hi:.3g} K: 선형 모델의 유효 범위를 넘습니다. 펄스 에너지(파워)를 줄이세요.")
        elif dT_hi > 10:
            st.warning(f"예상 피크 상승 최대 {dT_hi:.3g} K: 물성·흡수율의 온도 의존성으로 절대값에 수 % 오차가 생길 수 있습니다.")
        for w in d["warnings"]:
            st.warning(w)
        if grid_err:
            st.error(grid_err)

        submitted = st.form_submit_button("▶ 적용", type="primary", use_container_width=True)
    apply = submitted and grid_err is None


# ----------------------------------------------------------------------------- run
if apply:
    cfg = build_config()
    box = st.container()
    bar, cb = progress_ui(box)
    try:
        base, void, sig = run_pair(cfg, progress=cb)
        ana = analytic_comparison(base)
        st.session_state.result = dict(cfg=cfg, base=base, void=void, sig=sig, ana=ana)
        st.session_state.conv = None
    except Exception as e:  # noqa: BLE001
        st.exception(e)
    finally:
        bar.empty()

res = st.session_state.result
if res is None:
    st.info("왼쪽 패널에서 프리셋과 void 형상을 선택한 뒤 **적용** 을 누르세요.")
    st.markdown(
        """
        **모델 요약**
        - 축대칭 (r, z) 열확산 `ρc ∂T/∂t = ∇·(k∇T)`, 유한체적 + Crank–Nicolson (SuperLU 1회 분해)
        - 레이저: Beer–Lambert 흡수를 표면 flux `-k ∂T/∂z = I₀(1-R) f(t) e^{-2r²/w²}` 로 처리 (Δz ≫ 흡수깊이 조건 확인)
        - 경계: 축 대칭, 구리/실리카 계면 온도·flux 연속, 후면(z=500 μm)·외곽 단열
        - 신호: 프로브 가중 표면 온도의 void 유/무 차이 ΔT(t) 와 상대 대비 ΔT/ΔT_baseline
        """
    )
    st.stop()

cfg: SimConfig = res["cfg"]
base, void, sig, ana = res["base"], res["void"], res["sig"], res["ana"]
tscale, tunit = time_unit(cfg.t_end)
T0 = cfg.numerics.T0
diag = void.diagnostics

# ----------------------------------------------------------------------------- headline metrics
m = st.columns(5)
m[0].metric("피크 ΔT (baseline)", f"{sig['peak_rise_base']:.3g} K",
            help="void 없는 시료의 프로브 가중 표면 온도 상승 최대값. 선형 모델 유효 범위(10 K / 50 K)와 손상 여부 판단에 사용.")
m[1].metric("void 신호 피크 (ΔT)", f"{sig['peak_dT']:+.3g} K", help="프로브 가중 표면 온도의 void − baseline 최대 차이")
rr_peak = cfg.laser.c_tr * sig["peak_dT"]
m[2].metric("void 신호 피크 (ΔR/R)", f"{rr_peak * 1e4:+.3g} ×10⁻⁴",
            help=f"위 온도 차이에 열반사 계수 (dR/dT)/R = {cfg.laser.c_tr:.2e} /K 를 곱한 값. 실험에서 baseline 대비 재야 하는 반사율 변화.")
m[3].metric("상대 대비", f"{sig['peak_contrast'] * 100:+.1f} %",
            help="ΔT/ΔT_baseline — baseline 온도 상승이 피크의 0.1 % 이상인 구간에서의 최대 상대 차이. 검출 난이도의 척도.")
m[4].metric("신호 피크 시각", f"{sig['t_peak'] * tscale:.3g} {tunit}",
            help=f"void 신호(ΔT)가 최대가 되는 시각. 참고: 2·d²/D ≈ {2 * cfg.void.depth ** 2 / cfg.copper.alpha * tscale:.3g} {tunit}")
for w in diag["warnings"]:
    st.warning(w)
if not cfg.void.enabled:
    st.info("void 가 비활성화되어 있어 신호 지표는 0 입니다 (baseline 만 계산).")

# A radio (not st.tabs) so the selected view survives the rerun triggered by the test buttons.
VIEWS = ["📈 신호", "🌡 온도장", "✅ 검증 지표", "🔬 Grid convergence"]
view = st.radio("보기", VIEWS, horizontal=True, key="view", label_visibility="collapsed")

# ----------------------------------------------------------------------------- signal tab
if view == VIEWS[0]:
    SIG_W = 900          # fixed pixel width so curve shapes do not stretch with the window size
    t = base.times * tscale

    def apply_x(fig_):
        """Fixed linear x-range (= observation window) so runs with different conditions line up."""
        fig_.update_xaxes(range=[0.0, cfg.t_end * tscale])
    # ---- thermoreflectance signal dR/R = c_tr * dT (probe-weighted)
    c_tr = cfg.laser.c_tr
    SCALE = 1e4                                           # plotted in units of 1e-4
    rr_base = c_tr * base.dT_probe * SCALE
    figr = go.Figure()
    # all three curves share ONE axis: the shaded band between baseline and void IS the difference curve
    figr.add_trace(go.Scatter(x=t, y=rr_base, name="baseline (void 없음)", line=dict(color="#1f77b4")))
    if cfg.void.enabled:
        rr_void = c_tr * void.dT_probe * SCALE
        rr_diff = c_tr * sig["dT"] * SCALE
        figr.add_trace(go.Scatter(x=void.times * tscale, y=rr_void, name=f"void (k_void={cfg.void.k:g})", line=dict(color="#d62728"),
                                  fill="tonexty", fillcolor="rgba(214,39,40,0.12)"))
        figr.add_trace(go.Scatter(x=void.times * tscale, y=rr_diff, name="void − baseline (같은 축)", line=dict(color="#2ca02c", dash="dot")))
    figr.add_hline(y=0.0, line=dict(color="gray", width=1))
    pulse_shading(figr, cfg.laser, tscale)
    figr.update_layout(
        title=f"열반사 신호 ΔR/R  (프로브 {PROBE_WAVELENGTH_NM:g} nm, (dR/dT)/R = {c_tr:.2e} /K)", xaxis_title=f"t [{tunit}]",
        yaxis_title="ΔR/R [×10⁻⁴]", height=420, legend=dict(orientation="h", y=-0.2),
    )
    apply_x(figr)
    figr.update_layout(width=SIG_W, height=420, autosize=False)
    st.plotly_chart(figr, use_container_width=False)
    i_pk = int(np.argmax(np.abs(rr_base)))
    txt = f"baseline 피크 ΔR/R = {rr_base[i_pk] / SCALE:.3e} ({rr_base[i_pk]:+.3g}×10⁻⁴)"
    if cfg.void.enabled:
        j_pk = int(np.argmax(np.abs(rr_diff)))
        txt += (f"  ·  void 에 의한 변화 피크 = {rr_diff[j_pk] / SCALE:.3e} ({rr_diff[j_pk]:+.3g}×10⁻⁴) "
                f"at t = {void.times[j_pk] * tscale:.3g} {tunit}")
    st.caption(txt + ". 열반사 계수의 부호에 따라 신호 부호가 뒤집힐 수 있으며, 크기는 ΔT 에 비례합니다 (선형 모델).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=base.dT_probe, name="baseline (void 없음)", line=dict(color="#1f77b4")))
    if cfg.void.enabled:
        fig.add_trace(go.Scatter(x=void.times * tscale, y=void.dT_probe, name=f"void (k_void={cfg.void.k:g})", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=t, y=ana["analytic"], name="해석해 (semi-infinite Cu)", line=dict(color="gray", dash="dash")))
    pulse_shading(fig, cfg.laser, tscale)
    fig.update_layout(
        title="프로브 가중 표면 온도 상승", xaxis_title=f"t [{tunit}]",
        yaxis_title="ΔT [K]", height=420, legend=dict(orientation="h", y=-0.2),
    )
    add_pulse_trace(fig, base, tscale)
    apply_x(fig)
    fig.update_layout(width=SIG_W, height=420, autosize=False)
    st.plotly_chart(fig, use_container_width=False)

    if cfg.void.enabled:
        c1, c2 = st.columns(2)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=void.times * tscale, y=sig["dT"], name="ΔT = void − baseline", line=dict(color="#d62728")))
        pulse_shading(fig2, cfg.laser, tscale)
        add_pulse_trace(fig2, void, tscale)
        apply_x(fig2)
        fig2.update_layout(title="void 신호 ΔT(t)", xaxis_title=f"t [{tunit}]", yaxis_title="ΔT [K]", height=360)
        fig2.update_layout(width=SIG_W // 2 - 10, height=360, autosize=False)
        c1.plotly_chart(fig2, use_container_width=False)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=void.times * tscale, y=sig["contrast"] * 100, name="상대 대비", line=dict(color="#2ca02c")))
        pulse_shading(fig3, cfg.laser, tscale)
        apply_x(fig3)
        fig3.update_layout(title="상대 대비 ΔT / ΔT_baseline", xaxis_title=f"t [{tunit}]", yaxis_title="[%]", height=360)
        fig3.update_layout(width=SIG_W // 2 - 10, height=360, autosize=False)
        c2.plotly_chart(fig3, use_container_width=False)
        tau_void = cfg.void.depth ** 2 / cfg.copper.alpha
        st.caption(
            f"참고: void 깊이 d = {cfg.void.depth * 1e6:.3g} μm 의 특징 시간 τ_void = d²/D = {fmt_time(tau_void)} "
            f"(펄스폭 τp = {fmt_time(cfg.laser.tau_p)}, τp/τ_void = {cfg.laser.tau_p / tau_void:.2g}). "
            f"신호 피크는 보통 τ_void 이후에 나타납니다."
        )

# ----------------------------------------------------------------------------- field tab
if view == VIEWS[1]:
    src_choice = "void 케이스"      # fixed: the field view always shows the void case (baseline / difference views removed)
    snaps = void.snapshots if cfg.void.enabled else base.snapshots
    is_diff = src_choice.startswith("차이")
    grid = void.grid if cfg.void.enabled else base.grid
    g_plot = base.grid if src_choice == "baseline" else grid

    # Display region: the whole copper cylinder, |x| <= 40 μm, 0 <= z <= 500 μm.  2-D results are
    # mirrored about the axis; 3-D results already hold the (x, z) plane through the off-axis void.
    R_cu_um, L_um = cfg.geometry.R_cu * 1e6, cfg.geometry.L * 1e6
    ir = int(np.argmin(np.abs(g_plot.r_faces - cfg.geometry.R_cu)))      # number of radial cells inside the copper
    x_um = np.concatenate([-g_plot.r_c[:ir][::-1], g_plot.r_c[:ir]]) * 1e6
    z_um = g_plot.z_c * 1e6
    void_3d = cfg.void.enabled and void.is_3d

    def mirror(f: np.ndarray) -> np.ndarray:
        f = f[:, :ir]
        return np.concatenate([f[:, ::-1], f], axis=1)

    def base_plane(i: int, on_grid) -> np.ndarray:
        """Mirrored baseline ΔT plane (nz, 2 ir), interpolated onto `on_grid` if it differs."""
        Tb_ = base.snapshots[i][1] - T0
        if on_grid is not base.grid:
            itp = RegularGridInterpolator((base.grid.z_c, base.grid.r_c), Tb_, bounds_error=False, fill_value=None)
            ZZ, RR = np.meshgrid(on_grid.z_c, on_grid.r_c, indexing="ij")
            Tb_ = itp(np.stack([ZZ.ravel(), RR.ravel()], axis=1)).reshape(len(on_grid.z_c), len(on_grid.r_c))
        return mirror(Tb_)

    def plane_at(i: int) -> np.ndarray:
        """Full-width ΔT plane (nz, 2 ir) for snapshot i of the selected quantity."""
        if src_choice == "baseline":
            return base_plane(i, base.grid)
        Tv = snaps[i][1] - T0
        nr_v = grid.nr
        vplane = Tv[:, nr_v - ir: nr_v + ir] if void_3d else mirror(Tv)   # 3-D snapshot = (nz, 2 nr) plane
        return vplane - base_plane(i, grid) if is_diff else vplane

    # All snapshots are embedded as Plotly animation frames: moving the in-figure slider swaps the frame
    # data in the browser only (no Streamlit rerun, no redraw of the figure, zoom/pan preserved).
    # Frames are thinned if the total payload would be too large.
    MAX_VALUES = 2_000_000
    per_frame = len(z_um) * len(x_um)
    step = max(1, int(math.ceil(len(snaps) * per_frame / MAX_VALUES)))
    frame_idx = sorted(set(range(0, len(snaps), step)) | {len(snaps) - 1})
    cache_key = ("field_frames", id(res), src_choice)
    if cache_key not in st.session_state:
        fields = []
        for i in frame_idx:
            fields.append(plane_at(i).astype(np.float32))
        gmax = max(max(float(np.max(np.abs(f))) for f in fields), 1e-30)
        st.session_state[cache_key] = (fields, gmax)
    fields, gmax = st.session_state[cache_key]
    frame_t = [snaps[i][0] for i in frame_idx]
    frame_max = [float(np.max(np.abs(f))) for f in fields]

    def frame_title(k: int) -> str:
        return f"t = {frame_t[k] * tscale:.3g} {tunit}   ·   이 스냅샷 최대 |ΔT| = {frame_max[k]:.4g} K"

    init = min(len(fields) - 1, len(fields) // 3)
    colorscale = "RdBu_r" if is_diff else TEMP_COLORSCALE
    heat_kw = dict(
        x=x_um, y=z_um, colorscale=colorscale, zmid=0.0 if is_diff else None,
        zmin=-gmax if is_diff else 0.0, zmax=gmax, zauto=False,
        colorbar=dict(title="ΔT [K]", thickness=12, len=0.9),
        hovertemplate="r=%{x:.2f} μm<br>z=%{y:.2f} μm<br>ΔT=%{z:.4g} K<extra></extra>",
    )
    frames = [
        go.Frame(name=str(k), data=[go.Heatmap(z=fields[k], **heat_kw)], layout=go.Layout(title_text=frame_title(k)))
        for k in range(len(fields))
    ]
    fig = go.Figure(data=[go.Heatmap(z=fields[init], **heat_kw)], frames=frames)

    if cfg.void.enabled and src_choice != "baseline":
        v = cfg.void
        for sgn in (1,):                      # single void; an off-axis void lies on the x > 0 side of the plane
            xa, xb = sgn * (v.r_center - v.r_half) * 1e6, sgn * (v.r_center + v.r_half) * 1e6
            fig.add_shape(type="circle" if v.shape == "ellipse" else "rect",
                          x0=min(xa, xb), x1=max(xa, xb), y0=v.depth * 1e6, y1=v.z_bottom * 1e6,
                          line=dict(color="magenta", width=2), fillcolor="rgba(0,0,0,0)")
    # Copper boundary markers (the heatmap covers |r| <= R_cu; the grey area outside is silica, not plotted).
    for xr in (-R_cu_um, R_cu_um):
        fig.add_vline(x=xr, line=dict(color="gray", dash="dot", width=1))

    anim_args = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
    slider_steps = [
        dict(method="animate", args=[[str(k)], anim_args], label=f"{frame_t[k] * tscale:.3g}")
        for k in range(len(fields))
    ]
    # Canvas: height fixed; width = 3x the width needed for the 1:1 full-cylinder view.
    # constrain="range": the plot area fills the whole canvas and the 1:1 aspect is kept by widening the
    # r-axis range instead of shrinking the plot area, so the tick labels stay tied to the map coordinates.
    # layout.meta tags the figure for the browser-side handlers in FIELD_JS.
    PLOT_H = 900
    base_w = int(PLOT_H * (2 * R_cu_um) / L_um) + 230
    fig.update_layout(
        title=dict(text=frame_title(init), font=dict(size=14)), meta=dict(tag="ttr-field", R=R_cu_um, L=L_um),
        xaxis=dict(title="r [μm]", range=[-R_cu_um, R_cu_um], constrain="range", zeroline=False),
        yaxis=dict(title="z (깊이) [μm]", range=[L_um, 0], scaleanchor="x", scaleratio=1, constrain="range"),
        dragmode="pan", uirevision="field", height=PLOT_H, width=3 * base_w,
        margin=dict(l=60, r=10, t=50, b=90), plot_bgcolor="#e8e8e8",
        sliders=[dict(
            active=init, steps=slider_steps, x=0.0, y=-0.02, len=1.0, pad=dict(t=40, b=0),
            currentvalue=dict(prefix="스냅샷 t = ", suffix=f" {tunit}", visible=True, xanchor="left"),
        )],
    )
    st.plotly_chart(
        fig, use_container_width=False,
        config={
            "scrollZoom": True, "displaylogo": False, "doubleClick": "reset",
            # always-visible modebar reduced to the single "reset axes" (전체 보기) icon, overlaid top-right
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "toImage"],
        },
    )
    components.html(FIELD_TOOLBAR_HTML, height=0)   # invisible: installs the Ctrl+wheel / arrow-key handlers (see FIELD_PARENT_JS)

# ----------------------------------------------------------------------------- validation tab
if view == VIEWS[2]:
    st.subheader("에너지 보존")
    c1, c2, c3 = st.columns(3)
    c1.metric("baseline: (E_stored − E_in)/E_in", fmt_sci(base.energy_error))
    c2.metric("void: (E_stored − E_in)/E_in", fmt_sci(void.energy_error) if cfg.void.enabled else "—")
    c3.metric("흡수 에너지 (구리 표면 내)", f"{diag['absorbed_energy'] * 1e9:.4g} nJ",
              help=f"입사 {cfg.laser.energy * 1e9:.4g} nJ × (1−R={1 - cfg.laser.reflectivity:.2f}) × 로드 내 비율 {diag['absorbed_fraction_in_rod']:.3f}")
    fe = go.Figure()
    fe.add_trace(go.Scatter(x=void.times * tscale, y=void.E_in * 1e9, name="입력(흡수) 에너지", line=dict(color="orange")))
    fe.add_trace(go.Scatter(x=void.times * tscale, y=void.E_stored * 1e9, name="저장 열에너지 Σρc·V·ΔT", line=dict(color="#1f77b4", dash="dot")))
    fe.update_layout(xaxis_title=f"t [{tunit}]", yaxis_title="E [nJ]", height=320, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fe, use_container_width=True)
    st.caption(
        "모든 경계가 단열(표면 flux 제외)이고 유한체적 이산화가 보존적이므로 이 오차는 행렬 조립 + 선형해 정밀도(~1e-13)를 반영합니다."
    )
    with st.expander("에너지 보존 지표 읽는 법", expanded=False):
        st.markdown(
            """
**무엇을 비교하나** 레이저가 구리 표면에 넣어준 에너지(흡수 flux의 시간 적분)와, 계산 영역 전체에 저장된 열에너지 Σρc·V·ΔT를
매 스텝 비교합니다. 표면 flux 말고는 모든 경계가 단열이므로 두 값은 이론적으로 정확히 같아야 합니다.

**정상 범위**

| 값 | 판정 |
|---|---|
| 1e-15 ~ 1e-11 | 정상. 부동소수 반올림 + LU 직접해 정밀도 수준. 격자·시간 간격과 무관하게 이 범위여야 합니다. |
| 1e-10 ~ 1e-8 | 주의. 셀 수가 수십만 개이거나 Δt 가 극단적으로 작거나 커서 행렬 조건수가 나빠진 경우. 결과는 대개 쓸 수 있지만 원인을 확인하세요. |
| 1e-6 이상, 또는 NaN/inf | 이상. 행렬 조립 오류, 특이행렬, 재질 물성 입력 오류(0 또는 음수), 격자 면이 겹치는 등 구조적 문제입니다. 결과를 신뢰하지 마세요. |

**이 지표가 잡지 못하는 것** 유한체적 + Crank–Nicolson 은 격자가 아무리 거칠고 Δt 가 아무리 커도 에너지를 정확히 보존합니다.
따라서 이 값이 작다고 해서 결과가 정확한 것은 아닙니다. 격자·시간 간격이 거칠어서 생기는 오차(이산화 오차)는
아래의 **해석해 비교** 와 **Grid convergence** 탭에서 확인해야 합니다. 이 지표는 "계산이 구조적으로 깨지지 않았다"는 것만 보증합니다.

**시간에 따른 그래프** 펄스 동안 입력 에너지 곡선이 올라가고 저장 열에너지가 그 위에 정확히 겹쳐야 합니다.
펄스가 끝난 뒤 두 곡선이 모두 수평이어야 하며, 저장 에너지가 서서히 줄어든다면 어딘가로 열이 새고 있다는 뜻입니다(현재 모델에서는 발생하지 않아야 함).
            """
        )

    st.subheader("사용된 수치 파라미터")
    rows = [
        ("펄스폭 τp", fmt_time(diag["tau_p"])),
        ("Δz (프리셋, void 영역)", fmt_length(diag["dz"])),
        ("Δz (표면 첫 셀)", fmt_length(void.grid.dz_c[0])),
        ("Δr (관심 영역)", fmt_length(diag["dr"])),
        ("Δt (펄스 중 / 최대)", f"{fmt_time(diag['dt'])} / {fmt_time(diag['dt_max'])}"),
        ("펄스 종료 후 Δt 증가율", f"+{(diag['dt_growth'] - 1) * 100:.0f} % / 스텝 (8스텝 블록 단위)"),
        ("펄스 중 스텝 수 N (Δt = τp/N) / Fo = D·Δt/Δz²", f"{round(100 / diag['fo'])} / {diag['fo']:g}"),
        ("스텝 수 / 행렬 재분해 횟수", f"{diag['n_steps']:,} / {diag.get('n_factorisations', 1)}"),
        ("계산 차원 (void 케이스)", f"3D (r, θ, z), nθ = {diag['n_theta']} (반원), 미지수 {diag['n_cells']:,}, {diag['solver']}"
         + (f", 평균 반복 {diag['mean_iterations']:.1f}회" if diag.get("mean_iterations") else "")
         if void.is_3d else "2D 축대칭 (r, z), 직접 LU"),
        ("void 신호 예상 피크 시각 ≈ 2·d²/D", fmt_time(2.0 * cfg.void.depth ** 2 / cfg.copper.alpha)),
        ("셀 수 (nr × nz)", f"{diag['n_cells']:,} ({diag['nr']} × {diag['nz']})"),
        ("관측 시간창 t_end", fmt_time(diag["t_end"])),
        ("열 침투 깊이 √(D·t_end)", fmt_length(diag["L_diff"])),
        ("실리카 침투 깊이 √(D_si·t_end)", fmt_length(diag["L_diff_silica"])),
        ("Δz / 광학 흡수깊이 (≥7 필요)", f"{diag['flux_ratio']:.1f}  (δ_abs = {COPPER_ABSORPTION_DEPTH * 1e9:.0f} nm)"),
        ("피크 흡수 flux I₀(1−R)", f"{diag['q_abs_peak']:.3g} W/m²"),
        ("피크 입사 fluence I₀·τp", f"{diag['fluence_peak']:.3g} J/m²"),
        ("계산 시간 (baseline + void)", f"{base.wall_time + void.wall_time:.1f} s"),
    ]
    st.table(pd.DataFrame(rows, columns=["항목", "값"]).set_index("항목"))

    st.subheader("해석해 비교 (void 없는 baseline)")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMS 상대 편차", f"{ana['rms_rel'] * 100:.2f} %")
    c2.metric("최대 상대 편차", f"{ana['max_rel'] * 100:.2f} %")
    c3.metric("피크 상대 편차", fmt_pct(ana["peak_rel"]))
    for n in ana["notes"]:
        st.info(n)
    fa = go.Figure()
    fa.add_trace(go.Scatter(x=base.times * tscale, y=ana["numeric"], name="수치해 (baseline)", line=dict(color="#1f77b4")))
    fa.add_trace(go.Scatter(x=base.times * tscale, y=ana["analytic"], name="해석해", line=dict(color="gray", dash="dash")))
    fa.add_trace(go.Scatter(x=base.times * tscale, y=ana["numeric"] - ana["analytic"], name="차이", line=dict(color="#d62728"), yaxis="y2"))
    fa.update_layout(xaxis_title=f"t [{tunit}]", yaxis_title="ΔT [K]", yaxis2=dict(title="차이 [K]", overlaying="y", side="right"),
                     height=360, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fa, use_container_width=True)
    st.caption(
        "해석해: semi-infinite 균질 구리, 표면 Gaussian flux (Carslaw & Jaeger 형태). "
        "ΔT(0,0,t) = q₀w/(k√(2π))·arctan(√(8Dt)/w) 의 step 응답을 펄스 형상으로 중첩/컨볼루션. "
        "수치해와 같은 프로브 가중치로 평가합니다."
    )
    with st.expander("해석해 비교 지표 읽는 법", expanded=False):
        st.markdown(
            """
**기대하는 모습** 두 곡선은 거의 겹쳐야 하고, 차이 곡선은 0 근처에서 작은 진동만 보여야 합니다.
격자를 2배 조밀하게 하면(Grid convergence 탭) 차이가 약 1/4 로 줄어드는 2차 수렴이 정상입니다.

**차이가 0으로 수렴하는 조건** 해석해는 "무한히 넓고 깊은 균질 구리"를 가정합니다. 따라서 차이가 0으로 수렴하는 것은
두 조건이 만족될 때뿐입니다. (1) 격자·Δt 를 조밀하게 할수록 → 이산화 오차 감소. (2) 열이 구리/실리카 계면(r = 40 μm)이나
후면(z = 500 μm)에 도달하지 않는 시간창 → 기하 조건이 같음. 열이 계면에 닿은 뒤에는 실제 물리가 해석해와 달라지므로
(실리카가 열을 가둬 표면이 더 뜨겁게 유지됨) 차이가 남는 것이 **맞는** 결과이고, 이때는 위에 안내 문구가 표시됩니다.
이산화 오차만 따로 분리한 검증(실리카를 구리로 바꾼 균질 구리 조건)은 자동 테스트(`pytest`)에 포함되어 있으며,
그 조건에서는 차이가 격자 조밀화에 따라 0으로 수렴함을 확인합니다.

**수치 기준**

| RMS 상대 편차 | 판정 |
|---|---|
| 0.5 % 미만 | 정상. 권장 Δz, Fo = 0.5 에서의 전형적인 값. |
| 0.5 ~ 3 % | 열이 계면·후면에 도달했는지 확인 (안내 문구). 균질 구리 모드에서도 이 수준이면 격자가 거친 것이니 Δz 를 줄여 보세요. |
| 3 % 이상 (균질 구리 모드) | 이산화 오차가 큽니다. Δz, Fo, Δt 성장률을 줄이고 Grid convergence 로 수렴을 확인하세요. |

**최대 상대 편차가 RMS 보다 훨씬 큰 경우** 사각 펄스의 켜짐/꺼짐 순간에는 온도 기울기가 불연속이라 그 한두 스텝에서 편차가 튑니다.
이는 시간 이산화의 고유한 현상이며 신호 해석에는 영향이 없습니다. 판정에는 RMS 를 쓰세요.

**피크 상대 편차** 표면 온도 피크의 높이 차이입니다. 펄스 동안 표면 셀 온도를 셀 면으로 외삽하는 보정이 들어가 있어 권장 격자에서 1 % 이내여야 합니다.
            """
        )

# ----------------------------------------------------------------------------- grid convergence tab
if view == VIEWS[3]:
    st.markdown(
        "같은 물리 조건에서 격자를 **2배 조밀**(Δz, Δr 절반, Fo 고정 → Δt 1/4)하게 재계산하고, "
        "추가로 **거친 격자**(Δ×2)와 **시간 간격 절반**(Fo/2) 케이스를 계산하여 수치 파라미터 민감도를 보여줍니다."
    )
    est = (base.wall_time + void.wall_time) * (16 + 1 / 16 + 2)
    c1, c2 = st.columns(2)
    inc_coarse = c1.checkbox("거친 격자 (Δ×2) 포함", value=True)
    inc_dt = c2.checkbox("시간 간격 절반 (Fo/2) 포함", value=True)
    st.caption(f"예상 소요 시간 ≈ {est:.0f} s (현재 계산 {base.wall_time + void.wall_time:.1f} s 기준)")
    if st.button("▶ Grid convergence 테스트 실행", disabled=not cfg.void.enabled):
        box = st.container()
        bar, cb = progress_ui(box)
        try:
            st.session_state.conv = grid_convergence(cfg, base, void, sig, factor=2.0, include_coarse=inc_coarse,
                                                     include_dt_half=inc_dt, progress=cb)
        except Exception as e:  # noqa: BLE001
            st.exception(e)
        finally:
            bar.empty()
    if not cfg.void.enabled:
        st.info("void 를 활성화한 뒤 실행하세요 (void 신호의 수렴을 평가합니다).")
    conv = st.session_state.conv
    if conv:
        rows = conv["rows"]
        # ---- one-line verdict: how much the current-grid void signal moves when the grid is refined 2x
        cur = conv["comparisons"].get("current", {})
        err_dT = cur.get("d_peak_dT", float("nan"))
        err_base = cur.get("d_peak_rise", float("nan"))
        if np.isfinite(err_dT):
            pct = abs(err_dT) * 100
            if pct <= 2:
                verdict, box = "격자 오차 무시 가능 — 현재 격자 결과를 그대로 사용해도 됩니다.", st.success
            elif pct <= 10:
                verdict, box = "신호 크기에 이 정도 불확실성이 있습니다. 절대값이 중요하면 조밀 격자 결과를 채택하세요.", st.warning
            else:
                verdict, box = "격자가 부족합니다 (void 가 너무 얇거나 작아 분해가 안 되는 조합). 결과를 신뢰하지 마세요.", st.error
            box(
                f"**현재 격자의 격자 오차 추정: void 신호 피크 {err_dT * 100:+.1f} %, baseline 피크 {err_base * 100:+.1f} %** "
                f"(격자를 2배 조밀하게 한 결과 대비). {verdict}"
            )
        order = ["coarse", "current", "fine", "dt_half"]
        table = []
        for key in order:
            if key not in rows:
                continue
            r = rows[key]
            cmp = conv["comparisons"].get(key)
            table.append({
                "케이스": r["label"], "Δz": fmt_length(r["dz"]), "Δr": fmt_length(r["dr"]), "Δt": fmt_time(r["dt"]),
                "셀": f"{r['n_cells']:,}", "스텝": f"{r['n_steps']:,}",
                "피크 ΔT_base [K]": f"{r['peak_rise']:.5g}", "피크 void ΔT [K]": f"{r['peak_dT']:+.5g}",
                "대비 [%]": f"{r['peak_contrast'] * 100:+.3f}",
                "vs 조밀: ΔT_base": fmt_pct(cmp["d_peak_rise"]) if cmp else "기준",
                "vs 조밀: void ΔT": fmt_pct(cmp["d_peak_dT"]) if cmp else "기준",
                "vs 조밀: 신호 RMS": fmt_pct(cmp["rms_dT_rel"]) if cmp else "기준",
                "에너지 오차": fmt_sci(r["energy_error"]), "시간 [s]": f"{r['wall']:.1f}",
            })
        st.dataframe(pd.DataFrame(table).set_index("케이스"), use_container_width=True)
        if conv["order"] is not None:
            st.metric("관측 수렴 차수 (void ΔT 피크, 3단계 Richardson)", f"{conv['order']:.2f}",
                      help="2차 정확 스킴이면 ≈2. 격자에 정렬되지 않은 void 경계나 Δt 오차가 지배하면 낮아질 수 있습니다.")
        fc = go.Figure()
        for key in order:
            if key in rows:
                r = rows[key]
                fc.add_trace(go.Scatter(x=r["t"] * tscale, y=r["dT"], name=r["label"]))
        pulse_shading(fc, cfg.laser, tscale)
        fc.update_layout(title="void 신호 ΔT(t) — 격자/시간간격 비교", xaxis_title=f"t [{tunit}]", yaxis_title="ΔT [K]", height=400,
                         legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fc, use_container_width=True)

# ----------------------------------------------------------------------------- k_void sensitivity tab
