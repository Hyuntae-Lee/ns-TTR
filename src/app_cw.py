"""CW transmission-mode GUI (docs/cw-transmission-spec.md): CW pump on the via bottom (z = 0), probes on both
faces, step response + frequency-domain lock-in, the k_eff deliverable (phase-based glass-shell TLM inversion)
and the relative void indicators, baseline-vs-void comparison.

Run:  streamlit run app_cw.py        (the reflection-mode GUI app.py is untouched and runs separately)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ttr_sim import VoidSpec
from ttr_sim import analysis as an
from ttr_sim.materials import COPPER, COPPER_C_TR_PROBE, FUSED_SILICA
from ttr_sim.presets import KVOID_PRESETS
from ttr_sim.solver import MAT_VOID, baseline_config, build_grid
from ttr_sim.solver3d import count_unknowns, needs_3d
from ttr_sim.transmission import (
    DEFAULT_POWER, DEFAULT_REFLECTIVITY, DEFAULT_WINDOW, PROBE_RADIUS, cw_config, frequency_response, lockin_dataset,
    penetration_depth, run_step_response, slab_far_face_step, slab_intercept,
)

st.set_page_config(page_title="CW 투과 열전도도 시뮬레이터", layout="wide")
UM = 1e-6
F_MAX_LIMIT = 5000.0            # decision 6: nothing above 5 kHz
TEMP_COLORSCALE = [
    [0.00, "#141466"], [0.14, "#0000ff"], [0.32, "#00c8ff"], [0.50, "#00e000"],
    [0.68, "#ffff00"], [0.85, "#ff8c00"], [1.00, "#ff0000"],
]
OUTER_OPTIONS = {"far": "far — 무한 유리 (기본: 이웃 비아를 고려하지 않음)",
                 "neighbor_sink": "이웃 비아 = 완전 싱크 (r = pitch − R_cu 등온, 하한)",
                 "neighbor_adiabatic": "이웃 비아 대칭 셀 (r = pitch/2 단열, 상한 — 비물리적 DC)"}
VIEWS = ["신호", "분석", "락인", "온도장", "검증"]
# legends go below the x-axis title (a legend inside the bottom margin overlaps "t [ms]")
LEGEND_BELOW = dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0.0)
MARGIN_BELOW = dict(l=50, r=20, t=50, b=110)
INDICATORS = ["step_intercept", "parker", "two_face"]
IND_LABEL = {"step_intercept": "스텝 절편 t_int", "parker": "Parker 반상승 t½", "two_face": "양면 온도차 R_th(t*)"}
# Widgets that exist only for some settings get their default as `value`, not as a session-state seed: a keyed
# number_input created later than its seed shows its minimum instead of the seed (Streamlit 1.50).
PITCH_DEFAULT_UM = 130.0


# ----------------------------------------------------------------------------- state / config
def init_state():
    d = dict(power_mW=DEFAULT_POWER * 1e3, refl=DEFAULT_REFLECTIVITY, spot_a_um=12.0, spot_b_um=10.0, spot_3d=False,
             dT_max=5.0, probe_d_um=2 * PROBE_RADIUS / UM, c_tr=COPPER_C_TR_PROBE, L_um=500.0, R_cu_um=40.0, outer="far",
             void_on=False, void_depth_um=245.0, void_thick_um=10.0, void_r_um=20.0, kvoid_idx=0,
             t_end_ms=DEFAULT_WINDOW * 1e3, dz_um=2.5, dt_growth=1.05, f_min=10.0, f_max=F_MAX_LIMIT, n_f=9,
             result=None, view="신호")
    for k, v in d.items():
        st.session_state.setdefault(k, v)


def pitch_um() -> float:
    return float(st.session_state.get("pitch_um", PITCH_DEFAULT_UM))


def spot_offset_um() -> float:
    return float(st.session_state.get("spot_offset_um", 0.0))


def spot_radius() -> float:
    s = st.session_state
    return math.sqrt(0.25 * s.spot_a_um * s.spot_b_um) * UM          # equivalent circular radius of the flat ellipse


def build_void() -> VoidSpec | None:
    s = st.session_state
    if not s.void_on:
        return None
    return VoidSpec(enabled=True, depth=s.void_depth_um * UM, thickness=s.void_thick_um * UM, r_half=s.void_r_um * UM,
                    k=KVOID_PRESETS[s.kvoid_idx].k)


def build_config():
    s = st.session_state
    spot3d = s.spot_3d and (abs(s.spot_a_um - s.spot_b_um) > 1e-9 or spot_offset_um() != 0.0)
    return cw_config(power=s.power_mW * 1e-3, reflectivity=s.refl, L=s.L_um * UM, R_cu=s.R_cu_um * UM, pump_radius=spot_radius(),
                     spot_axes=((s.spot_a_um * UM, s.spot_b_um * UM) if spot3d else None),
                     spot_offset=(spot_offset_um() * UM if spot3d else 0.0),
                     probe_w=0.5 * s.probe_d_um * UM, t_end=s.t_end_ms * 1e-3, void=build_void(), outer=s.outer,
                     pitch=(pitch_um() * UM if s.outer != "far" else None), f_min=(s.f_min if s.outer == "far" else None),
                     dz=s.dz_um * UM, dt_growth=s.dt_growth, c_tr=s.c_tr)


def freqs() -> np.ndarray:
    s = st.session_state
    return np.unique(np.round(np.logspace(math.log10(s.f_min), math.log10(min(s.f_max, F_MAX_LIMIT)), int(s.n_f)), 3))


def run_all():
    s = st.session_state
    cfg_void = build_config()
    has_void = cfg_void.void.enabled
    cfg_base = baseline_config(cfg_void) if has_void else cfg_void
    bar = st.progress(0.0, text="준비 중...")
    fr_list = freqs()
    out = {}
    stages = [("baseline", cfg_base)] + ([("void", cfg_void)] if has_void else [])
    for i, (name, cfg) in enumerate(stages):
        cb = lambda p, i=i, name=name: bar.progress((i + 0.8 * p) / len(stages), text=f"{name}: 스텝 응답 {p * 100:.0f}%" + (" (3-D)" if needs_3d(cfg) else ""))   # noqa: E731
        resp = run_step_response(cfg, progress=cb)
        bar.progress((i + 0.8) / len(stages), text=f"{name}: 주파수 영역 풀이")
        fr = frequency_response(cfg, fr_list)
        ds = lockin_dataset(resp, fr)
        out[name] = dict(cfg=cfg, resp=resp, fr=fr, ds=ds)
    bar.progress(1.0, text="완료")
    s.result = dict(runs=out, has_void=has_void, freqs=fr_list, A=math.pi * cfg_base.geometry.R_cu ** 2,
                    L=cfg_base.geometry.L, spot=cfg_base.laser.w, dT_max=s.dT_max)


def analyses(opts: dict) -> dict:
    r = st.session_state.result
    out = {}
    for name, run in r["runs"].items():
        resp, ds = run["resp"], run["ds"]
        out[name] = an.analyze(resp.t, resp.dT_pump, resp.dT_far, resp.P_abs, r["L"], r["A"], COPPER.rho_cp, spot_radius=r["spot"],
                               freqs=ds.freqs, H_far=ds.H_far, t_star=opts["t_star"], intercept_window=opts["window"],
                               shell=(run["cfg"].geometry.R_cu, FUSED_SILICA))
    return out


def deliverable_table(run: dict, L: float, A: float) -> dict:
    """k_eff of the phase-based glass-shell inversion on the three data routes."""
    ds, fr, resp = run["ds"], run["fr"], run["resp"]
    shell = (run["cfg"].geometry.R_cu, FUSED_SILICA)
    out = {"자동 (1/창 미만 직접, 이상 스텝 FFT)": an.invert_tlm(ds.freqs, ds.H_far, COPPER.rho_cp, L, A, shell=shell)["k_eff"],
           "락인 = 주파수 영역 직접 풀이": an.invert_tlm(fr.freqs, fr.H_far, COPPER.rho_cp, L, A, shell=shell)["k_eff"]}
    ds_step = lockin_dataset(resp, None, ds.freqs)
    sel = ds.freqs >= ds.f_switch
    if sel.sum() >= 1:
        out[f"스텝 과도응답 FFT (f ≥ {ds.f_switch:.3g} Hz)"] = an.invert_tlm(ds.freqs[sel], ds_step.H_far[sel], COPPER.rho_cp, L, A, shell=shell)["k_eff"]
    return out


# ----------------------------------------------------------------------------- sidebar
init_state()
with st.sidebar:
    st.title("CW 투과 열전도도 시뮬레이터")
    st.caption("바닥면 CW 펌프 → 윗면/바닥면 프로브. 납품값: 위상 기반 유리 껍질 TLM 역산의 k_eff. 반사형(ns 펄스) 모드는 app.py 에 그대로 있습니다.")
    with st.container(border=True):
        st.markdown("**1. CW 펌프 (532 nm, z = 0 바닥면)**")
        c1, c2 = st.columns(2)
        c1.number_input("출력 P [mW]", 0.01, 1000.0, key="power_mW", step=1.0, format="%.3g")
        c2.number_input("Cu 반사율 R", 0.0, 0.99, key="refl", step=0.05, format="%.2f", help="CMP 후 바닥면 반사율 0.6 (확정). 납품값(위상 역산)에는 들어가지 않습니다.")
        c1.number_input("플랫 스팟 장축 지름 [μm]", 1.0, 80.0, key="spot_a_um", step=1.0, format="%.3g")
        c2.number_input("플랫 스팟 단축 지름 [μm]", 1.0, 80.0, key="spot_b_um", step=1.0, format="%.3g")
        st.checkbox("타원 스팟·정렬 오차를 3-D 로 계산 (느림: 수 분)", key="spot_3d",
                    help="끄면 등가 원형 플랫 반경 √(ab) 로 축대칭 계산. 켜면 스텝 응답을 반원 3-D 솔버로 풀어 펌프면 신호의 스팟 형상 효과를 봅니다 "
                         "(반대면·납품값에는 거의 영향 없음). 주파수 영역 풀이는 등가 원형으로 유지됩니다.")
        if st.session_state.spot_3d:
            st.number_input("스팟 중심 정렬 오차 [μm] (x 방향)", 0.0, 40.0, value=spot_offset_um(), key="spot_offset_um", step=0.5, format="%.3g")
        c1.number_input("선형 한계 ΔT_max [K] (펌프면, 창 끝)", 0.1, 100.0, key="dT_max", step=1.0, format="%.3g",
                        help="관측창 끝의 펌프면 상승이 이 값 아래면 선형 영역으로 봅니다. 정상상태가 없어 ΔT 는 창 길이에 따라 커집니다.")
        a = spot_radius()
        P_abs = st.session_state.power_mW * 1e-3 * (1 - st.session_state.refl)
        q = P_abs / (math.pi * a ** 2)
        st.caption(f"등가 원형 플랫 반경 a = √(ab) = **{a / UM:.2f} μm**. 흡수 {P_abs * 1e3:.3g} mW, 출력 밀도 {q:.3g} W/m², "
                   f"스프레딩 저항 1/(4ka) ≈ {1 / (4 * COPPER.k * a):.0f} K/W → 초기 국소 상승 ≈ {P_abs / (4 * COPPER.k * a):.3g} K "
                   f"(창 끝 값과 최대 출력은 계산 후 상단에 표시)")
    with st.container(border=True):
        st.markdown("**2. 프로브 (632.8 nm, 양면 동일)**")
        c1, c2 = st.columns(2)
        c1.number_input("프로브 스팟 지름 [μm]", 1.0, 80.0, key="probe_d_um", step=1.0, format="%.3g", help="1/e² 지름, 가우시안 가중")
        c2.number_input("c_tr = (dR/dT)/R [1/K]", -1e-2, 1e-2, key="c_tr", step=1e-5, format="%.2e", help="ΔR/R 표시에만 쓰입니다 (납품값은 위상 기반이라 무관).")
    with st.container(border=True):
        st.markdown("**3. 시편 (떠 있는 기판: 상하면 공기 = 단열, 이웃 비아 미고려)**")
        c1, c2 = st.columns(2)
        c1.number_input("비아 길이 = 유리 두께 L [μm]", 10.0, 2000.0, key="L_um", step=10.0, format="%.4g")
        c2.number_input("비아 반경 R_cu [μm]", 5.0, 200.0, key="R_cu_um", step=5.0, format="%.4g")
        with st.expander("고급: 외곽 유리 경계 (기본 far)", expanded=False):
            st.selectbox("외곽 유리 경계", list(OUTER_OPTIONS), format_func=OUTER_OPTIONS.get, key="outer")
            if st.session_state.outer != "far":
                st.number_input("비아 피치 [μm]", 2 * st.session_state.R_cu_um + 1.0, 5000.0,
                                value=max(pitch_um(), 2 * st.session_state.R_cu_um + 1.0), key="pitch_um", step=10.0, format="%.4g")
            if st.session_state.outer == "neighbor_adiabatic":
                st.error("**비물리적 상한**: 닫힌 유리 셀에는 열이 빠질 곳이 없어 DC 온도가 한없이 올라갑니다 (20 ms 에 수 K). "
                         "저주파 위상이 무한 유리 모델에서 벗어나므로 **납품값 역산에 쓰지 마세요** — 이웃 지배 정도를 가늠하는 상한 참고용입니다.")
            elif st.session_state.outer == "neighbor_sink":
                st.warning("하한 참고용: 이웃 비아를 완전 싱크로 봅니다. 저주파(≲ 200 Hz) 위상이 무한 유리 모델에서 벗어나 납품값이 수 % 치우칩니다.")
    with st.container(border=True):
        st.markdown("**4. void (깊이는 펌프면 z = 0 기준)**")
        st.checkbox("void 포함 (기준 via 와 함께 계산)", key="void_on")
        c1, c2, c3 = st.columns(3)
        c1.number_input("깊이 [μm]", 0.0, 2000.0, key="void_depth_um", step=5.0, format="%.4g", disabled=not st.session_state.void_on)
        c2.number_input("두께 [μm]", 0.1, 500.0, key="void_thick_um", step=1.0, format="%.4g", disabled=not st.session_state.void_on)
        c3.number_input("반경 반폭 [μm]", 0.5, 200.0, key="void_r_um", step=1.0, format="%.4g", disabled=not st.session_state.void_on)
        st.selectbox("void 충전 k", list(range(len(KVOID_PRESETS))), format_func=lambda i: KVOID_PRESETS[i].label, key="kvoid_idx",
                     disabled=not st.session_state.void_on)
        st.caption("축 위 원판(타원 단면) void, 2-D 축대칭 (3-D 스팟을 켜면 3-D 로 풉니다).")
    with st.container(border=True):
        st.markdown("**5. 수치 / 락인 주파수**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("관측창 [ms]", 0.1, 2000.0, key="t_end_ms", step=5.0, format="%.4g", help="기본 20 ms ≈ 10·L²/α; 과도응답 지표에는 5–10 ms 면 충분")
        c2.number_input("격자 Δz [μm]", 0.5, 20.0, key="dz_um", step=0.5, format="%.3g")
        c3.number_input("Δt 성장률", 1.0, 1.2, key="dt_growth", step=0.01, format="%.2f")
        c1.number_input("락인 f 최소 [Hz]", 0.1, F_MAX_LIMIT, key="f_min", format="%.4g",
                        help="far 경계의 유리 껍질을 이 주파수의 침투 깊이 3배 이상으로 넓힙니다.")
        c2.number_input("락인 f 최대 [Hz] (≤ 5 kHz)", 0.1, F_MAX_LIMIT, key="f_max", format="%.4g")
        c3.number_input("주파수 개수", 1, 60, key="n_f", step=1)
    run_clicked = st.button("적용 (계산 실행)", type="primary", use_container_width=True)
    try:
        cfg_preview = build_config()
        d = cfg_preview.diagnostics()
        g = build_grid(cfg_preview)
        extra = ""
        if needs_3d(cfg_preview):
            n2, nth, nu = count_unknowns(cfg_preview)
            extra = f" · **3-D**: 방위각 {nth} 섹터, 미지수 {nu:,}"
        st.caption(f"격자 {g.n_cells:,} 셀 (nr {g.nr} × nz {g.nz}, 외곽 반경 {g.r_faces[-1] / UM:.0f} μm, "
                   f"f_min 침투 깊이 {penetration_depth(st.session_state.f_min) / UM:.0f} μm) · Δr {d['dr'] / UM:.2f} μm · "
                   f"시간 스텝 {d['n_steps']} · t_end {d['t_end'] * 1e3:.3g} ms{extra}")
        for w in d["warnings"]:
            st.warning(w)
    except (ValueError, NotImplementedError) as e:
        st.error(str(e))

if run_clicked:
    try:
        run_all()
    except (ValueError, NotImplementedError, RuntimeError) as e:
        st.error(str(e))

# ----------------------------------------------------------------------------- main
res = st.session_state.result
st.title("CW 투과 모드 결과")
if res is None:
    st.info("왼쪽에서 조건을 정하고 **적용** 을 누르세요. 기준 via (와 void 가 있으면 void via) 의 스텝 응답과 주파수 응답을 계산합니다.")
    st.markdown("""
**측정 구성** — CW 펌프가 비아 바닥면(z = 0)을 가열하고, 프로브는 윗면(z = L, 투과) / 바닥면(반사) / 양면의 온도 상승을 읽습니다.
시스템이 선형(LTI)이므로 스텝 응답 하나에서 임펄스 응답·임의 chopping·락인 진폭/위상이 모두 나오며, 락인 값은 주파수 영역 직접 풀이로 정확히 구해 교차 검증합니다.

**납품값** — 반대면 락인 **위상**만으로 유리 껍질 어드미턴스 TLM 을 역산한 절대 k_eff (반사율·c_tr 무관). 락인 데이터든 기록한 스텝 과도응답의 FFT 든 같은 역산에 넣습니다.

**void 지표** — 스텝 절편 t_int, Parker t½, 양면 온도차 R_th(t*): 같은 이웃의 기준 via 와의 차이만 씁니다. 절대값은 유리로 새는 열 때문에 편향됩니다(분석 탭에 표시).
""")
    st.stop()

runs, has_void = res["runs"], res["has_void"]
base = runs["baseline"]
rb, cfg_b = base["resp"], base["cfg"]
ts = 1e3  # ms
P_max = rb.P_in * res["dT_max"] / rb.dT_pump[-1]
# ---- header metrics
c = st.columns(6)
c[0].metric("흡수 출력", f"{rb.P_abs * 1e3:.3g} mW", f"입사 {rb.P_in * 1e3:.3g} mW")
c[1].metric("펌프면 ΔT (창 끝)", f"{rb.dT_pump[-1]:.4g} K", f"한계 {res['dT_max']:.3g} K", delta_color="off")
c[2].metric("선형 한계 최대 출력", f"{P_max * 1e3:.3g} mW", f"{'초과' if rb.P_in > P_max else '여유'} ×{P_max / rb.P_in:.2f}",
            delta_color="inverse" if rb.P_in > P_max else "normal")
c[3].metric("반대면 ΔT (창 끝)", f"{rb.dT_far[-1]:.4g} K", f"양면 차 {rb.dT_diff[-1]:.4g} K", delta_color="off")
c[4].metric("에너지 보존 오차", f"{rb.energy_error:.1e}")
c[5].metric("격자 / 스텝 / 시간", f"{rb.result.n_cells:,} / {rb.result.n_steps}", f"{rb.result.wall_time:.1f} s + {base['fr'].wall_time:.1f} s"
            + (" (3-D)" if rb.result.is_3d else ""))
if rb.P_in > P_max:
    st.error(f"설정 출력 {rb.P_in * 1e3:.3g} mW 에서 창 끝({rb.t[-1] * ts:.3g} ms) 펌프면 상승 {rb.dT_pump[-1]:.3g} K 가 선형 한계 ΔT_max = {res['dT_max']:.3g} K 를 넘습니다. "
             f"최대 {P_max * 1e3:.3g} mW 까지 낮추세요 (ΔT ∝ P).")
for w in rb.warnings + base["fr"].warnings:
    st.warning(w)
if has_void:
    rv = runs["void"]["resp"]
    v = runs["void"]["cfg"].void
    st.caption(f"void: 깊이 {v.depth / UM:.4g}–{v.z_bottom / UM:.4g} μm, 반경 반폭 {v.r_half / UM:.3g} μm (단면 {min(1.0, (v.r_half / cfg_b.geometry.R_cu) ** 2) * 100:.0f} %), "
               f"k = {v.k} W/mK · void via: 펌프면 {rv.dT_pump[-1]:.4g} K, 반대면 {rv.dT_far[-1]:.4g} K, 에너지 오차 {rv.energy_error:.1e}")

st.radio("보기", VIEWS, horizontal=True, key="view", label_visibility="collapsed")
view = st.session_state.view


def logx(fig):
    fig.update_xaxes(type="log", range=[math.log10(max(rb.t[1] * ts, 1e-5)), math.log10(rb.t[-1] * ts)])


if view == "신호":
    use_log = st.checkbox("시간축 로그", value=True)
    c1, c2 = st.columns(2)
    f1 = go.Figure()
    for name, run in runs.items():
        r = run["resp"]
        dash = "solid" if name == "baseline" else "dash"
        f1.add_trace(go.Scatter(x=r.t * ts, y=r.dT_pump, name=f"펌프면 z=0 ({name})", line=dict(color="#d62728", dash=dash)))
        f1.add_trace(go.Scatter(x=r.t * ts, y=r.dT_far, name=f"반대면 z=L ({name})", line=dict(color="#1f77b4", dash=dash)))
    f1.add_hline(y=res["dT_max"], line=dict(color="gray", dash="dot"), annotation_text="선형 한계 ΔT_max")
    f1.update_layout(title="스텝 응답 ΔT(t)", xaxis_title="t [ms]", yaxis_title="ΔT [K]", height=420, legend=LEGEND_BELOW, margin=MARGIN_BELOW)
    if use_log:
        logx(f1)
    c1.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure()
    for name, run in runs.items():
        r = run["resp"]
        dash = "solid" if name == "baseline" else "dash"
        f2.add_trace(go.Scatter(x=r.t * ts, y=r.dRR_pump * 1e6, name=f"펌프면 ({name})", line=dict(color="#d62728", dash=dash)))
        f2.add_trace(go.Scatter(x=r.t * ts, y=r.dRR_far * 1e6, name=f"반대면 ({name})", line=dict(color="#1f77b4", dash=dash)))
    f2.update_layout(title=f"열반사 신호 ΔR/R (c_tr = {cfg_b.laser.c_tr:.2e} /K)", xaxis_title="t [ms]", yaxis_title="ΔR/R [ppm]", height=420,
                     legend=LEGEND_BELOW, margin=MARGIN_BELOW)
    if use_log:
        logx(f2)
    c2.plotly_chart(f2, use_container_width=True)
    if has_void:
        rv = runs["void"]["resp"]
        f3 = go.Figure()
        for key, lab, col in (("dT_pump", "펌프면", "#d62728"), ("dT_far", "반대면", "#1f77b4"), ("dT_diff", "양면 차", "#2ca02c")):
            b, vv = getattr(rb, key), getattr(rv, key)
            f3.add_trace(go.Scatter(x=rb.t * ts, y=np.where(np.abs(b) > 1e-9 * np.abs(b).max(), (vv - b) / np.where(b == 0, 1, b) * 100, np.nan),
                                    name=f"{lab} 상대 변화", line=dict(color=col)))
        f3.update_layout(title="void 신호: (void − 기준) / 기준 [%]", xaxis_title="t [ms]", yaxis_title="상대 변화 [%]", height=380, legend=LEGEND_BELOW, margin=MARGIN_BELOW)
        if use_log:
            logx(f3)
        st.plotly_chart(f3, use_container_width=True)

elif view == "분석":
    st.subheader("납품값: k_eff (위상 기반 유리 껍질 TLM 역산)")
    rows = []
    for name, run in runs.items():
        for route, k in deliverable_table(run, res["L"], res["A"]).items():
            rows.append({"케이스": name, "데이터 경로": route, "k_eff [W/mK]": k, "참값 대비 (모델 오차) [%]": (k / COPPER.k - 1) * 100})
    df = pd.DataFrame(rows)
    if has_void:
        kb = df[(df["케이스"] == "baseline")].set_index("데이터 경로")["k_eff [W/mK]"]
        df["Δk/k (void − 기준) [%]"] = [((r["k_eff [W/mK]"] / kb[r["데이터 경로"]] - 1) * 100) if r["케이스"] == "void" else np.nan for _, r in df.iterrows()]
    st.dataframe(df.style.format({c: "{:.1f}" for c in df.columns if c not in ("케이스", "데이터 경로")}), use_container_width=True, hide_index=True)
    ds = base["ds"]
    st.caption(f"위상만 사용 → 반사율·c_tr 무관. 주파수 {ds.freqs.min():g}–{ds.freqs.max():g} Hz ({len(ds.freqs)} 점), 자동 경로 전환 {ds.f_switch:.3g} Hz (= 1/관측창). "
               f"'참값 대비' 는 Cu 398 W/mK 와의 차이 = 유리 껍질 모델(실린더 주위 무한 유리) 의 모델 오차; 위상 잡음 요구 사양은 results/cw_phase_noise 참조.")
    st.subheader("void 지표 (기준 via 와의 상대 변화만 사용)")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        method = c1.selectbox("지표", INDICATORS, format_func=IND_LABEL.get)
        t_star = c2.number_input("양면차 t* [ms]", 0.0, float(rb.t[-1] * ts), float(rb.t[-1] * ts), step=0.5, format="%.3g") * 1e-3
        manual = c3.checkbox("절편법 직선 피팅 구간 지정", value=False)
        window = None
        if manual:
            w0, w1 = st.slider("절편법 피팅 구간 [ms]", 0.0, float(rb.t[-1] * ts), (0.5, min(1.5, float(rb.t[-1] * ts))), step=0.05)
            window = (w0 * 1e-3, w1 * 1e-3)
    A_res = analyses(dict(t_star=t_star, window=window))
    rows = []
    for m in INDICATORS:
        key = an.INDICATOR_KEYS[m]
        unit = {"t_int": "ms", "t_half": "ms", "R_th": "K/W"}[key]
        scale = 1e3 if unit == "ms" else 1.0
        row = {"지표": IND_LABEL[m], "값": f"{key} [{unit}]", "기준 via": A_res["baseline"][m][key] * scale}
        if has_void:
            row["void via"] = A_res["void"][m][key] * scale
            row["Δ (void − 기준) [%]"] = (A_res["void"][m][key] / A_res["baseline"][m][key] - 1) * 100
        if "k_eff" in A_res["baseline"][m]:
            row["절대값 편향: 겉보기 α·ρc [W/mK] (참값 398)"] = A_res["baseline"][m]["k_eff"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in rows[0] if c not in ("지표", "값")}), use_container_width=True, hide_index=True)
    st.caption("편향 표시: 스텝 절편·Parker 의 겉보기 α 는 유리로 새는 열 때문에 참값보다 **약 +30 % / +19 %** 높게 나옵니다 (far 경계, 스터디 results/cw_lateral_loss). "
               "양면 온도차는 k_eff 를 내지 않습니다: 떠 있는 via 에는 정상상태가 없고, 이 지표의 감도는 void 평면의 flux ∝ 1 − z/L 로 펌프면에서 반대면으로 "
               "선형으로 줄어 **반대면 쪽 void 는 거의 보이지 않습니다** (과도응답 지표에는 이 한계가 없음).")
    fig = go.Figure()
    for name, run in runs.items():
        r, a_ = run["resp"], A_res[name][method]
        dash = "solid" if name == "baseline" else "dash"
        if method == "step_intercept":
            fig.add_trace(go.Scatter(x=r.t * ts, y=r.dT_far, name=f"ΔT_far ({name})", line=dict(color="#1f77b4", dash=dash)))
            tt = np.array([a_["t_int"], r.t[-1]])
            fig.add_trace(go.Scatter(x=tt * ts, y=a_["slope"] * tt + a_["offset"], name=f"점근선 ({name}), t_int = {a_['t_int'] * ts:.3g} ms",
                                     line=dict(color="#ff7f0e", dash=dash)))
            fig.update_layout(xaxis_title="t [ms]", yaxis_title="ΔT [K]", title="스텝 절편: 반대면 곡선과 최대 기울기 접선")
        elif method == "parker":
            fig.add_trace(go.Scatter(x=r.t * ts, y=a_["h"], name=f"임펄스 응답 dΔT_far/dt ({name})", line=dict(color="#1f77b4", dash=dash)))
            fig.add_vline(x=a_["t_half"] * ts, line=dict(color="#ff7f0e", dash=dash), annotation_text=f"t½ = {a_['t_half'] * ts:.3g} ms ({name})")
            fig.update_layout(xaxis_title="t [ms]", yaxis_title="h [K/s]", title="Parker: 임펄스 응답과 반상승 시각", xaxis_type="log")
        else:
            fig.add_trace(go.Scatter(x=r.t * ts, y=a_["R_th_curve"], name=f"R_th(t) ({name})", line=dict(color="#2ca02c", dash=dash)))
            fig.add_hline(y=a_["R_through"], line=dict(color="gray", dash="dot"), annotation_text="통과 흐름 P·L/(kA) (열이 반대면으로 빠질 때)")
            fig.add_hline(y=a_["R_quasi_steady"] + a_.get("R_spread", 0.0), line=dict(color="gray", dash="dash"),
                          annotation_text="단열 준정상 P·L/(2kA) + 스프레딩 1/(4ka)")
            fig.add_vline(x=t_star * ts, line=dict(color="#2ca02c", dash="dot"), annotation_text="t*")
            fig.update_layout(xaxis_title="t [ms]", yaxis_title="R_th [K/W]", title="양면 온도차 R_th(t) = (ΔT_pump − ΔT_far)/P_abs", xaxis_type="log")
    fig.update_layout(height=420, legend=LEGEND_BELOW, margin=MARGIN_BELOW)
    st.plotly_chart(fig, use_container_width=True)

elif view == "락인":
    chop = st.radio("변조", ["square", "sine"], horizontal=True, format_func=lambda x: "사각 chopping (기본파 2P/π)" if x == "square" else "정현 변조 (P/2)")
    c1, c2 = st.columns(2)
    fa, fp = go.Figure(), go.Figure()
    for name, run in runs.items():
        fr, ds = run["fr"], run["ds"]
        dash = "solid" if name == "baseline" else "dash"
        for H, lab, col in ((fr.H_pump, "펌프면", "#d62728"), (fr.H_far, "반대면", "#1f77b4")):
            lk = an.lockin(H, fr.P_abs, chop)
            fa.add_trace(go.Scatter(x=fr.freqs, y=lk["amplitude"] * 1e3, name=f"{lab} 직접 풀이 ({name})", line=dict(color=col, dash=dash)))
            fp.add_trace(go.Scatter(x=fr.freqs, y=np.degrees(np.unwrap(np.angle(lk["complex"]))), name=f"{lab} 직접 풀이 ({name})", line=dict(color=col, dash=dash)))
        ds_step = lockin_dataset(run["resp"], None, ds.freqs)
        for H, lab, col in ((ds_step.H_pump, "펌프면", "#d62728"), (ds_step.H_far, "반대면", "#1f77b4")):
            lk = an.lockin(H, ds.P_abs, chop)
            fa.add_trace(go.Scatter(x=ds.freqs, y=lk["amplitude"] * 1e3, name=f"{lab} 스텝 FFT ({name})", mode="markers", marker=dict(color=col, symbol="x")))
            fp.add_trace(go.Scatter(x=ds.freqs, y=np.degrees(np.unwrap(np.angle(lk["complex"]))), name=f"{lab} 스텝 FFT ({name})", mode="markers",
                                    marker=dict(color=col, symbol="x")))
    fa.add_vline(x=base["ds"].f_switch, line=dict(color="gray", dash="dot"), annotation_text="1/창: 아래는 직접 풀이 사용")
    fp.add_vline(x=base["ds"].f_switch, line=dict(color="gray", dash="dot"))
    fa.update_layout(title="락인 진폭 (기본파)", xaxis_title="f [Hz]", yaxis_title="진폭 [mK]", xaxis_type="log", yaxis_type="log", height=440, legend=LEGEND_BELOW, margin=MARGIN_BELOW)
    fp.update_layout(title="락인 위상 (지연 음수, unwrap)", xaxis_title="f [Hz]", yaxis_title="위상 [°]", xaxis_type="log", height=440, legend=LEGEND_BELOW, margin=MARGIN_BELOW)
    c1.plotly_chart(fa, use_container_width=True)
    c2.plotly_chart(fp, use_container_width=True)
    tbl = []
    for i, f in enumerate(base["ds"].freqs):
        row = {"f [Hz]": f, "사용 경로": base["ds"].source[i]}
        for name, run in runs.items():
            lk = an.lockin(run["ds"].H_far, run["ds"].P_abs, chop)
            row[f"반대면 진폭 [μK] ({name})"] = lk["amplitude"][i] * 1e6
            row[f"반대면 위상 [°] ({name})"] = lk["phase_deg"][i]
        if has_void:
            lb, lv = an.lockin(base["ds"].H_far, base["ds"].P_abs, chop), an.lockin(runs["void"]["ds"].H_far, runs["void"]["ds"].P_abs, chop)
            row["진폭 변화 [%]"] = (lv["amplitude"][i] / lb["amplitude"][i] - 1) * 100
            row["위상 변화 [°]"] = lv["phase_deg"][i] - lb["phase_deg"][i]
        tbl.append(row)
    st.dataframe(pd.DataFrame(tbl).style.format({c: "{:.4g}" for c in tbl[0] if c != "사용 경로"}), use_container_width=True, hide_index=True)
    st.caption(f"'사용 경로': 관측창 {rb.t[-1] * ts:.3g} ms 가 분해하지 못하는 f < {base['ds'].f_switch:.3g} Hz 는 주파수 영역 직접 풀이, 그 위는 스텝 과도응답의 FFT. "
               "두 경로의 차이가 시간 적분기의 검증 항목입니다 (검증 탭).")

elif view == "온도장":
    name = st.radio("케이스", list(runs), horizontal=True) if has_void else "baseline"
    r = runs[name]["resp"].result
    snaps = r.snapshots
    i = st.slider("스냅샷", 0, len(snaps) - 1, len(snaps) - 1)
    t_i, T = snaps[i]
    rc, zc = r.grid.r_c / UM, r.grid.z_c / UM
    sel = r.grid.r_c <= r.config.geometry.R_cu
    if r.is_3d:                                               # snapshots already hold the (x, z) plane: x < 0 | x > 0
        nr = r.grid.nr
        keep = np.concatenate([sel[::-1], sel])
        x = np.concatenate([-rc[::-1], rc])[keep]
        Z = (T - r.config.numerics.T0)[:, keep]
        mv = r.material[:, keep]
    else:
        Tr = T[:, sel] - r.config.numerics.T0
        x = np.concatenate([-rc[sel][::-1], rc[sel]])
        Z = np.concatenate([Tr[:, ::-1], Tr], axis=1)
        mv = np.concatenate([r.material[:, sel][:, ::-1], r.material[:, sel]], axis=1)
    vmax = max(float(np.max([s[1].max() for s in snaps]) - r.config.numerics.T0), 1e-9)
    fig = go.Figure(go.Heatmap(x=x, y=zc, z=Z, colorscale=TEMP_COLORSCALE, zmin=0, zmax=vmax, colorbar=dict(title="ΔT [K]")))
    if r.config.void.enabled:
        fig.add_trace(go.Contour(x=x, y=zc, z=(mv == MAT_VOID).astype(float), showscale=False, contours=dict(start=0.5, end=0.5, coloring="none"),
                                 line=dict(color="white", width=1), hoverinfo="skip"))
    fig.update_layout(title=f"ΔT(x, z) @ t = {t_i * ts:.3g} ms ({name}{', 3-D θ=0 단면' if r.is_3d else ''}) — 펌프면 z = 0 (아래), 반대면 z = L (위)",
                      xaxis_title="x [μm]", yaxis_title="z [μm]", height=700, yaxis=dict(autorange="reversed"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("구리 단면(|x| ≤ R_cu)만 표시. 실리카는 계산에 포함되지만 표시 범위 밖.")

elif view == "검증":
    st.markdown("**에너지 보존** — 흡수 = 저장 (+ 이웃 싱크 방출)")
    rows = []
    for name, run in runs.items():
        r = run["resp"]
        rows.append({"케이스": name, "흡수 [μJ]": r.E_in[-1] * 1e6, "저장 [μJ]": r.E_stored[-1] * 1e6, "싱크 방출 [μJ]": r.E_out[-1] * 1e6,
                     "오차 (끝)": r.energy_error, "오차 (최대)": r.energy_error_max, "솔버": "3-D" if r.result.is_3d else "2-D 축대칭"})
    st.dataframe(pd.DataFrame(rows).style.format({"흡수 [μJ]": "{:.4g}", "저장 [μJ]": "{:.4g}", "싱크 방출 [μJ]": "{:.4g}", "오차 (끝)": "{:.1e}", "오차 (최대)": "{:.1e}"}),
                 use_container_width=True, hide_index=True)
    st.markdown("**주파수 영역 직접 풀이 vs 스텝 응답 FFT** (기준 via, 반대면) — 납품값 역산의 두 입력 경로")
    fr, ds = base["fr"], base["ds"]
    Hs = lockin_dataset(rb, None, ds.freqs).H_far
    shell = (cfg_b.geometry.R_cu, FUSED_SILICA)
    tbl = pd.DataFrame({"f [Hz]": fr.freqs, "|H| 직접 [K/W]": np.abs(fr.H_far), "|H| 스텝 FFT [K/W]": np.abs(Hs),
                        "진폭비": np.abs(Hs) / np.abs(fr.H_far), "위상차 [°]": np.degrees(np.angle(Hs / fr.H_far))})
    st.dataframe(tbl.style.format("{:.4g}"), use_container_width=True, hide_index=True)
    k_d = an.invert_tlm(fr.freqs, fr.H_far, COPPER.rho_cp, res["L"], res["A"], shell=shell)["k_eff"]
    sel = ds.freqs >= ds.f_switch
    k_s = an.invert_tlm(ds.freqs[sel], Hs[sel], COPPER.rho_cp, res["L"], res["A"], shell=shell)["k_eff"] if sel.any() else float("nan")
    st.caption(f"같은 역산의 k_eff: 직접 풀이 {k_d:.1f} W/mK, 스텝 FFT (f ≥ 1/창) {k_s:.1f} W/mK → 차이 {(k_s / k_d - 1) * 100:+.2f} % (요구 ±2 %).")
    st.markdown("**1-D 해석해 (Carslaw–Jaeger, 측면 단열 막대)** — 측면 손실이 없을 때의 반대면 곡선. 실제(유리 매립) 곡선과의 차이가 측면 손실의 크기입니다.")
    q = rb.P_abs / res["A"]
    ana = slab_far_face_step(rb.t, q, res["L"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rb.t * ts, y=rb.dT_far, name="반대면 계산 (기준)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=rb.t * ts, y=ana, name="1-D 단열 막대 해석해", line=dict(color="gray", dash="dot")))
    fig.update_layout(xaxis_title="t [ms]", yaxis_title="ΔT [K]", height=400, xaxis_type="log", legend=LEGEND_BELOW, margin=MARGIN_BELOW,
                      title=f"절편 L²/(6α) = {slab_intercept(res['L']) * ts:.3g} ms (Cu)")
    st.plotly_chart(fig, use_container_width=True)
    g = rb.result.grid
    st.caption(f"격자: nr = {g.nr}, nz = {g.nz}, 외곽 반경 {g.r_faces[-1] / UM:.0f} μm, 경계 {cfg_b.geometry.outer_bc}; "
               f"Δz 최대 {np.max(np.diff(g.z_faces)) / UM:.1f} μm; 시간 스텝 {rb.result.n_steps}, Δt 최대 {np.max(np.diff(rb.t)) * ts:.3g} ms.")
