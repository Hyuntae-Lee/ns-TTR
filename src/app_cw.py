"""CW transmission-mode GUI (docs/cw-transmission-spec.md): CW pump on the via bottom (z = 0), probes on both
faces, step response + frequency-domain lock-in, k_eff analysis methods, baseline-vs-void comparison.

Run:  streamlit run app_cw.py        (the reflection-mode GUI app.py is untouched and runs separately)
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ttr_sim import VoidSpec
from ttr_sim import analysis as an
from ttr_sim.materials import COPPER, COPPER_C_TR_PROBE, FUSED_SILICA
from ttr_sim.presets import KVOID_PRESETS
from ttr_sim.solver import MAT_VOID, baseline_config, build_grid
from ttr_sim.transmission import (
    DEFAULT_POWER, DEFAULT_REFLECTIVITY, DEFAULT_WINDOW, PROBE_RADIUS, cw_config, frequency_response, run_step_response,
    slab_far_face_step, slab_intercept,
)

st.set_page_config(page_title="CW 투과 열전도도 시뮬레이터", layout="wide")
UM = 1e-6
TEMP_COLORSCALE = [
    [0.00, "#141466"], [0.14, "#0000ff"], [0.32, "#00c8ff"], [0.50, "#00e000"],
    [0.68, "#ffff00"], [0.85, "#ff8c00"], [1.00, "#ff0000"],
]
OUTER_OPTIONS = {"far": "far — 무한 유리 (자동 반경, 단열)", "neighbor_sink": "이웃 비아 = 완전 싱크 (r = pitch − R_cu 등온, 하한)",
                 "neighbor_adiabatic": "이웃 비아 대칭 셀 (r = pitch/2 단열, 상한)"}
VIEWS = ["신호", "분석", "락인", "온도장", "검증"]


# ----------------------------------------------------------------------------- state / config
def init_state():
    d = dict(power_mW=DEFAULT_POWER * 1e3, refl=DEFAULT_REFLECTIVITY, spot_a_um=12.0, spot_b_um=10.0, probe_d_um=2 * PROBE_RADIUS / UM,
             c_tr=COPPER_C_TR_PROBE, L_um=500.0, R_cu_um=40.0, outer="far",
             void_on=False, void_depth_um=245.0, void_thick_um=10.0, void_r_um=20.0, kvoid_idx=0,
             t_end_ms=DEFAULT_WINDOW * 1e3, dz_um=2.5, dt_growth=1.05, f_min=10.0, f_max=5000.0, n_f=9,
             result=None, view="신호")
    for k, v in d.items():
        st.session_state.setdefault(k, v)


# Widgets that exist only for some settings (pitch, lattice) get their default as `value`, not as a session-state
# seed: a keyed number_input created later than its seed shows its minimum instead of the seed (Streamlit 1.50).
PITCH_DEFAULT_UM = 130.0


def pitch_um() -> float:
    return float(st.session_state.get("pitch_um", PITCH_DEFAULT_UM))


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
    return cw_config(power=s.power_mW * 1e-3, reflectivity=s.refl, L=s.L_um * UM, R_cu=s.R_cu_um * UM, pump_radius=spot_radius(),
                     probe_w=0.5 * s.probe_d_um * UM, t_end=s.t_end_ms * 1e-3, void=build_void(), outer=s.outer,
                     pitch=(pitch_um() * UM if s.outer != "far" else None), dz=s.dz_um * UM, dt_growth=s.dt_growth, c_tr=s.c_tr)


def freqs() -> np.ndarray:
    s = st.session_state
    return np.unique(np.round(np.logspace(math.log10(s.f_min), math.log10(s.f_max), int(s.n_f)), 3))


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
        cb = lambda p, i=i, name=name: bar.progress((i + 0.8 * p) / len(stages), text=f"{name}: 스텝 응답 {p * 100:.0f}%")   # noqa: E731
        resp = run_step_response(cfg, progress=cb)
        bar.progress((i + 0.8) / len(stages), text=f"{name}: 주파수 영역 풀이")
        fr = frequency_response(cfg, fr_list)
        out[name] = dict(cfg=cfg, resp=resp, fr=fr)
    bar.progress(1.0, text="완료")
    s.result = dict(runs=out, has_void=has_void, freqs=fr_list, A=math.pi * cfg_base.geometry.R_cu ** 2,
                    L=cfg_base.geometry.L, spot=cfg_base.laser.w)


def analyses(opts: dict) -> dict:
    """analysis.analyze on every run with the current method options."""
    r = st.session_state.result
    out = {}
    for name, run in r["runs"].items():
        resp, fr = run["resp"], run["fr"]
        out[name] = an.analyze(resp.t, resp.dT_pump, resp.dT_far, resp.P_abs, r["L"], r["A"], COPPER.rho_cp, spot_radius=r["spot"],
                               freqs=fr.freqs, H_far=fr.H_far, t_star=opts["t_star"], intercept_window=opts["window"],
                               fit_t_max=opts["fit_t_max"], shell=(run["cfg"].geometry.R_cu, FUSED_SILICA))
    return out


# ----------------------------------------------------------------------------- sidebar
init_state()
with st.sidebar:
    st.title("CW 투과 열전도도 시뮬레이터")
    st.caption("바닥면 CW 펌프 → 윗면/바닥면 프로브. 반사형(ns 펄스) 모드는 app.py 에 그대로 있습니다.")
    with st.container(border=True):
        st.markdown("**1. CW 펌프 (532 nm, z = 0 바닥면)**")
        c1, c2 = st.columns(2)
        c1.number_input("출력 P [mW]", 0.01, 1000.0, key="power_mW", step=1.0, format="%.3g")
        c2.number_input("Cu 반사율 R", 0.0, 0.99, key="refl", step=0.05, format="%.2f", help="CMP 후 바닥면 반사율, 미확정 (기본 0.6)")
        c1.number_input("플랫 스팟 장축 지름 [μm]", 1.0, 80.0, key="spot_a_um", step=1.0, format="%.3g")
        c2.number_input("플랫 스팟 단축 지름 [μm]", 1.0, 80.0, key="spot_b_um", step=1.0, format="%.3g")
        a = spot_radius()
        P_abs = st.session_state.power_mW * 1e-3 * (1 - st.session_state.refl)
        q = P_abs / (math.pi * a ** 2)
        st.caption(f"등가 원형 플랫 반경 a = √(ab) = **{a / UM:.2f} μm** (축대칭 계산). 흡수 {P_abs * 1e3:.3g} mW, "
                   f"출력 밀도 {q:.3g} W/m², 스프레딩 저항 1/(4ka) ≈ {1 / (4 * COPPER.k * a):.0f} K/W → 펌프면 국소 상승 ≈ **{P_abs / (4 * COPPER.k * a):.3g} K** "
                   f"(선형 영역 판단: ΔT ≪ 1 K 권장)")
    with st.container(border=True):
        st.markdown("**2. 프로브 (632.8 nm, 양면 동일)**")
        c1, c2 = st.columns(2)
        c1.number_input("프로브 스팟 지름 [μm]", 1.0, 80.0, key="probe_d_um", step=1.0, format="%.3g", help="1/e² 지름, 가우시안 가중")
        c2.number_input("c_tr = (dR/dT)/R [1/K]", -1e-2, 1e-2, key="c_tr", step=1e-5, format="%.2e")
    with st.container(border=True):
        st.markdown("**3. 시편 (떠 있는 기판: 상하면 공기 = 단열)**")
        c1, c2 = st.columns(2)
        c1.number_input("비아 길이 = 유리 두께 L [μm]", 10.0, 2000.0, key="L_um", step=10.0, format="%.4g")
        c2.number_input("비아 반경 R_cu [μm]", 5.0, 200.0, key="R_cu_um", step=5.0, format="%.4g")
        st.selectbox("외곽 유리 경계", list(OUTER_OPTIONS), format_func=OUTER_OPTIONS.get, key="outer",
                     help="DC/저주파 부분만 영향. 두 이웃 옵션 사이에 진실이 있고, 그 차이가 'DC 값이 배열에 얼마나 지배되는가'의 지표입니다.")
        if st.session_state.outer != "far":
            c1, c2 = st.columns(2)
            c1.number_input("비아 피치 [μm]", 2 * st.session_state.R_cu_um + 1.0, 5000.0, value=max(pitch_um(), 2 * st.session_state.R_cu_um + 1.0),
                            key="pitch_um", step=10.0, format="%.4g")
            c2.selectbox("배열", ["정방", "육방"], key="lattice", help="현재 경계 반경은 배열과 무관하게 pitch 로만 정해집니다 (스펙 §2.1). 배열은 기록용.")
    with st.container(border=True):
        st.markdown("**4. void (깊이는 펌프면 z = 0 기준)**")
        st.checkbox("void 포함 (기준 via 와 함께 계산)", key="void_on")
        c1, c2, c3 = st.columns(3)
        c1.number_input("깊이 [μm]", 0.0, 2000.0, key="void_depth_um", step=5.0, format="%.4g", disabled=not st.session_state.void_on)
        c2.number_input("두께 [μm]", 0.1, 500.0, key="void_thick_um", step=1.0, format="%.4g", disabled=not st.session_state.void_on)
        c3.number_input("반경 반폭 [μm]", 0.5, 200.0, key="void_r_um", step=1.0, format="%.4g", disabled=not st.session_state.void_on)
        st.selectbox("void 충전 k", list(range(len(KVOID_PRESETS))), format_func=lambda i: KVOID_PRESETS[i].label, key="kvoid_idx",
                     disabled=not st.session_state.void_on)
        st.caption("축 위 원판(타원 단면) void, 2-D 축대칭. 축 밖/다중 void 의 CW 모드는 3-D 솔버의 반대면 기록이 추가되면 열립니다.")
    with st.container(border=True):
        st.markdown("**5. 수치**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("관측창 [ms]", 0.1, 2000.0, key="t_end_ms", step=5.0, format="%.4g", help="기본 20 ms ≈ 10·L²/α")
        c2.number_input("격자 Δz [μm]", 0.5, 20.0, key="dz_um", step=0.5, format="%.3g")
        c3.number_input("Δt 성장률", 1.0, 1.2, key="dt_growth", step=0.01, format="%.2f")
        c1.number_input("락인 f 최소 [Hz]", 0.1, 1e5, key="f_min", format="%.4g")
        c2.number_input("락인 f 최대 [Hz]", 0.1, 1e6, key="f_max", format="%.4g")
        c3.number_input("주파수 개수", 2, 60, key="n_f", step=1)
    run_clicked = st.button("적용 (계산 실행)", type="primary", use_container_width=True)
    try:
        cfg_preview = build_config()
        d = cfg_preview.diagnostics()
        g = build_grid(cfg_preview)
        st.caption(f"격자 {g.n_cells:,} 셀 (nr {g.nr} × nz {g.nz}, 외곽 반경 {g.r_faces[-1] / UM:.0f} μm) · Δr {d['dr'] / UM:.2f} μm · "
                   f"시간 스텝 {d['n_steps']} · t_end {d['t_end'] * 1e3:.3g} ms")
        for w in d["warnings"]:
            st.warning(w)
    except (ValueError, NotImplementedError) as e:
        st.error(str(e))

if run_clicked:
    try:
        run_all()
    except (ValueError, NotImplementedError) as e:
        st.error(str(e))

# ----------------------------------------------------------------------------- main
res = st.session_state.result
st.title("CW 투과 모드 결과")
if res is None:
    st.info("왼쪽에서 조건을 정하고 **적용** 을 누르세요. 기준 via (와 void 가 있으면 void via) 의 스텝 응답과 주파수 응답을 계산합니다.")
    st.markdown("""
**측정 구성** — CW 펌프가 비아 바닥면(z = 0)을 가열하고, 프로브는 윗면(z = L, 투과) / 바닥면(반사) / 양면의 온도 상승을 읽습니다.
시스템이 선형(LTI)이므로 스텝 응답 하나에서 임펄스 응답·임의 chopping·락인 진폭/위상이 모두 나오며, 락인 값은 주파수 영역 직접 풀이로 정확히 구해 교차 검증합니다.

**분석 방법** (분석 탭) — 스텝 절편 L²/(6α), Parker 반상승, 양면 온도차 R_th, 1-D 핀 모델 피팅, 락인 TLM 역산 (단열 / 균일 손실 / 유리 껍질 모델).
떠 있는 기판에는 DC 평형이 없어(유리 속 2-D 확산의 log 발산) 방법마다 측면 손실 편향이 다르며, 그 불일치 자체가 void 지표가 됩니다.
""")
    st.stop()

runs, has_void = res["runs"], res["has_void"]
base = runs["baseline"]
rb, cfg_b = base["resp"], base["cfg"]
ts = 1e3  # ms
# ---- header metrics
c = st.columns(6)
c[0].metric("흡수 출력", f"{rb.P_abs * 1e3:.3g} mW", f"입사 {rb.P_in * 1e3:.3g} mW")
c[1].metric("펌프면 ΔT (t_end)", f"{rb.dT_pump[-1]:.4g} K")
c[2].metric("반대면 ΔT (t_end)", f"{rb.dT_far[-1]:.4g} K")
c[3].metric("양면 차 ΔT_pump − ΔT_far", f"{rb.dT_diff[-1]:.4g} K", f"R_th = {rb.dT_diff[-1] / rb.P_abs:.1f} K/W")
c[4].metric("에너지 보존 오차", f"{rb.energy_error:.1e}")
c[5].metric("격자 / 스텝 / 시간", f"{rb.result.n_cells:,} / {rb.result.n_steps}", f"{rb.result.wall_time:.1f} s + {base['fr'].wall_time:.1f} s")
for w in rb.warnings:
    st.warning(w)
if has_void:
    rv = runs["void"]["resp"]
    v = runs["void"]["cfg"].void
    st.caption(f"void: 깊이 {v.depth / UM:.4g}–{v.z_bottom / UM:.4g} μm, 반경 반폭 {v.r_half / UM:.3g} μm (단면 {min(1.0, (v.r_half / cfg_b.geometry.R_cu) ** 2) * 100:.0f} %), "
               f"k = {v.k} W/mK · void via: 펌프면 {rv.dT_pump[-1]:.4g} K, 반대면 {rv.dT_far[-1]:.4g} K, 에너지 오차 {rv.energy_error:.1e}")

st.radio("보기", VIEWS, horizontal=True, key="view", label_visibility="collapsed")
view = st.session_state.view


def signal_fig(key: str, title: str, ylab: str, scale: float = 1.0, logx: bool = True) -> go.Figure:
    fig = go.Figure()
    for name, run in runs.items():
        r = run["resp"]
        dash = "solid" if name == "baseline" else "dash"
        fig.add_trace(go.Scatter(x=r.t * ts, y=getattr(r, key) * scale, name=f"{name}", line=dict(dash=dash)))
    fig.update_layout(title=title, xaxis_title="t [ms]", yaxis_title=ylab, height=380, margin=dict(l=40, r=20, t=50, b=40),
                      legend=dict(orientation="h"))
    if logx:
        fig.update_xaxes(type="log", range=[math.log10(max(rb.t[1] * ts, 1e-5)), math.log10(rb.t[-1] * ts)])
    return fig


if view == "신호":
    logx = st.checkbox("시간축 로그", value=True)
    c1, c2 = st.columns(2)
    f1 = go.Figure()
    for name, run in runs.items():
        r = run["resp"]
        dash = "solid" if name == "baseline" else "dash"
        f1.add_trace(go.Scatter(x=r.t * ts, y=r.dT_pump, name=f"펌프면 z=0 ({name})", line=dict(color="#d62728", dash=dash)))
        f1.add_trace(go.Scatter(x=r.t * ts, y=r.dT_far, name=f"반대면 z=L ({name})", line=dict(color="#1f77b4", dash=dash)))
    f1.update_layout(title="스텝 응답 ΔT(t)", xaxis_title="t [ms]", yaxis_title="ΔT [K]", height=420, legend=dict(orientation="h"))
    if logx:
        f1.update_xaxes(type="log", range=[math.log10(max(rb.t[1] * ts, 1e-5)), math.log10(rb.t[-1] * ts)])
    c1.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure()
    for name, run in runs.items():
        r = run["resp"]
        dash = "solid" if name == "baseline" else "dash"
        f2.add_trace(go.Scatter(x=r.t * ts, y=r.dRR_pump * 1e6, name=f"펌프면 ({name})", line=dict(color="#d62728", dash=dash)))
        f2.add_trace(go.Scatter(x=r.t * ts, y=r.dRR_far * 1e6, name=f"반대면 ({name})", line=dict(color="#1f77b4", dash=dash)))
    f2.update_layout(title=f"열반사 신호 ΔR/R (c_tr = {cfg_b.laser.c_tr:.2e} /K)", xaxis_title="t [ms]", yaxis_title="ΔR/R [ppm]", height=420,
                     legend=dict(orientation="h"))
    if logx:
        f2.update_xaxes(type="log", range=[math.log10(max(rb.t[1] * ts, 1e-5)), math.log10(rb.t[-1] * ts)])
    c2.plotly_chart(f2, use_container_width=True)
    if has_void:
        rv = runs["void"]["resp"]
        f3 = go.Figure()
        for key, lab, col in (("dT_pump", "펌프면", "#d62728"), ("dT_far", "반대면", "#1f77b4"), ("dT_diff", "양면 차", "#2ca02c")):
            b, vv = getattr(rb, key), getattr(rv, key)
            f3.add_trace(go.Scatter(x=rb.t * ts, y=np.where(np.abs(b) > 1e-9 * np.abs(b).max(), (vv - b) / np.where(b == 0, 1, b) * 100, np.nan),
                                    name=f"{lab} 상대 변화", line=dict(color=col)))
        f3.update_layout(title="void 신호: (void − 기준) / 기준 [%]", xaxis_title="t [ms]", yaxis_title="상대 변화 [%]", height=380, legend=dict(orientation="h"))
        if logx:
            f3.update_xaxes(type="log", range=[math.log10(max(rb.t[1] * ts, 1e-5)), math.log10(rb.t[-1] * ts)])
        st.plotly_chart(f3, use_container_width=True)
        i = int(np.argmax(np.abs(rv.dT_far - rb.dT_far) / np.maximum(rb.dT_far, 1e-12) * (rb.dT_far > 1e-3 * rb.dT_far.max())))
        st.caption(f"반대면 최대 상대 변화 {((rv.dT_far[i] - rb.dT_far[i]) / rb.dT_far[i]) * 100:+.2f} % @ {rb.t[i] * ts:.3g} ms · "
                   f"t_end 에서 펌프면 {((rv.dT_pump[-1] - rb.dT_pump[-1]) / rb.dT_pump[-1]) * 100:+.2f} %, 반대면 {((rv.dT_far[-1] - rb.dT_far[-1]) / rb.dT_far[-1]) * 100:+.2f} %, "
                   f"양면 차 {((rv.dT_diff[-1] - rb.dT_diff[-1]) / rb.dT_diff[-1]) * 100:+.2f} %")

elif view == "분석":
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        method = c1.selectbox("분석 방법", list(an.METHOD_LABELS), format_func=an.METHOD_LABELS.get)
        t_star = c2.number_input("양면차 t* [ms]", 0.0, float(rb.t[-1] * ts), float(rb.t[-1] * ts), step=0.5, format="%.3g") * 1e-3
        fit_t_max = c3.number_input("핀 모델 피팅 구간 끝 [ms]", 0.01, float(rb.t[-1] * ts), min(3.0, float(rb.t[-1] * ts)), step=0.5, format="%.3g") * 1e-3
        manual = c4.checkbox("절편법 직선 피팅 구간 지정", value=False)
        window = None
        if manual:
            w0, w1 = st.slider("절편법 피팅 구간 [ms]", 0.0, float(rb.t[-1] * ts), (0.5, min(1.5, float(rb.t[-1] * ts))), step=0.05)
            window = (w0 * 1e-3, w1 * 1e-3)
    opts = dict(t_star=t_star, window=window, fit_t_max=fit_t_max)
    A_res = analyses(opts)
    # table of all methods
    rows = []
    for m in an.METHOD_LABELS:
        if m not in A_res["baseline"]:
            continue
        kb = an.k_eff_table({m: A_res["baseline"][m]})[m]
        row = {"방법": an.METHOD_LABELS[m], "k_eff 기준 [W/mK]": kb, "편향 [%]": (kb / COPPER.k - 1) * 100}
        if has_void:
            kv = an.k_eff_table({m: A_res["void"][m]})[m]
            row.update({"k_eff void [W/mK]": kv, "Δk/k (void) [%]": (kv / kb - 1) * 100})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.1f}" for c in rows[0] if c != "방법"}), use_container_width=True, hide_index=True)
    st.caption("편향 = 기준 via 의 k_eff 와 Cu 참값 398 W/mK 의 차이 (측면 손실·모델 가정에 의한 체계 편향). 같은 void 에 대해 방법 간 Δk 가 다르게 나타나는 것 자체가 void 지표입니다.")
    # method detail
    rb_a = A_res["baseline"][method]
    detail_keys = {"step_intercept": ["t_int", "slope", "alpha", "k_eff", "rho_cp_eff", "t_from", "t_to"],
                   "parker": ["t_half", "t_peak", "h_max", "alpha", "k_eff"],
                   "two_face": ["t_star", "R_th", "k_eff", "k_eff_spot_corrected", "R_spread"],
                   "fin_fit": ["k_eff", "G_prime", "alpha", "rms", "rms_rel", "success"],
                   "fin_fit_lossless": ["k_eff", "alpha", "rms", "rms_rel", "success"],
                   "tlm": ["k_eff", "alpha", "rms", "success"], "tlm_loss": ["k_eff", "G_prime", "alpha", "rms", "success"],
                   "tlm_shell": ["k_eff", "alpha", "rms", "success"]}
    drows = []
    for key in detail_keys[method]:
        if key not in rb_a:
            continue
        row = {"항목": key, "기준": rb_a[key]}
        if has_void:
            row["void"] = A_res["void"][method].get(key)
        drows.append(row)
    c1, c2 = st.columns([1, 2])
    c1.dataframe(pd.DataFrame(drows), use_container_width=True, hide_index=True)
    fig = go.Figure()
    for name, run in runs.items():
        r, a_ = run["resp"], A_res[name][method]
        dash = "solid" if name == "baseline" else "dash"
        if method == "step_intercept":
            fig.add_trace(go.Scatter(x=r.t * ts, y=r.dT_far, name=f"ΔT_far ({name})", line=dict(color="#1f77b4", dash=dash)))
            tt = np.array([a_["t_int"], r.t[-1]])
            fig.add_trace(go.Scatter(x=tt * ts, y=a_["slope"] * tt + a_["offset"], name=f"점근선 ({name}), t_int = {a_['t_int'] * ts:.3g} ms",
                                     line=dict(color="#ff7f0e", dash=dash)))
            fig.update_layout(xaxis_title="t [ms]", yaxis_title="ΔT [K]", title="스텝 절편법: 반대면 곡선과 선형 점근선")
        elif method == "parker":
            fig.add_trace(go.Scatter(x=r.t * ts, y=a_["h"], name=f"임펄스 응답 dΔT_far/dt ({name})", line=dict(color="#1f77b4", dash=dash)))
            fig.add_vline(x=a_["t_half"] * ts, line=dict(color="#ff7f0e", dash=dash), annotation_text=f"t½ = {a_['t_half'] * ts:.3g} ms ({name})")
            fig.update_layout(xaxis_title="t [ms]", yaxis_title="h [K/s]", title="Parker: 임펄스 응답과 반상승 시각", xaxis_type="log")
        elif method == "two_face":
            fig.add_trace(go.Scatter(x=r.t * ts, y=a_["R_th_curve"], name=f"R_th(t) ({name})", line=dict(color="#2ca02c", dash=dash)))
            fig.add_hline(y=r.result.config.geometry.L / (COPPER.k * res["A"]), line=dict(color="gray", dash="dot"), annotation_text="L/(kA) (정상상태 직렬 저항)")
            fig.add_hline(y=r.result.config.geometry.L / (2 * COPPER.k * res["A"]), line=dict(color="gray", dash="dot"), annotation_text="L/(2kA) (단열 로드 준정상)")
            fig.update_layout(xaxis_title="t [ms]", yaxis_title="R_th [K/W]", title="양면 온도차 열저항 R_th(t) = (ΔT_pump − ΔT_far)/P_abs", xaxis_type="log")
        elif method in ("fin_fit", "fin_fit_lossless"):
            fig.add_trace(go.Scatter(x=r.t * ts, y=r.dT_far, name=f"ΔT_far ({name})", line=dict(color="#1f77b4", dash=dash)))
            fig.add_trace(go.Scatter(x=r.t * ts, y=a_["curve"], name=f"핀 모델 피팅 ({name}) k = {a_['k_eff']:.0f}", line=dict(color="#ff7f0e", dash=dash)))
            fig.update_layout(xaxis_title="t [ms]", yaxis_title="ΔT [K]", title="1-D 핀 모델 피팅 (피팅 구간 끝까지)", xaxis_type="log")
        else:
            fr = run["fr"]
            shell = (run["cfg"].geometry.R_cu, FUSED_SILICA) if method == "tlm_shell" else None
            _, Hm = an.tlm_response(fr.freqs, a_["k_eff"], COPPER.rho_cp, res["L"], res["A"], a_.get("G_prime", 0.0), shell)
            fig.add_trace(go.Scatter(x=fr.freqs, y=np.abs(fr.H_far), name=f"|H_far| 계산 ({name})", mode="markers", marker=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=fr.freqs, y=np.abs(Hm), name=f"|H_far| TLM 피팅 ({name}) k = {a_['k_eff']:.0f}", line=dict(color="#ff7f0e", dash=dash)))
            fig.update_layout(xaxis_title="f [Hz]", yaxis_title="|H_far| [K/W]", xaxis_type="log", yaxis_type="log", title="락인 TLM 역산: 반대면 진폭 응답")
    fig.update_layout(height=420, legend=dict(orientation="h"))
    c2.plotly_chart(fig, use_container_width=True)

elif view == "락인":
    chop = st.radio("변조", ["square", "sine"], horizontal=True, format_func=lambda x: "사각 chopping (기본파 2P/π)" if x == "square" else "정현 변조 (P/2)")
    show_fft = st.checkbox("스텝 응답 FFT 교차검증 표시", value=True)
    c1, c2 = st.columns(2)
    fa, fp = go.Figure(), go.Figure()
    for name, run in runs.items():
        fr, r = run["fr"], run["resp"]
        dash = "solid" if name == "baseline" else "dash"
        for H, lab, col in ((fr.H_pump, "펌프면", "#d62728"), (fr.H_far, "반대면", "#1f77b4")):
            lk = an.lockin(H, fr.P_abs, chop)
            fa.add_trace(go.Scatter(x=fr.freqs, y=lk["amplitude"] * 1e3, name=f"{lab} ({name})", line=dict(color=col, dash=dash)))
            fp.add_trace(go.Scatter(x=fr.freqs, y=np.degrees(np.unwrap(np.angle(lk["complex"]))), name=f"{lab} ({name})", line=dict(color=col, dash=dash)))
        if show_fft:
            for sig, lab, col in ((r.dT_pump, "펌프면", "#d62728"), (r.dT_far, "반대면", "#1f77b4")):
                Hs = an.frequency_response_from_step(r.t, sig, fr.freqs, r.P_abs)
                lk = an.lockin(Hs, fr.P_abs, chop)
                fa.add_trace(go.Scatter(x=fr.freqs, y=lk["amplitude"] * 1e3, name=f"{lab} 스텝 FFT ({name})", mode="markers", marker=dict(color=col, symbol="x")))
                fp.add_trace(go.Scatter(x=fr.freqs, y=np.degrees(np.unwrap(np.angle(lk["complex"]))), name=f"{lab} 스텝 FFT ({name})", mode="markers",
                                        marker=dict(color=col, symbol="x")))
    fa.update_layout(title="락인 진폭 (기본파)", xaxis_title="f [Hz]", yaxis_title="진폭 [mK]", xaxis_type="log", yaxis_type="log", height=440, legend=dict(orientation="h"))
    fp.update_layout(title="락인 위상 (지연 음수, unwrap)", xaxis_title="f [Hz]", yaxis_title="위상 [°]", xaxis_type="log", height=440, legend=dict(orientation="h"))
    c1.plotly_chart(fa, use_container_width=True)
    c2.plotly_chart(fp, use_container_width=True)
    tbl = []
    fr = base["fr"]
    for i, f in enumerate(fr.freqs):
        row = {"f [Hz]": f}
        for name, run in runs.items():
            lk = an.lockin(run["fr"].H_far, run["fr"].P_abs, chop)
            row[f"반대면 진폭 [μK] ({name})"] = lk["amplitude"][i] * 1e6
            row[f"반대면 위상 [°] ({name})"] = lk["phase_deg"][i]
        if has_void:
            lb, lv = an.lockin(base["fr"].H_far, base["fr"].P_abs, chop), an.lockin(runs["void"]["fr"].H_far, runs["void"]["fr"].P_abs, chop)
            row["진폭 변화 [%]"] = (lv["amplitude"][i] / lb["amplitude"][i] - 1) * 100
            row["위상 변화 [°]"] = lv["phase_deg"][i] - lb["phase_deg"][i]
        tbl.append(row)
    st.dataframe(pd.DataFrame(tbl).style.format("{:.4g}"), use_container_width=True, hide_index=True)
    st.caption("스텝 FFT 는 시간 적분기의 스텝 응답에서 유도한 값 (LTI). 직접 풀이와의 차이가 시간 해상도의 지표: 관측창 끝 Δt 가 주기보다 길어지는 고주파에서 벌어집니다.")

elif view == "온도장":
    name = st.radio("케이스", list(runs), horizontal=True) if has_void else "baseline"
    r = runs[name]["resp"].result
    snaps = r.snapshots
    i = st.slider("스냅샷", 0, len(snaps) - 1, len(snaps) - 1)
    t_i, T = snaps[i]
    rc, zc = r.grid.r_c / UM, r.grid.z_c / UM
    sel = r.grid.r_c <= r.config.geometry.R_cu
    Tr = T[:, sel] - r.config.numerics.T0
    x = np.concatenate([-rc[sel][::-1], rc[sel]])
    Z = np.concatenate([Tr[:, ::-1], Tr], axis=1)
    vmax = max(float(np.max([s[1].max() for s in snaps]) - r.config.numerics.T0), 1e-9)
    fig = go.Figure(go.Heatmap(x=x, y=zc, z=Z, colorscale=TEMP_COLORSCALE, zmin=0, zmax=vmax, colorbar=dict(title="ΔT [K]")))
    if r.config.void.enabled:
        mv = r.material[:, sel]
        fig.add_trace(go.Contour(x=x, y=zc, z=np.concatenate([mv[:, ::-1], mv], axis=1) == MAT_VOID, showscale=False, contours=dict(start=0.5, end=0.5, coloring="none"),
                                 line=dict(color="white", width=1), hoverinfo="skip"))
    fig.update_layout(title=f"ΔT(r, z) @ t = {t_i * ts:.3g} ms ({name}) — 펌프면 z = 0 (아래), 반대면 z = L (위)", xaxis_title="r [μm]", yaxis_title="z [μm]",
                      height=700, yaxis=dict(autorange="reversed"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("구리 단면(|r| ≤ R_cu)만 표시. 실리카는 계산에 포함되지만 표시 범위 밖.")

elif view == "검증":
    st.markdown("**에너지 보존** — 흡수 = 저장 (+ 이웃 싱크 방출)")
    rows = []
    for name, run in runs.items():
        r = run["resp"]
        rows.append({"케이스": name, "흡수 [μJ]": r.E_in[-1] * 1e6, "저장 [μJ]": r.E_stored[-1] * 1e6, "싱크 방출 [μJ]": r.E_out[-1] * 1e6,
                     "오차 (끝)": r.energy_error, "오차 (최대)": r.energy_error_max})
    st.dataframe(pd.DataFrame(rows).style.format({"흡수 [μJ]": "{:.4g}", "저장 [μJ]": "{:.4g}", "싱크 방출 [μJ]": "{:.4g}", "오차 (끝)": "{:.1e}", "오차 (최대)": "{:.1e}"}),
                 use_container_width=True, hide_index=True)
    st.markdown("**주파수 영역 직접 풀이 vs 스텝 응답 FFT** (기준 via, 반대면)")
    fr, r = base["fr"], rb
    Hs = an.frequency_response_from_step(r.t, r.dT_far, fr.freqs, r.P_abs)
    tbl = pd.DataFrame({"f [Hz]": fr.freqs, "|H| 직접 [K/W]": np.abs(fr.H_far), "|H| 스텝 FFT [K/W]": np.abs(Hs),
                        "진폭비": np.abs(Hs) / np.abs(fr.H_far), "위상차 [°]": np.degrees(np.angle(Hs / fr.H_far))})
    st.dataframe(tbl.style.format("{:.4g}"), use_container_width=True, hide_index=True)
    st.markdown("**1-D 해석해 (Carslaw–Jaeger, 측면 단열 막대)** — 측면 손실이 없을 때의 반대면 곡선. 실제(유리 매립) 곡선과의 차이가 측면 손실의 크기입니다.")
    q = rb.P_abs / res["A"]
    ana = slab_far_face_step(rb.t, q, res["L"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rb.t * ts, y=rb.dT_far, name="반대면 계산 (기준)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=rb.t * ts, y=ana, name="1-D 단열 막대 해석해", line=dict(color="gray", dash="dot")))
    fig.update_layout(xaxis_title="t [ms]", yaxis_title="ΔT [K]", height=400, xaxis_type="log", legend=dict(orientation="h"),
                      title=f"절편 L²/(6α) = {slab_intercept(res['L']) * ts:.3g} ms (Cu)")
    st.plotly_chart(fig, use_container_width=True)
    g = rb.result.grid
    st.caption(f"격자: nr = {g.nr}, nz = {g.nz}, 외곽 반경 {g.r_faces[-1] / UM:.0f} μm, 경계 {cfg_b.geometry.outer_bc}; "
               f"Δz 최대 {np.max(np.diff(g.z_faces)) / UM:.1f} μm; 시간 스텝 {rb.result.n_steps}, Δt 최대 {np.max(np.diff(rb.t)) * ts:.3g} ms.")
