"""ns-TTR Void Detection Simulator - Streamlit GUI.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import RegularGridInterpolator

from ttr_sim import DEPTH_PRESETS, KVOID_PRESETS, Geometry, Laser, Numerics, SimConfig, VoidSpec
from ttr_sim.materials import AIR, COPPER, COPPER_ABSORPTION_DEPTH, FUSED_SILICA
from ttr_sim.presets import fmt_length, fmt_time
from ttr_sim.solver import MAT_VOID, build_grid, material_map
from ttr_sim.validation import analytic_comparison, grid_convergence, kvoid_sensitivity, run_pair

st.set_page_config(page_title="ns-TTR Void Simulator", page_icon="🔬", layout="wide")

UM = 1e-6


# ----------------------------------------------------------------------------- helpers
def time_unit(t_end: float) -> tuple[float, str]:
    if t_end < 2e-6:
        return 1e9, "ns"
    if t_end < 2e-3:
        return 1e6, "μs"
    return 1e3, "ms"


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
    )


def on_preset_change():
    for key, val in preset_defaults(st.session_state.preset_idx).items():
        st.session_state[key] = val


def init_state():
    if "preset_idx" not in st.session_state:
        st.session_state.preset_idx = 2
        for key, val in preset_defaults(2).items():
            st.session_state[key] = val
    st.session_state.setdefault("kvoid_idx", 2)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("conv", None)
    st.session_state.setdefault("kv", None)


def build_config() -> SimConfig:
    s = st.session_state
    p = DEPTH_PRESETS[s.preset_idx]
    kv = KVOID_PRESETS[s.kvoid_idx]
    void = VoidSpec(
        enabled=s.void_enabled, depth=s.void_depth_um * UM, thickness=s.void_thickness_um * UM,
        r_center=s.void_rc_um * UM, r_half=s.void_r_um * UM, k=kv.k, rho_cp=AIR.rho_cp,
    )
    laser = Laser(
        tau_p=p.tau_p, profile=s.profile, energy=s.energy_nJ * 1e-9, w=s.spot_um * UM,
        reflectivity=s.reflectivity, probe_w=s.probe_um * UM,
    )
    numerics = Numerics(dz=p.dz, fo=s.fo, t_end_factor=s.t_end_factor, stretch=s.stretch)
    geometry = Geometry(homogeneous_copper=s.homogeneous)
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
                  annotation_text="펄스", annotation_position="top left")


def fmt_pct(x: float) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}%"


def fmt_sci(x: float) -> str:
    return f"{x:.2e}"


# ----------------------------------------------------------------------------- sidebar
init_state()
with st.sidebar:
    st.title("🔬 ns-TTR Void Simulator")
    st.caption("구리 마이크로 실린더(⌀80 μm × 500 μm, fused silica 매립) 내부 void의 펌프-프로브 thermoreflectance 검출 가능성 시뮬레이션")

    st.header("1. Void 깊이 프리셋")
    st.selectbox(
        "목표 void 깊이 → 펄스폭 + 격자 (짝으로 결정)", options=list(range(len(DEPTH_PRESETS))),
        format_func=lambda i: DEPTH_PRESETS[i].display, key="preset_idx", on_change=on_preset_change,
        help="τp = 2·d²/D_th, Δz = √(D_th·τp)/10. 펄스폭과 격자는 직접 입력하지 않고 프리셋으로만 바꿉니다.",
    )
    preset = DEPTH_PRESETS[st.session_state.preset_idx]
    if preset.warning:
        st.warning(preset.warning)

    st.header("2. k_void (void 열전도율)")
    st.selectbox(
        "수치적 floor 값 — 민감도를 꼭 확인하세요", options=list(range(len(KVOID_PRESETS))),
        format_func=lambda i: KVOID_PRESETS[i].label, key="kvoid_idx",
    )
    kv = KVOID_PRESETS[st.session_state.kvoid_idx]
    st.caption(kv.description)
    if kv.warning:
        st.warning(kv.warning)

    st.header("3. Void 형상")
    st.checkbox("void 포함", value=True, key="void_enabled")
    c1, c2 = st.columns(2)
    c1.number_input("깊이 (윗면 z) [μm]", min_value=0.0, max_value=500.0, step=0.1, key="void_depth_um", format="%.3f")
    c2.number_input("두께 [μm]", min_value=0.01, max_value=500.0, step=0.1, key="void_thickness_um", format="%.3f")
    c1.number_input("반경 위치 (0 = 축상) [μm]", min_value=0.0, max_value=40.0, step=0.5, key="void_rc_um", value=0.0, format="%.2f",
                    help="0이면 축상의 원판형 void, 0보다 크면 축대칭 링(고리)형 void로 근사합니다.")
    c2.number_input("반경 반폭 [μm]", min_value=0.01, max_value=40.0, step=0.5, key="void_r_um", format="%.3f")

    st.header("4. 레이저")
    st.radio("펄스 시간 프로파일", ["square", "gaussian"], key="profile", horizontal=True,
             help="square: 폭 τp의 사각 펄스, gaussian: FWHM = τp (동일 fluence로 정규화)")
    c1, c2 = st.columns(2)
    c1.number_input("펄스 에너지 [nJ]", min_value=1e-3, max_value=1e7, key="energy_nJ", format="%.4g")
    c2.number_input("펌프 1/e² 반경 [μm]", min_value=1.0, max_value=40.0, value=10.0, step=1.0, key="spot_um")
    c1.number_input("프로브 1/e² 반경 [μm]", min_value=0.0, max_value=40.0, value=5.0, step=0.5, key="probe_um",
                    help="0이면 중심 셀 온도")
    c2.number_input("구리 반사율 R", min_value=0.0, max_value=0.99, value=0.6, step=0.05, key="reflectivity")

    with st.expander("고급 수치 설정"):
        st.select_slider("Fourier 수 Fo = D·Δt/Δz² (Δt 결정)", options=[0.125, 0.25, 0.5, 1.0, 2.0], value=0.5, key="fo",
                         help="Crank–Nicolson은 무조건 안정이지만 Fo가 크면 급격한 transient에서 진동/정확도 저하가 생길 수 있습니다.")
        st.slider("관측 시간창 (× τp)", min_value=2.0, max_value=20.0, value=5.0, step=1.0, key="t_end_factor")
        st.slider("격자 성장률 (외곽 stretched 영역)", min_value=1.05, max_value=1.5, value=1.15, step=0.05, key="stretch")
        st.checkbox("균질 구리 검증 모드 (실리카 → 구리)", value=False, key="homogeneous",
                    help="해석해(semi-infinite 구리)와 직접 비교할 때 이산화 오차만 분리하기 위한 모드")

    cfg_preview = build_config()
    d = cfg_preview.diagnostics()
    try:
        grid_preview = build_grid(cfg_preview)
        n_cells = grid_preview.n_cells
        grid_err = None
    except ValueError as e:
        n_cells, grid_err = None, str(e)
    st.markdown("---")
    st.markdown(
        f"**τp** = {fmt_time(d['tau_p'])}  ·  **Δz** = {fmt_length(d['dz'])}  ·  **Δr** = {fmt_length(d['dr'])}  \n"
        f"**Δt** = {fmt_time(d['dt'])} (Fo={d['fo']:g})  ·  **스텝** = {d['n_steps']:,}  ·  **셀** = {n_cells if n_cells else '—'}  \n"
        f"**관측창** = {fmt_time(d['t_end'])}  ·  **√(D·t_end)** = {fmt_length(d['L_diff'])}"
    )
    for w in d["warnings"]:
        st.warning(w)
    if grid_err:
        st.error(grid_err)
    apply = st.button("▶ 적용 (재계산)", type="primary", use_container_width=True, disabled=grid_err is not None)


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
        st.session_state.kv = None
    except Exception as e:  # noqa: BLE001
        st.exception(e)
    finally:
        bar.empty()

res = st.session_state.result
st.title("ns-TTR Void Detection Simulator")
if res is None:
    st.info("왼쪽 패널에서 프리셋과 void 형상을 선택한 뒤 **적용 (재계산)** 을 누르세요.")
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
m = st.columns(6)
m[0].metric("피크 ΔT (baseline)", f"{sig['peak_rise_base']:.3g} K", help="프로브 가중 표면 온도 상승의 최대값 (void 없음)")
m[1].metric("void 신호 피크", f"{sig['peak_dT']:+.3g} K", help="프로브 가중 표면온도: void − baseline")
m[2].metric("상대 대비", f"{sig['peak_contrast'] * 100:+.1f} %",
            help="ΔT/ΔT_baseline — baseline 온도상승이 피크의 1% 이상인 구간에서의 최대 상대 차이")
m[3].metric("신호 피크 시각", f"{sig['t_peak'] * tscale:.3g} {tunit}")
m[4].metric("에너지 오차", f"{max(abs(base.energy_error), abs(void.energy_error)):.1e}",
            help="(저장 열에너지 − 입력 에너지)/입력 에너지, 최종 시각")
m[5].metric("해석해 RMS 편차", f"{ana['rms_rel'] * 100:.2f} %", help="baseline vs semi-infinite 구리 + Gaussian spot 해석해")
for w in diag["warnings"]:
    st.warning(w)
if not cfg.void.enabled:
    st.info("void 가 비활성화되어 있어 신호 지표는 0 입니다 (baseline 만 계산).")

# A radio (not st.tabs) so the selected view survives the rerun triggered by the test buttons.
VIEWS = ["📈 신호", "🌡 온도장", "✅ 검증 지표", "🔬 Grid convergence", "🧪 k_void 민감도"]
view = st.radio("보기", VIEWS, horizontal=True, key="view", label_visibility="collapsed")

# ----------------------------------------------------------------------------- signal tab
if view == VIEWS[0]:
    t = base.times * tscale
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=base.dT_probe, name="baseline (void 없음)", line=dict(color="#1f77b4")))
    if cfg.void.enabled:
        fig.add_trace(go.Scatter(x=void.times * tscale, y=void.dT_probe, name=f"void (k_void={cfg.void.k:g})", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=t, y=ana["analytic"], name="해석해 (semi-infinite Cu)", line=dict(color="gray", dash="dash")))
    pulse_shading(fig, cfg.laser, tscale)
    fig.update_layout(title="프로브 가중 표면 온도 상승", xaxis_title=f"t [{tunit}]", yaxis_title="ΔT [K]", height=420,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    if cfg.void.enabled:
        c1, c2 = st.columns(2)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=void.times * tscale, y=sig["dT"], name="ΔT = void − baseline", line=dict(color="#d62728")))
        pulse_shading(fig2, cfg.laser, tscale)
        fig2.update_layout(title="void 신호 ΔT(t)", xaxis_title=f"t [{tunit}]", yaxis_title="ΔT [K]", height=360)
        c1.plotly_chart(fig2, use_container_width=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=void.times * tscale, y=sig["contrast"] * 100, name="상대 대비", line=dict(color="#2ca02c")))
        pulse_shading(fig3, cfg.laser, tscale)
        fig3.update_layout(title="상대 대비 ΔT / ΔT_baseline", xaxis_title=f"t [{tunit}]", yaxis_title="[%]", height=360)
        c2.plotly_chart(fig3, use_container_width=True)
        tau_void = cfg.void.depth ** 2 / cfg.copper.alpha
        st.caption(
            f"참고: void 깊이 d = {cfg.void.depth * 1e6:.3g} μm 의 특징 시간 τ_void = d²/D = {fmt_time(tau_void)} "
            f"(펄스폭 τp = {fmt_time(cfg.laser.tau_p)}, τp/τ_void = {cfg.laser.tau_p / tau_void:.2g}). "
            f"신호 피크는 보통 τ_void 이후에 나타납니다."
        )

# ----------------------------------------------------------------------------- field tab
if view == VIEWS[1]:
    src_choice = st.radio("표시 대상", ["void 케이스", "baseline", "차이 (void − baseline)"], horizontal=True,
                          disabled=not cfg.void.enabled)
    full = st.checkbox("전체 도메인 표시 (기본: 관심 영역 확대)", value=False)
    snaps = void.snapshots if cfg.void.enabled else base.snapshots
    idx = st.slider("스냅샷", 0, len(snaps) - 1, min(len(snaps) - 1, len(snaps) // 3),
                    format="%d", help="시간 순으로 저장된 온도장 스냅샷")
    t_snap, T2 = snaps[idx]
    grid = void.grid if cfg.void.enabled else base.grid
    field = T2 - T0
    if src_choice == "baseline":
        _, Tb = base.snapshots[idx]
        grid, field = base.grid, Tb - T0
    elif src_choice.startswith("차이"):
        _, Tb = base.snapshots[idx]
        interp = RegularGridInterpolator((base.grid.z_c, base.grid.r_c), Tb, bounds_error=False, fill_value=None)
        ZZ, RR = np.meshgrid(grid.z_c, grid.r_c, indexing="ij")
        field = T2 - interp(np.stack([ZZ.ravel(), RR.ravel()], axis=1)).reshape(T2.shape)

    if full:
        r_lim, z_lim = grid.r_faces[-1], grid.z_faces[-1]
    else:
        r_lim = min(grid.r_faces[-1], max(3 * cfg.laser.w, (cfg.void.r_outer + 3 * cfg.dr) if cfg.void.enabled else 0, 3 * cfg.penetration_length(t_snap if t_snap > 0 else cfg.dt)))
        z_lim = min(grid.z_faces[-1], max(3 * cfg.penetration_length(max(t_snap, cfg.dt)), (cfg.void.z_bottom + 5 * cfg.numerics.dz) if cfg.void.enabled else 0, 5 * cfg.numerics.dz))
    ir = np.searchsorted(grid.r_c, r_lim) + 1
    iz = np.searchsorted(grid.z_c, z_lim) + 1
    sub = field[:iz, :ir]
    cmax = float(np.max(np.abs(sub))) if sub.size else 1.0
    is_diff = src_choice.startswith("차이")
    fig = go.Figure(go.Heatmap(
        x=grid.r_c[:ir] * 1e6, y=grid.z_c[:iz] * 1e6, z=sub,
        colorscale="RdBu_r" if is_diff else "Inferno", zmid=0.0 if is_diff else None,
        zmin=-cmax if is_diff else 0.0, zmax=cmax if is_diff else None,
        colorbar=dict(title="ΔT [K]"),
        hovertemplate="r=%{x:.2f} μm<br>z=%{y:.2f} μm<br>ΔT=%{z:.4g} K<extra></extra>",
    ))
    # material outlines
    if grid.r_faces[ir - 1] * 1e6 >= cfg.geometry.R_cu * 1e6 * 0.999 and not cfg.geometry.homogeneous_copper:
        fig.add_vline(x=cfg.geometry.R_cu * 1e6, line=dict(color="white", dash="dot"), annotation_text="Cu | SiO₂")
    if cfg.void.enabled and src_choice != "baseline":
        v = cfg.void
        fig.add_shape(type="rect", x0=v.r_inner * 1e6, x1=min(v.r_outer, cfg.geometry.R_cu) * 1e6, y0=v.depth * 1e6, y1=v.z_bottom * 1e6,
                      line=dict(color="cyan", width=2), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=f"{src_choice} — t = {t_snap * tscale:.3g} {tunit}", xaxis_title="r [μm]", yaxis_title="z (깊이) [μm]",
        yaxis=dict(autorange="reversed"), height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**축상(r≈0) 깊이 프로파일**")
        figa = go.Figure()
        for k_i in np.linspace(0, len(snaps) - 1, min(6, len(snaps))).round().astype(int):
            ts, Tk = snaps[k_i]
            figa.add_trace(go.Scatter(x=grid.z_c[:iz] * 1e6 if src_choice != "baseline" else base.grid.z_c[:iz] * 1e6,
                                      y=(Tk[:iz, 0] - T0), name=f"t={ts * tscale:.3g} {tunit}"))
        if cfg.void.enabled:
            figa.add_vrect(x0=cfg.void.depth * 1e6, x1=cfg.void.z_bottom * 1e6, fillcolor="cyan", opacity=0.15, line_width=0, annotation_text="void")
        figa.update_layout(xaxis_title="z [μm]", yaxis_title="ΔT [K]", height=380, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(figa, use_container_width=True)
    with c2:
        st.markdown("**표면(z=0) 반경 프로파일**")
        figr = go.Figure()
        res_r = void if (cfg.void.enabled and src_choice != "baseline") else base
        for k_i in np.linspace(0, len(res_r.times) - 1, 6).round().astype(int):
            figr.add_trace(go.Scatter(x=res_r.grid.r_c[:ir] * 1e6, y=res_r.T_surface[k_i, :ir] - T0,
                                      name=f"t={res_r.times[k_i] * tscale:.3g} {tunit}"))
        figr.update_layout(xaxis_title="r [μm]", yaxis_title="ΔT [K]", height=380, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(figr, use_container_width=True)

    if st.checkbox("3D 표면 플롯 (T(r,z) surface)", value=False):
        fig3d = go.Figure(go.Surface(x=grid.r_c[:ir] * 1e6, y=grid.z_c[:iz] * 1e6, z=sub, colorscale="Inferno"))
        fig3d.update_layout(scene=dict(xaxis_title="r [μm]", yaxis_title="z [μm]", zaxis_title="ΔT [K]"), height=600)
        st.plotly_chart(fig3d, use_container_width=True)

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
        "모든 경계가 단열(표면 flux 제외)이고 유한체적 이산화가 보존적이므로 이 오차는 행렬 조립 + 선형해 정밀도(~1e-13)를 반영합니다. "
        "값이 1e-8 이상으로 커지면 격자/시간 간격 설정에 문제가 있다는 신호입니다."
    )

    st.subheader("사용된 수치 파라미터")
    rows = [
        ("펄스폭 τp", fmt_time(diag["tau_p"])),
        ("Δz (프리셋, void 영역)", fmt_length(diag["dz"])),
        ("Δz (표면 첫 셀)", fmt_length(void.grid.dz_c[0])),
        ("Δr (관심 영역)", fmt_length(diag["dr"])),
        ("Δt", fmt_time(diag["dt"])),
        ("Fo = D·Δt/Δz²", f"{diag['fo']:g}"),
        ("스텝 수", f"{diag['n_steps']:,}"),
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
if view == VIEWS[4]:
    st.markdown(
        "k_void 는 물리량이 아닌 **수치적 타협값**입니다. 아래 버튼은 §4 의 후보값 4개를 자동으로 순회하여 "
        "void 신호 크기와 에너지 보존 오차를 나란히 비교합니다 (baseline 은 재사용)."
    )
    st.caption(f"예상 소요 시간 ≈ {void.wall_time * 4:.0f} s")
    if st.button("▶ k_void 민감도 테스트 실행", disabled=not cfg.void.enabled):
        box = st.container()
        bar, cb = progress_ui(box)
        try:
            st.session_state.kv = kvoid_sensitivity(cfg, base, progress=cb)
        except Exception as e:  # noqa: BLE001
            st.exception(e)
        finally:
            bar.empty()
    kvr = st.session_state.kv
    if kvr:
        labels = {p.k: p.label for p in KVOID_PRESETS}
        table = [{
            "k_void 옵션": labels.get(r["k"], f"{r['k']:g}"), "k [W/m·K]": f"{r['k']:g}",
            "피크 void ΔT [K]": f"{r['peak_dT']:+.5g}", "대비 [%]": f"{r['peak_contrast'] * 100:+.3f}",
            "피크 시각": f"{r['t_peak'] * tscale:.3g} {tunit}",
            "vs 공기값(0.026)": fmt_pct(r["d_peak_dT_vs_min"]), "에너지 오차": fmt_sci(r["energy_error"]), "시간 [s]": f"{r['wall']:.1f}",
        } for r in kvr]
        st.dataframe(pd.DataFrame(table).set_index("k_void 옵션"), use_container_width=True)
        fk = go.Figure()
        for r in kvr:
            fk.add_trace(go.Scatter(x=r["t"] * tscale, y=r["dT"], name=f"k_void = {r['k']:g}"))
        pulse_shading(fk, cfg.laser, tscale)
        fk.update_layout(title="void 신호 ΔT(t) — k_void 비교", xaxis_title=f"t [{tunit}]", yaxis_title="ΔT [K]", height=400,
                         legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fk, use_container_width=True)
        big = [r for r in kvr if r["k"] >= 10]
        if big and abs(big[0]["d_peak_dT_vs_min"]) > 0.05:
            st.warning(
                f"k_void = 10 W/m·K 는 실제 공기값 대비 신호 피크가 {big[0]['d_peak_dT_vs_min'] * 100:+.1f}% 다릅니다. "
                "이 값은 물리적으로 정당화되지 않으므로 결과 해석 시 주의하세요."
            )
        else:
            st.success("k_void 선택에 대한 신호 민감도가 5% 이내입니다: 결과는 floor 값에 크게 의존하지 않습니다.")
