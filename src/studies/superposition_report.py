"""Word report 'TGV 다중 void 검출 시뮬레이션 결과 보고': the multi-void study (results/multi_void) and the
superposition study (results/superposition) in one document, written as "cases tried -> results -> features
seen when compared with a single void".

Run from the src directory after superposition_analysis.py:
    .venv\\Scripts\\python.exe studies\\superposition_report.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "studies"))

from detection_map import CRITERIA_DEFAULT, N_SIGMA  # noqa: E402
from multi_void import WINDOW_US  # noqa: E402
from superposition import OUT, P1_GAPS, P2_SPACINGS  # noqa: E402

MV = SRC / "results" / "multi_void"
SUM_MV, SUM_SP = MV / "summary", OUT / "summary"
FIG_MV, FIG_SP = MV / "figs", OUT / "figs"
SIGMA3 = N_SIGMA * CRITERIA_DEFAULT["sigma"]
C_MIN = CRITERIA_DEFAULT["c_min"]
TITLE = "TGV 다중 void 검출 시뮬레이션 결과 보고"


def load():
    A = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    rows = {}
    for folder in (MV / "cases", OUT / "cases"):
        for jp in folder.glob("*.json"):
            r = json.loads(jp.read_text(encoding="utf-8"))
            rows[r["id"]] = r
    return A, rows


def build():
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    A, R = load()
    FONT = "Malgun Gothic"
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.0)
        s.top_margin = s.bottom_margin = Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = FONT, Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for name in ("Heading 1", "Heading 2", "Heading 3", "Title"):
        hs = doc.styles[name]
        hs.font.name = FONT
        hs.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        hs.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

    def P(text, bold=False, italic=False, size=None, align=None, style=None):
        p = doc.add_paragraph(style=style)
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if align == "c":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return p

    def H(text, level):
        return doc.add_heading(text, level)

    def B(text):
        return P(text, style="List Bullet")

    def table(header, rows, widths=None, size=8.5):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            run = c.paragraphs[0].add_run(str(h))
            run.bold = True
            run.font.size = Pt(size)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(v))
                run.font.size = Pt(size)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        doc.add_paragraph()

    def fig(path: Path, caption, width_cm=16.5):
        if not path.exists():
            P(f"[그림 없음: {path.name}]", italic=True)
            return
        doc.add_picture(str(path), width=Cm(width_cm))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        P(caption, italic=True, size=9, align="c")

    def mk(x):
        return f"{x * 1e3:.3g} mK"

    def vd(r):
        return "검출 가능" if r["detectable"] else f"불가 ({r['limiting']})"

    def gain(cid, ref="A_d070_axis"):
        return abs(R[cid]["void_minus_base_K"] / R[ref]["void_minus_base_K"])

    A1, A2, A3 = A["A1"], A["A2"], A["A3"]
    base = R["A_d070_axis"]
    b9, d6, d3, c3 = R.get("B_N9"), R.get("D_N6"), R.get("D_N3"), R.get("C_stack3")
    p1 = {r["gap_um"]: r for r in A1["P1"]}
    p2 = {r["spacing_um"]: r for r in A1["P2"]}
    p3 = {r["rc_lower_um"]: r for r in A1["P3"]}
    preds = A3["predictions"]
    preds_in = [p for p in preds if p.get("within_library", True)]          # all voids inside the library's validated range
    preds_out = [p for p in preds if not p.get("within_library", True)]
    far_ratio = [r["R_peak"] for r in A1["P2"] if r["spacing_um"] >= 35] + [r["R_peak"] for r in A1["B"]]
    rf = A3["library"]["radial_factors"]

    # ======================================================================= document
    doc.add_heading(TITLE, 0)
    P("ns-TTR Simulator 다중 void 스터디 (studies/multi_void.py, studies/superposition.py)", size=9, align="c")
    P(f"케이스 {len(R)}개 · 생성 {time.strftime('%Y-%m-%d %H:%M')}", size=9, align="c")

    H("요약", 1)
    B(f"질문: 단독으로는 검출되지 않는 void(70 μm 깊이, 반폭·두께 10 μm 공기)가 여러 개 있으면 신호는 어떻게 되는가. "
      f"118 케이스(1차 34, 2차 84)를 계산해 단독 void 응답과 비교했다.")
    B(f"여러 void 의 신호는 각 void 의 단독 신호 곡선 ΔT(t) 의 합이다. 서로 떨어진 배치(B 계열, 간격 35 μm 이상의 쌍)에서 피크 비율 "
      f"계산/합 = {min(far_ratio):.2f}~{max(far_ratio):.2f}. 무작위 10개 배치 {len(preds_in)}종을 단독 라이브러리의 합으로 예측하면 피크의 예측/계산 비가 "
      f"{min(p['ratio_pred_over_computed'] for p in preds_in):.2f}~{max(p['ratio_pred_over_computed'] for p in preds_in):.2f} 이다.")
    if p1 and p2:
        B(f"가까우면 보정이 생긴다. 같은 깊이에 2 μm 틈으로 붙은 쌍은 합의 {p2[min(p2)]['R_peak']:.2f}배, 위아래로 5 μm 틈을 두고 겹친 쌍은 "
          f"합의 {p1[min(p1)]['R_peak']:.2f}배이며, 간격이 커지면 1 로 수렴한다.")
    if b9 and d6:
        B(f"검출을 바꾸는 것은 개수가 아니라 배치다. 70 μm void 아래에 더 깊은 void 9개를 더해도 피크 {gain('B_N9'):.2f}배로 "
          f"'{vd(b9)}' 이고, 같은 깊이에 6개를 나란히 두면 {gain('D_N6'):.2f}배로 '{vd(d6)}' 이다.")
    B("단독 응답의 크기는 부피도 면적도 아닌 열 쌍극자 분극률 V/(1−L_z) 에 비례한다"
      + (f" (로그 기울기 70 μm {A2['70']['slope']:.2f}, 120 μm {A2['120']['slope']:.2f})." if "70" in A2 and "120" in A2 else "."))
    B("따라서 측정 신호는 '깊이로 가중된 분극률의 합'을 알려 주며, 큰 void 하나와 작은 void 여러 개를 구별하지 못한다. "
      "'여러 개'의 기준은 개수가 아니라 같은 깊이 근처에 모인 분극률 합이 단독 검출 한계의 몇 배인가로 두는 것이 합리적이다.")

    H("1. 목적과 질문", 1)
    P("TGV 구리 로드 안의 void 는 ns-TTR 신호에서 열 흐름을 막는 장애물로 나타난다. 앞선 검출한계 스터디는 void 하나의 깊이·크기 "
      "지도를 만들었고, 사용자가 GUI 로 확인한 70 μm 깊이의 10/10 μm void 는 기준에 못 미쳤다. 이 보고서의 질문은 "
      "'void 가 여러 개일 때 신호는 어떻게 되는가' 이다. '여러 개'의 기준이 정해져 있지 않으므로, 개수·깊이 분포·가로 배치·"
      "형상·무작위 배치를 바꾼 케이스들을 시험하고(2장), 결과를 단독 void 의 응답과 비교해(3장) 드러나는 특징을 정리한다(4장).")

    H("2. 방법", 1)
    H("2.1 모델과 공통 조건", 2)
    table(["항목", "값"], [
        ["구조", "구리 로드 반지름 40 μm, 길이 500 μm, 실리카에 매립. 후면 산화막 5 nm + 열패드 200 μm + 히트싱크(실온)"],
        ["void", "공기 (k = 0.026 W/m·K) 회전타원체. 기본 반폭 10 μm, 두께 10 μm. 윗면 깊이로 위치를 적음"],
        ["펄스·빔", "사각 펄스 τp = 27.6 μs (40~100 μm 프리셋), 에너지 GUI 기본값(표면 피크 ≈ 0.13 K), 펌프 가우시안 w = 10 μm, 프로브 5 μm, 축 위"],
        ["관측창", f"{WINDOW_US:g} μs (전 케이스 동일; 모든 곡선이 같은 324점 시간축을 가져 더할 수 있음)"],
        ["격자·시간", "표면층 Δz = 5.7 μm, void 주변은 void 치수의 1/8, Δt = τp/200 → 펄스 후 스텝당 +5 %"],
        ["계산", "축 위 void 만 있으면 2D 축대칭 직접해, 축을 벗어나면 3D (r, θ, z): 격자 구조 전처리 CG (잔차 1e-11)"],
        ["baseline", "void 를 끈 같은 격자의 계산. 신호 = void − baseline 의 프로브 가중 표면 온도 차 ΔT(t)"],
    ], widths=[2.5, 14.0])
    H("2.2 검출 기준", 2)
    P(f"검출한계 스터디와 같다. 펄스 에너지를 baseline 피크가 {CRITERIA_DEFAULT['dT_allowed']:g} K 가 되도록 스케일한 뒤 "
      f"(i) |ΔR/R| 차이 ≥ {N_SIGMA:g}σ = {SIGMA3:g} (σ = {CRITERIA_DEFAULT['sigma']:g}), (ii) 신호 피크 시각의 상대 대비 ≥ {C_MIN * 100:g} %. "
      "둘 다 만족해야 '검출 가능'. '여유'는 기준 대비 배수다.")
    H("2.3 시험한 케이스", 2)
    P("1차 스터디(34 케이스): 배치의 종류를 바꾸어 어떤 것이 신호를 키우는지 보았다.")
    table(["계열", "내용", "케이스 수"], [
        ["A", "단독 void: 70 μm(축), 70~210 μm(r = 25 μm, 15 μm 간격)", "11"],
        ["B", "70 μm 축 void + 더 깊은 void N개 (90~210 μm, r = 25 μm, 방위각 40° 간격), N = 1, 3, 6, 9", "4"],
        ["C", "70 μm void 바로 아래 적층 3개 (90/105/120 μm, 축 위)", "1"],
        ["D", "70 μm 에 축 + r = 25 μm 고리 1, 2, 5개 (총 2, 3, 6개)", "3"],
        ["E", "70 μm 단독, 형상 6종: 같은 부피 (14.1, 5)/(10, 10)/(7.1, 20), 같은 면적 두께 2/5/10/20 μm", "6"],
        ["F / F0", "무작위 10개: 윗면 깊이 ≥ 70 μm (F) 와 로드 전체 (F0, GUI 랜덤 모드), 시드 1~5", "10"],
    ], widths=[1.5, 13.0, 2.0])
    P("2차 스터디(84 케이스): 1차에서 본 '단독 신호의 합' 틀을 거리의 함수로 검증하고, 단독 응답 라이브러리를 만들었다.")
    table(["계열", "내용", "케이스 수"], [
        ["R", "회귀: A_d070_axis, A_d070_r25 재계산 → 기존 곡선과 오차 0", "2"],
        ["P0", "축 위 단독, 윗면 깊이 10~480 μm 10 μm 간격 + 85/105/115 μm (2D)", str(len([k for k in R if k.startswith("S_axis")]))],
        ["P0'", "축 벗어난 단독: 70 μm r = 10/15/20/28, 90 μm r = 10/15/20, 150 μm r = 10/20/28 (3D)", "10"],
        ["P1", "위아래 쌍(축 위): 70 μm + 윗면 80 + g μm, 틈 g = 5/10/20/35/50/80 μm", "6"],
        ["P2", "나란히 쌍(70 μm, r = 25 μm 고리): 중심 간격 s = 22/25/30/35/42/50 μm", "6"],
        ["P3", "대각선 쌍: 70 μm 축 + 90 μm r = 0(= P1 g10)/10/15/20/25 μm", "4"],
        ["P4", "120 μm 축 위, E 와 같은 형상 6종", "5"],
    ], widths=[1.5, 13.0, 2.0])
    P("기하 규칙: 모든 void 는 r ≤ 38 μm(벽에서 2 μm 여유) 안에 있고 서로 겹치지 않는다(84 케이스 모두 통과). 2D 케이스를 먼저, 3D 는 4개 병렬로 돌렸다.")

    H("3. 결과", 1)
    # ---------------------------------------------------------------- 3.1 single void
    H("3.1 단독 void 의 응답", 2)
    fig(SUM_SP / "fig_A3_library.png", "그림 1. 단독 void 응답의 깊이 축(왼쪽; 축 위 2D 10 μm 간격과 r = 25 μm 의 A 계열)과 반경 위치 인자(오른쪽).", 15.5)
    lib = A3["library"]
    pk = lib["peaks_mK"]
    P(f"축 위 10/10 μm void 의 피크는 70 μm 에서 {pk['70']:.3g} mK, 120 μm 에서 {pk['120']:.3g} mK, 210 μm 에서 {pk['210']:.3g} mK 로 "
      f"깊이 70 → 210 μm 에서 {pk['70'] / pk['210']:.0f}배 줄고, 피크 시각은 깊이의 제곱에 비례해 늦어진다(70 μm 48 μs → 210 μm 270 μs). "
      "400 μm 보다 깊으면 1 ms 창 안에서 피크에 이르지 못한다.")
    rf_txt = []
    for d, pts in rf.items():
        fs = ", ".join(f"r = {p['r']:g}: {p['factor']:.2f}" for p in pts if p["r"] > 0)
        rf_txt.append(f"{d} μm 에서 {fs}")
    P("축에서 벗어나도 응답은 거의 같다. 축 위 대비 피크 비율은 " + "; ".join(rf_txt) + ". "
      "로드 반지름(40 μm)보다 깊은 곳에서는 열 흐름이 거의 1차원이라 가로 위치가 중요하지 않다는 뜻이며, 정규화한 곡선 모양의 차이도 "
      f"{max(p['l2_shape_vs_axis'] for pts in rf.values() for p in pts) * 100:.0f} % 이내다.")
    fig(SUM_SP / "fig_A2_shape.png", "그림 2. 형상 6종의 피크 대 쌍극자 분극률 V/(1−L_z)(왼쪽, 70 μm 와 120 μm), 피크로 정규화한 곡선(가운데 70 μm, 오른쪽 120 μm).")
    rows = []
    for depth, d in A2.items():
        for s in d["shapes"]:
            rows.append([depth, f"{s['rh']:g} / {s['thick']:g}", f"{s['L_z']:.3f}", f"{s['pol_um3']:,.0f}", f"{s['peak_mK']:.4g}", f"{s['t_peak_us']:.0f}", f"{s['l2_shape_vs_ref']:.3f}"])
    table(["깊이 [μm]", "반폭 / 두께 [μm]", "L_z", "V/(1−L_z) [μm³]", "피크 [mK]", "t_peak [μs]", "정규화 곡선 L2 (대 10/10)"], rows, widths=[1.6, 2.6, 1.4, 2.6, 2.0, 1.8, 3.2])
    if "70" in A2 and "120" in A2:
        P(f"같은 깊이에서 여섯 형상의 피크는 분극률 한 직선에 모인다(로그 기울기 70 μm {A2['70']['slope']:.2f}, 산포 {A2['70']['scatter'] * 100:.0f} %; "
          f"120 μm {A2['120']['slope']:.2f}, 산포 {A2['120']['scatter'] * 100:.0f} %). 피크로 정규화한 곡선은 형상과 무관하게 겹치므로"
          f"(L2 차이 최대 {max(s['l2_shape_vs_ref'] for d in A2.values() for s in d['shapes']) * 100:.0f} %), 같은 깊이의 void 는 곡선 전체가 "
          "분극률 하나로 비례 조정된다. 얇은 원판도 부피가 0 에 가까운데 8a³/3 의 분극률을 가지고, 같은 반폭에서 두께가 늘면 분극률이 구(2πa³)까지 "
          "2.4배 는다. '두께는 무관하다'는 1차 스터디 전의 예상은 틀렸다.")
    # ---------------------------------------------------------------- 3.2 separated voids
    H("3.2 서로 떨어진 void 들: 신호는 단독 곡선의 합", 2)
    fig(SUM_MV / "fig_B_superposition.png", "그림 3. B 계열: 70 μm void 아래에 깊은 void 를 N개 더할 때, 계산(실선)과 단독 곡선의 합(점선), 피크와 검출 지표.")
    rows = [[r["id"], r.get("n_added", ""), f"{r['peak_computed_mK']:.4g}", f"{r['peak_sum_mK']:.4g}", f"{r['R_peak']:.3f}", f"{r['dt_peak_us']:+.1f}", f"{r['l2_rel']:.3f}",
             r["computed"]["verdict"], r["summed"]["verdict"]] for r in A1["B"]]
    table(["케이스", "추가 N", "계산 피크 [mK]", "합 피크 [mK]", "계산/합", "Δt_peak [μs]", "곡선 L2 차", "판정(계산)", "판정(합)"], rows, widths=[1.8, 1.2, 2.0, 2.0, 1.4, 1.8, 1.6, 2.4, 2.4])
    if A1["B"]:
        bb = A1["B"][-1]
        P(f"깊이가 다르고 가로로 비켜선 void 들은 곡선이 그대로 더해진다. B_N9 의 계산/합 = {bb['R_peak']:.3f}, 곡선 차이 {bb['l2_rel'] * 100:.0f} %. "
          f"그러나 깊은 void 의 기여는 깊이에 따라 급격히 줄어 합이 포화한다: 70 μm 단독 {mk(base['void_minus_base_K'])} → 9개 추가 {mk(b9['void_minus_base_K'])} "
          f"({gain('B_N9'):.2f}배), 6개 → 9개의 증가는 {(gain('B_N9') / gain('B_N6') - 1) * 100:.0f} %.")
    # ---------------------------------------------------------------- 3.3 close voids
    H("3.3 가까운 void 들: 합에 대한 보정", 2)
    fig(SUM_SP / "fig_A1_ratio.png", "그림 4. 계산 피크 / 단독 합 피크: 위아래 틈 g(왼쪽, C 적층 포함), 같은 깊이 중심 간격 s(가운데, D 포함), 대각선 쌍의 아래 void 반경(오른쪽, B 포함).")
    fig(SUM_SP / "fig_A1_curves.png", "그림 5. 쌍 케이스의 곡선: 계산(실선) 대 단독 합(점선).")
    for key, xname, xkey in (("P1", "틈 g [μm]", "gap_um"), ("P2", "중심 간격 s [μm]", "spacing_um"), ("P3", "아래 void r [μm]", "rc_lower_um"), ("D", "총 개수", "n_total"), ("C", "", None)):
        rows = A1[key]
        if not rows:
            continue
        table(["케이스", xname or "-", "계산 피크 [mK]", "합 피크 [mK]", "계산/합", "Δt_peak [μs]", "곡선 L2 차", "판정(계산)", "판정(합)", "뒤바뀜"],
              [[r["id"], r.get(xkey, "-") if xkey else "-", f"{r['peak_computed_mK']:.4g}", f"{r['peak_sum_mK']:.4g}", f"{r['R_peak']:.3f}", f"{r['dt_peak_us']:+.1f}",
                f"{r['l2_rel']:.3f}", r["computed"]["verdict"], r["summed"]["verdict"], "예" if r["verdict_flips"] else ""] for r in rows],
              widths=[1.8, 1.6, 1.9, 1.9, 1.3, 1.7, 1.5, 2.2, 2.2, 1.0])
    if p1:
        g_sorted = sorted(p1)
        P(f"위아래로 겹친 쌍(P1): 틈 {g_sorted[0]:g} μm 에서 계산/합 = {p1[g_sorted[0]]['R_peak']:.2f}, "
          + ", ".join(f"{g:g} μm 에서 {p1[g]['R_peak']:.2f}" for g in g_sorted[1:]) + ". "
          + ("아래 void 가 위 void 의 열그림자에 들어가므로 합보다 작아지고, 틈이 커질수록 1 에 가까워진다." if p1[g_sorted[0]]["R_peak"] < 0.97 else
             "열그림자에 의한 감소가 거의 없다: 두 void 가 하나의 두꺼운 장애물처럼 작용하는 효과(3.1 의 두께 효과)가 그림자 손실을 상쇄한다."))
    if c3 and A1["C"]:
        P(f"적층 3개(C_stack3)는 축 위 단독 곡선의 합 대비 {A1['C'][0]['R_peak']:.2f} (1차 보고서에서 r = 25 μm 단독으로 대신 더했을 때의 "
          f"{A1['C'][0]['R_peak_with_r25_proxy']:.2f} 는 대용 곡선 때문에 과소평가된 값이었다).")
    if p2:
        s_sorted = sorted(p2)
        P(f"같은 깊이에 나란히 놓인 쌍(P2): 중심 간격 {s_sorted[0]:g} μm(틈 2 μm)에서 계산/합 = {p2[s_sorted[0]]['R_peak']:.2f}, "
          + ", ".join(f"{s:g} μm 에서 {p2[s]['R_peak']:.2f}" for s in s_sorted[1:]) + ". "
          "이웃한 void 가 함께 단면을 좁혀 우회로가 줄어드는 상호작용으로 합보다 커지며, 간격이 반폭의 3배를 넘으면 사라진다. "
          + (f"D 계열의 2/3/6개(중심 간격 25 μm)도 {', '.join(f'{r['R_peak']:.2f}' for r in A1['D'])} 로 같은 경향이다." if A1["D"] else ""))
    if p3:
        r_sorted = sorted(p3)
        P("대각선 쌍(P3): 아래 void 를 축에서 r 만큼 비키면 계산/합 = " + ", ".join(f"r = {r:g}: {p3[r]['R_peak']:.2f}" for r in r_sorted)
          + ". 위아래 겹침에서 가로 분리로 넘어가는 전환이 반폭 한두 개 거리 안에서 일어난다.")
    # ---------------------------------------------------------------- 3.4 random prediction
    H("3.4 무작위 배치의 예측", 2)
    fig(SUM_SP / "fig_A3_prediction.png", "그림 6. 무작위 10개 배치: 계산(실선)과 단독 라이브러리 합 예측(점선), 피크의 예측 대 계산.")
    P("단독 라이브러리 s(t; 깊이, r): 축 위 2D 곡선(10~480 μm, 10 μm 간격)을 이웃 깊이 사이에서 선형 보간하고, 반경 인자 f(깊이, r)를 "
      "70/90/150 μm 에서 잰 값으로 r 과 깊이에 대해 선형 보간해 곱했다(곡선 모양은 r 과 무관하다고 두었고, 3.1 에서 확인). "
      "각 void 의 깊이와 r 을 넣어 곡선을 더한 것이 예측이다.")
    rows = [[p["id"], f"{p['depth_min_um']:.0f}", f"{p['peak_computed_mK']:.4g}", f"{p['peak_predicted_mK']:.4g}", f"{p['ratio_pred_over_computed']:.2f}",
             f"{p['t_peak_computed_us']:.0f} / {p['t_peak_predicted_us']:.0f}", f"{p['l2_rel']:.2f}",
             f"×{p['computed']['noise_margin']:.2f} / ×{p['predicted']['noise_margin']:.2f}", f"×{p['computed']['ratio_margin']:.2f} / ×{p['predicted']['ratio_margin']:.2f}",
             f"{p['computed']['verdict']} / {p['predicted']['verdict']}", "외삽" if p["extrapolation_notes"] else ""] for p in preds]
    table(["케이스", "얕은 깊이", "계산 [mK]", "예측 [mK]", "예측/계산", "t_peak 계산/예측 [μs]", "곡선 L2 차", "잡음 여유 계산/예측", "대비 여유 계산/예측", "판정 계산/예측", ""],
          rows, widths=[1.7, 1.2, 1.4, 1.4, 1.3, 2.2, 1.3, 2.2, 2.2, 2.4, 0.9], size=7.5)
    ratios = [p["ratio_pred_over_computed"] for p in preds_in]
    flips = [p["id"] for p in preds_in if p["verdict_flips"]]
    P(f"라이브러리 범위 안의 {len(preds_in)}종에서 예측/계산 피크 비는 {min(ratios):.2f}~{max(ratios):.2f}(중앙값 {np.median(ratios):.2f})이고 판정이 뒤바뀐 케이스는 "
      f"{'없다' if not flips else ', '.join(flips)}. 480 μm 보다 깊은 void 와 r = 30 μm 의 void 는 라이브러리 끝값으로 외삽했으나 기여가 "
      "작아 결과에 영향이 없다. F0_seed3 은 표면을 뚫은 void(윗면 0.7 μm)가 있어 제외했다(표면 구멍의 신호는 내부 void 와 다르다)."
      + (" " + ", ".join(f"{p['id']}(가장 얕은 void {p['depth_min_um']:.0f} μm, 예측/계산 {p['ratio_pred_over_computed']:.2f})" for p in preds_out)
         + " 은 70 μm 보다 얕은 void 를 포함한다. 얕은 곳에서는 열이 아직 가로로 퍼지기 전이라 가로 위치가 응답을 크게 바꾸는데, 반경 인자는 "
           "70 μm 이상에서만 쟀으므로 이 범위 밖의 예측은 신뢰할 수 없다. 라이브러리의 적용 한계를 보여 주는 사례로 남긴다." if preds_out else ""))
    # ---------------------------------------------------------------- 3.5 detection
    H("3.5 검출 판정", 2)
    key_ids = ["A_d070_axis", "B_N1", "B_N3", "B_N6", "B_N9", "C_stack3", "D_N2", "D_N3", "D_N6"] + [p["id"] for p in preds]
    rows = []
    for cid in key_ids:
        if cid in R:
            r = R[cid]
            rows.append([cid, r["n_voids"], f"{r['void_minus_base_K'] * 1e3:.3g}", f"{gain(cid):.2f}", f"{r['ratio'] * 100:.2f} %", f"{r['scaled_dRR_diff']:.2e}",
                         f"×{r['noise_margin']:.2f}", f"×{r['ratio_margin']:.2f}", vd(r)])
    table(["케이스", "void 수", "피크 [mK]", "단독 대비", "피크 대비", "|ΔR/R| (10 K)", "잡음 여유", "대비 여유", "판정"], rows, widths=[2.2, 1.2, 1.6, 1.4, 1.6, 2.0, 1.6, 1.6, 2.6])
    det = [cid for cid in key_ids if cid in R and R[cid]["detectable"]]
    P(f"다중 void 케이스 중 검출 가능은 {', '.join(det) if det else '없음'} 이다. "
      + (f"깊은 void 9개를 더한 B_N9 는 대비 기준은 넘지만(×{b9['ratio_margin']:.2f}) 잡음 기준에 ×{b9['noise_margin']:.2f} 로 조금 못 미치고, " if b9 else "")
      + (f"같은 깊이 6개(D_N6)는 두 기준 모두 통과한다(×{d6['noise_margin']:.2f}, ×{d6['ratio_margin']:.2f}). " if d6 else "")
      + (f"같은 깊이 3개(D_N3)는 잡음 기준은 넘지만 대비가 ×{d3['ratio_margin']:.2f} 로 경계에 있다." if d3 else ""))

    # ======================================================================= 4. features
    H("4. 단독 void 와 비교해 드러나는 특징", 1)
    P("아래 규칙은 위 케이스들에서 수치로 확인된 것이다. 적용 범위는 공기 void, 깊이 ≥ 30 μm, 반폭 ≤ 14 μm, 펄스 27.6 μs 이다.")
    B("규칙 1 (선형 중첩). 여러 void 의 신호 ΔT(t) 는 각 void 의 단독 신호 곡선의 합이다. 피크값이 아니라 곡선을 더해야 하며, "
      f"서로 떨어진 void 들에서 계산/합 = {min(far_ratio):.2f}~{max(far_ratio):.2f} 이다.")
    B(f"규칙 2 (단독 응답의 크기). 단독 응답은 깊이에 따라 급격히 줄고(70 → 210 μm 에서 1/{pk['70'] / pk['210']:.0f}), 가로 위치에는 거의 무관하며(반경 인자 "
      f"{min(p['factor'] for pts in rf.values() for p in pts):.2f}~{max(p['factor'] for pts in rf.values() for p in pts):.2f}), 형상은 분극률 V/(1−L_z) 하나로 들어간다.")
    if p1 and p2:
        B(f"규칙 3 (근접 보정). 같은 깊이에 나란히 붙으면 합보다 크고(틈 2 μm 에서 ×{p2[min(p2)]['R_peak']:.2f}), 위아래로 겹치면 "
          f"×{p1[min(p1)]['R_peak']:.2f} 이다. 보정은 틈이 반폭의 2~3배를 넘으면 사라진다.")
    if b9 and d6:
        B(f"규칙 4 (포화와 배치). 깊은 void 를 아무리 더해도 합은 가장 얕은 void 의 몇 배 안에서 포화한다(9개 추가에 ×{gain('B_N9'):.2f}). "
          f"검출을 바꾸는 것은 같은 깊이 근처에 가로로 모인 양이다(같은 깊이 6개에 ×{gain('D_N6'):.2f}, 검출 가능).")
    B("규칙 5 (예측 가능성). 단독 라이브러리의 합으로 임의 배치의 신호를 예측할 수 있고, 무작위 10개 배치에서 피크 오차는 "
      f"{(max(abs(r - 1) for r in ratios)) * 100:.0f} % 이내(깊이 ≥ 70 μm 범위), 판정은 모두 일치한다." if not flips else
      "규칙 5 (예측 가능성). 단독 라이브러리의 합으로 임의 배치의 신호를 예측할 수 있다. 판정이 뒤바뀐 경우는 경계값 케이스다.")

    H("5. 측정 쪽 함의", 1)
    B("비유일성: 신호가 합으로 정해지므로 신호 하나로는 큰 void 하나와 작은 void 여러 개를 구별할 수 없다. ns-TTR 신호가 알려 주는 것은 "
      "깊이별로 가중된 분극률의 합이며, 깊이 정보는 피크 시각(≈ 1.2 d²/D)에 들어 있다.")
    B("'여러 개'의 기준: 개수가 아니라 '가장 얕은 깊이 근처에 모인 분극률 합이 단독 검출 한계의 몇 배인가'다. 이 조건에서 70 μm 의 10/10 μm void 는 "
      f"단독으로 잡음 기준의 ×{base['noise_margin']:.2f} 이므로, 같은 깊이에 분극률 합이 약 {1 / base['noise_margin']:.1f}배(void 3개 안팎) 모여야 "
      "검출 경계에 이르고, 6개면 여유 있게 검출된다.")
    B("깊은 void 들은 늦게 작게 기여하므로, 관측창을 가장 깊은 관심 void 의 2~3 d²/D 까지 두고 피크 시각뿐 아니라 늦은 시각의 대비도 함께 평가해야 한다.")

    H("6. 한계", 1)
    B("선형 모델이며 물성·흡수율의 온도 의존성이 없다. 판정은 baseline 피크 10 K 스케일, σ = 1e-6, 대비 2 % 기준이다.")
    B("쌍극자 분극률 해석은 무한 매질의 균일 구배에 대한 것이다. 로드 벽에 가깝거나 서로 2 μm 이내로 붙은 void 는 3.3 의 보정이 필요하다.")
    B("라이브러리는 반폭·두께 10 μm, 깊이 10~480 μm, r ≤ 28 μm 범위다. 다른 형상은 분극률 비로 환산할 수 있으나 2 μm 두께·14 μm 반폭 범위에서만 확인했다.")
    B("1차 스터디 도중 baseline 이 추가 void 의 격자 구역을 쓰지 않던 문제를 찾아 고쳤다(t = 0 부터 신호 크기만 한 오차). 이 보고서의 모든 수치는 수정 후의 것이다.")

    H("참고", 1)
    B("ns-TTR void 검출한계 시뮬레이션 결과 보고 (results/detection_map): 검출 기준과 단독 void 깊이·반폭 지도.")
    B("ns-TTR 다중 void 검출 시뮬레이션 결과 보고 (results/multi_void, 1차 스터디).")
    B("중첩 검증 스터디 분석 (results/superposition/analysis.md, analysis.json).")
    B("H. S. Carslaw, J. C. Jaeger, Conduction of Heat in Solids, 2nd ed. (Oxford), Ch. 10 (image method).")
    B("J. A. Stratton, Electromagnetic Theory (McGraw-Hill), §3.27 (depolarisation factors of spheroids).")

    doc.add_page_break()
    H("부록 A. 2차 스터디 케이스 실행 표", 1)
    runs = A["runs"]
    table(["케이스", "계열", "차원", "미지수", "분", "에너지 오차", "평균 반복", "피크 [mK]", "t_peak [μs]", "판정"],
          [[r["id"], r["series"], r["dim"], f"{r['n_unknowns']:,}", f"{r['wall_min']:.1f}", f"{r['energy_error']:.1e}", f"{r['mean_iterations']:.1f}",
            f"{r['peak_mK']:.4g}", f"{r['t_peak_us']:.0f}", r["verdict"]] for r in runs], widths=[2.4, 1.2, 1.0, 1.8, 0.9, 1.7, 1.4, 1.7, 1.6, 2.6], size=7)
    H("부록 B. 쌍 케이스 그림", 1)
    P("각 그림: 온도장 MAP(void 위치 표시)과 평면도, GUI '신호' 페이지의 두 그래프, 지표 표와 void 정보.")
    for cid in [f"P1_g{int(g):02d}" for g in P1_GAPS] + [f"P2_s{int(s):02d}" for s in P2_SPACINGS] + [f"P3_r{r:02d}" for r in (10, 15, 20, 25)]:
        if (FIG_SP / f"{cid}.png").exists():
            H(f"{cid} — {R[cid]['label']}", 3)
            fig(FIG_SP / f"{cid}.png", "", 16.0)
    H("부록 C. 1차 스터디 대표 케이스 그림", 1)
    for cid in ("A_d070_axis", "B_N9", "C_stack3", "D_N6", "F_seed5"):
        if (FIG_MV / f"{cid}.png").exists():
            H(f"{cid} — {R[cid]['label']}", 3)
            fig(FIG_MV / f"{cid}.png", "", 16.0)
    out = OUT / f"{TITLE}.docx"
    doc.save(out)
    return out


if __name__ == "__main__":
    print("docx:", build())
