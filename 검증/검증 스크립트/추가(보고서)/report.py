# -*- coding: utf-8 -*-
"""Build the validation report (.docx) with python-docx."""
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"D:\projects\ns-TTR Simulator\git\검증\ns-TTR 시뮬레이터 검증 보고서.docx"
doc = Document()

# ---------- styles
def set_font(style, name="Malgun Gothic", size=10, bold=None, color=None):
    style.font.name = name; style.font.size = Pt(size)
    rpr = style.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"): rf.set(qn(k), name)
    if bold is not None: style.font.bold = bold
    if color is not None: style.font.color.rgb = RGBColor(*color)
set_font(doc.styles["Normal"], size=10)
doc.styles["Normal"].paragraph_format.space_after = Pt(4)
for lvl, sz in ((1, 15), (2, 12.5), (3, 11)):
    set_font(doc.styles[f"Heading {lvl}"], size=sz, bold=True, color=(0x1F, 0x3A, 0x5F))
set_font(doc.styles["Title"], size=20, bold=True, color=(0x1F, 0x3A, 0x5F))
sec = doc.sections[0]; sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0); sec.top_margin = sec.bottom_margin = Cm(2.0)
TW = 17.0  # text width cm

# ---------- helpers
def add_runs(par, text, size=None):
    """**bold** markup and plain text."""
    for i, chunk in enumerate(re.split(r"(\*\*.+?\*\*)", text)):
        if not chunk: continue
        if chunk.startswith("**"):
            r = par.add_run(chunk[2:-2]); r.bold = True
        else:
            r = par.add_run(chunk)
        if size: r.font.size = Pt(size)
    return par
def H(text, lvl=1): return doc.add_heading(text, level=lvl)
def P(text, size=None, style=None, italic=False):
    par = doc.add_paragraph(style=style); add_runs(par, text, size)
    if italic:
        for r in par.runs: r.italic = True
    return par
def B(text, lvl=0):
    par = doc.add_paragraph(style="List Bullet" if lvl == 0 else "List Bullet 2"); add_runs(par, text); return par
def N(text):
    par = doc.add_paragraph(style="List Number"); add_runs(par, text); return par
def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor); tcPr.append(shd)
def T(header, rows, widths=None, size=8.5, caption=None):
    if caption: P(f"**{caption}**", size=9)
    tbl = doc.add_table(rows=1, cols=len(header)); tbl.style = "Table Grid"; tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    if widths is None: widths = [TW / len(header)] * len(header)
    for i, h in enumerate(header):
        c = tbl.rows[0].cells[i]; c.text = ""; add_runs(c.paragraphs[0], f"**{h}**", size); shade(c, "DCE6F1")
    for row in rows:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; add_runs(cells[i].paragraphs[0], str(v), size)
    for row in tbl.rows:
        for i, c in enumerate(row.cells): c.width = Cm(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl
def FIG(path, caption, width=16.0):
    doc.add_picture(path, width=Cm(width)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    par = P(caption, size=8.5); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
def NOTE(text):
    par = P(text, size=9); par.paragraph_format.left_indent = Cm(0.5)
    for r in par.runs: r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)

# =====================================================================
doc.add_paragraph("ns-TTR 시뮬레이터 검증 보고서", style="Title")
P("**검증 계획서(2026-09-17, 개정 2026-09-18) 기반 실행 결과, 문헌 근거, 입력값(반사율·열반사 계수) 검토, 곡선 비교 가능 항목**", size=11)
P("작성일 2026-09-17, 개정 2026-09-18(계획서 개정판의 §3-G Mao 2024 비교와 §7 권장 순서를 반영) · 대상 코드: git 커밋 0946244 (가우시안 펄스 재정규화 포함) · 실행 환경: src/.venv (Python 3.14.7, numpy 2.5.2, scipy 1.18.1) · "
  "실행 스크립트: 검증/검증 스크립트/v1·v4·v5·v6.py(계획서 첨부) + 검증/검증 스크립트/추가(보고서)/vA·vC2·vF·vG·bw·optics·figs.py(보고서용 추가) + 검증/검증 스크립트/mao/dig·cmp·plot.py(계획서 저자 스크립트, 재실행)", size=9)
P("표기 규칙: 문헌은 [번호]로 인용하고 §9 목록에 제목·판·사용한 파트를 적었다. 문헌에 없는 계산·판독은 **(내 계산)** 또는 **(내 판독)**으로 표시한다. "
  "오차는 특별한 언급이 없으면 기준해(해석해)의 최대값으로 정규화한 값이다.", size=9)

# ---------------------------------------------------------------- 0
H("0. 결론 요약")
N("**계획서 §3의 검증 A–F 를 전부 재실행했고, 계획서에 적힌 수치가 모두 재현되었다.** 기존 pytest 35건도 통과(160 s). "
  "baseline 은 다층 해석해 대비 RMS 0.05–0.09 %, 넓은 void 극한의 void 신호는 해석해와 피크 0.1 %·피크 시각 동일, 유한 슬랩은 RMS 0.1–0.2 %.")
N("**계획서 권장 3항(r_half 스윕)을 새로 실행했다.** 원판(box)·타원체(ellipse) void 모두 r_half 를 40 → 2.5 µm 로 줄일 때 신호 피크가 적층 해석해 상한(16.21 mK)에서 단조 감소하며, "
  "r_half ≥ 2·스폿 반경이면 상한과 0.5 % 이내로 겹친다. 유한 void 신호가 해석적 상한과 연속적으로 이어진다는 간접 검증이다.")
N("**문헌 시나리오 2건(구리 220 ns 어블레이션, Si 27 ns 어닐링)은 문헌 그림과 모순 없다.** Si 는 문헌 Fig. 5(b) 곡선의 800 mJ/cm² 피크(≈1650 K, 내 판독)와 시뮬레이터(1733 K)가 5 % 안에서 맞고, "
  "용융 문턱은 시뮬레이터 774–776 mJ/cm² 대 문헌 850 mJ/cm². 구리는 포스터의 온도 곡선이 27.5 µm 를 반경으로 읽은 경우보다 훨씬 빠르게 가열되므로 스폿 정의가 모호하다(§2.6).")
N("**코드 지적사항 중 §4-1(가우시안 에너지 0.02 % 누락)은 커밋 0946244 에서 해결되었음을 확인했고, §4-2(square 펄스 on/off 순간의 면 온도 보정 아티팩트)는 아직 남아 있다** "
  "(오차 최대 1.2–1.3 %, t = 0 과 펄스 끝 두 지점에서만, Gaussian 펄스에서는 나타나지 않음). 이번 실행에서 VoidSpec.shape 가 잘못된 문자열이어도 오류 없이 box 로 처리되는 점도 확인했다(§3).")
N("**펌프 반사율 R = 0.60 은 수집 자료(McPeak 2015, Rakić 1998)의 깨끗한 구리 값 0.64–0.66 보다 낮고, '자연 산화막 1–3 nm 가 있는 구리'로 해석해야 근거가 선다.** "
  "흡수 깊이는 13 nm 가 아니라 16–17 nm 가 맞다(경고 판정에만 쓰여 결과 무영향). **열반사 계수 −1.5×10⁻⁴ /K 는 수집 자료 어디에도 구리 값이 없어 근거를 확인할 수 없다.** "
  "10⁻⁵–10⁻⁴ /K 범위 안이라는 것만 말할 수 있고, 절대 신호(ΔR/R)의 최대 불확실 요인이다(§4).")
N("**계획서 개정판(09-18) §3-G 의 Mao 2024 Fig. 6 디지타이즈 비교를 저자 스크립트(검증 스크립트/mao)로 재실행해 같은 수치를 얻었다.** Au/SiC 는 논문 피팅 곡선 ÷ 기준해가 12 ns 이후 1.00–1.04(검출기·오실로스코프 대역폭 포함 1.00–1.01, 로그 RMS 1.0 %), Al/사파이어는 1.09–1.11 로 9–10 % 어긋난다. 대역폭(400·500 MHz 1차 저역통과) 효과도 독립 계산으로 계획서와 같은 값(3.2 ns 펄스 피크 −3.4 %·+0.7 ns, 17.24 ns 펄스 −0.13 %)을 얻었다. A–E 가 기대는 기준해가 외부 실측·모델 곡선으로 한 번 확인된 셈이다(§2.7).")
N("**논문 그래프와 직접 곡선 비교**: Mao 2024 Fig. 6 은 위와 같이 완료됐고, Darif & Semmar 2008 Fig. 5(b)(Si)는 같은 축으로 그린 시뮬레이터 곡선을 실었으며(그림 5, 지금 바로 비교 가능), Dillmann 2016 Fig. 1 은 용융 전 구간만 정성 비교가 가능하다(§5).")

# ---------------------------------------------------------------- 1
H("1. 검증 대상·방법")
P("대상은 저장소 src/ 의 ttr_sim 패키지(축대칭 유한체적 + Crank–Nicolson, 표면 flux 열원)와 GUI app.py 이며, 계획서 §1의 구현 요약과 코드가 일치함을 다시 확인했다. "
  "기준해는 계획서가 첨부한 refml.py(임피던스 재귀 Ĝ(k,s) + Gauss–Legendre 240점 Hankel 적분 + 고정 Talbot M=24 역 Laplace + Laser.f′ 합성곱)이며, 이는 [1] 정리 노트 §2.4–2.5(원문 §II.B.1 quadrupole 해의 임피던스 재귀형)와 [2] §II 식 (1)–(4)의 ns-TTR 단일 펄스 형태를 그대로 구현한 것이다.")
T(["항목", "내용"], [
    ["코드 버전", "git main, 커밋 0946244 '가우시안 펄스 재정규화 (t<0 절단 꼬리 보정)'"],
    ["기존 테스트", "pytest -q tests → **35 passed in 160 s** (에너지 보존, 격자·Δt 수렴, C&J 해석해, 2D↔3D, k_void 단조성 등)"],
    ["계획서 스크립트", "v1.py(A, B), v4.py(C), v5.py(D), v6.py(E) — 수정 없이 그대로 실행; mao/dig.py·cmp.py·plot.py(G, 개정판) — 폰트 지정만 바꿔 재실행"],
    ["계획서 개정(09-18)", "§3-G 를 'Mao 2024 Fig. 6 디지타이즈 대 기준해 비교'로 교체, §0-3·§7-5~7·§8 갱신. 저자 스크립트는 git 폴더가 아니라 'ns-TTR Simulator/관련 논문/검증/검증 스크립트/mao' 에 있어 git/검증/검증 스크립트/mao 로 복사해 두었다."],
    ["추가 스크립트", "vA.py(A·E 곡선 저장), vC2.py(C 확장: r_half 스윕, box/ellipse), vF.py(F1 구리 220 ns, F2 Si 27 ns), vG.py(G 기준 곡선), bw.py(검출기 대역폭 효과, 계획서 §3-G 수치 재현), optics.py(§4 광학 계산), figs.py(그림)"],
    ["공통 설정", "구리 k=398, ρc=8960×385 (α=1.154×10⁻⁴); 입사 1 nJ, R=0.6; 펌프 w=10 µm, 프로브 5 µm; Gaussian 펄스(FWHM=τp, 중심 1.5τp); homogeneous_copper=True (별도 표기 없으면)"],
], widths=[3.5, 13.5], caption="표 1. 실행 조건")

# ---------------------------------------------------------------- 2
H("2. 검증 항목별 결과")
H("2.1 A. 다층 해석해 대비 baseline — 통과", 2)
P("**근거**: [1] 정리 노트 §2.4 '전체 스택과 표면 Green 함수' (Ĝ = −D/C), §2.5 '임피던스 재귀', §5 'ns-TTR 단일 펄스 응답으로의 적용'(ΔT(t) = E_abs ∫ f′(t′) S(t−t′) dt′) 및 §6 T1–T2; [2] §II 식 (1)–(3) (Zn 재귀, 반무한 기판은 Z=1/(kγ)). "
  "판정 기준(계획서 제안): RMS < 0.5 %, 피크 < 1 %.")
T(["프리셋", "τp", "Δz", "셀 / 스텝", "RMS", "최대", "피크 차", "계획서 값"], [
    ["0 (d=1 µm)", "17.2 ns", "0.141 µm", "28 143 / 663", "0.048 %", "0.101 %", "−0.035 %", "0.048 / 0.10 / −0.035 %"],
    ["2 (d=5 µm)", "431 ns", "0.707 µm", "7 938 / 663", "0.088 %", "0.204 %", "+0.087 %", "0.088 / 0.20 / +0.087 %"],
], widths=[2.2, 1.6, 1.6, 2.4, 1.6, 1.6, 1.8, 4.2], caption="표 2. A 결과 (v1.py, t_end = 20 τp)")
FIG("fig_A.png", "그림 1. A: 시뮬레이터(실선)와 Laplace–Hankel 다층 해석해(점선). 아래는 피크로 정규화한 차이. 차이의 최대는 상승 구간(피크 직전)에 있고, 이는 Δz = √(Dτp)/10 격자의 2차 이산화 오차 특성과 일치한다.")
P("**의미**: 기존 C&J 해석해 검증(analytic.py, 실공간 적분)과 독립적인 경로(Laplace–Hankel)로 같은 결론. 이 기준해는 C·E·G 로 확장된다.")

H("2.2 B. 펌프·프로브 RMS 반경 불변성 — 통과", 2)
P("**근거**: [1] 정리 노트 §3 '프로브 가중 (2.14)–(2.18)': 프로브 가중 표면 온도는 두 스폿 반경이 아니라 w₀² = (w₁²+w₂²)/2 하나에만 의존 (§6 T5).")
T(["(w, probe) [µm]", "w₁²+w₂² [µm²]", "RMS 차", "최대 차", "피크 차"], [
    ["(10, 5) 기준", "125", "—", "—", "—"],
    ["(5, 10)", "125", "0.0063 %", "0.012 %", "0.0032 %"],
    ["(7.906, 7.906)", "125", "0.0033 %", "0.0066 %", "0.0015 %"],
], widths=[3.5, 3.0, 3.5, 3.5, 3.5], caption="표 3. B 결과 (v1.py, 프리셋 2, t_end = 10 τp; 계획서: 0.012 %, 0.007 %)")
P("**의미**: 열원 환형 적분·프로브 가중·자동 격자가 서로 일관됨을 해석해 없이 확인하는 값싼 회귀 테스트. 실제 형상(실리카 외피)에서는 R_cu 절단 때문에 성립하지 않으므로 균질 모드 전용이다.")

H("2.3 C. 넓은 void 극한 = 구리/공기/구리 적층 — 통과 ★, 그리고 r_half 스윕(신규)", 2)
P("**근거**: [1] 정리 노트 §2.3–2.5 의 다층 임피던스 재귀(층 사이 G = 10¹⁵ 로 온도 연속), §5 마지막 문단('아래에 단열성 층이 있으면 냉각이 느려진다 — 무한히 넓은 void 의 상한 기준'), §6 T6; "
  "[11] Treweek 2024 §'Results' Fig. 2 는 매립 디본드를 '유효 열저항 층(G₂ ↓)'으로 다루는 같은 관점의 FDTR 사례.")
T(["비교량", "시뮬레이터", "해석해", "계획서"], [
    ["void 포함 표면 온도 (RMS / 최대 / 피크)", "0.104 % / 0.221 % / +0.124 %", "—", "0.10 % / 0.22 %"],
    ["void 신호 피크", "16.23 mK", "16.21 mK", "16.23 / 16.21 mK"],
    ["피크 시각", "1035.7 ns", "1035.7 ns", "1035.7 ns"],
    ["void 신호 전 구간 RMS / 피크", "0.15 %", "—", "0.15 %"],
    ["관측창 끝(3.02 µs) 대비 신호/baseline", "4.78", "4.82", "—"],
], widths=[6.0, 4.0, 3.0, 4.0], caption="표 4. C 결과 (v4.py: VoidSpec(shape='box', r_half=40 µm, 깊이 5 µm, 두께 2.5 µm, k=0.026), 11 466 셀)")
P("**r_half 스윕(계획서 권장 3항, vC2.py)**: 같은 깊이·두께에서 r_half = 40, 20, 10, 5, 2.5 µm, 원판(box)과 GUI 기본형인 타원체(ellipse) 두 가지.")
T(["r_half [µm]", "box 피크 [mK]", "box 피크 시각 [ns]", "ellipse 피크 [mK]", "ellipse 피크 시각 [ns]", "최대 대비 (box / ellipse)"], [
    ["40 (= R_cu)", "16.235", "1035.7", "16.215", "1033.6", "4.78 / 4.53"],
    ["20", "16.144", "1031.4", "15.415", "1018.4", "2.74 / 2.37"],
    ["10", "12.135", "951.2", "10.208", "944.7", "0.93 / 0.75"],
    ["5", "4.364", "873.2", "3.148", "875.4", "0.198 / 0.143"],
    ["2.5", "0.985", "840.7", "0.648", "845.1", "0.036 / 0.024"],
    ["적층 해석해(상한)", "16.213", "1035.7", "—", "—", "4.82"],
], widths=[2.6, 2.4, 2.8, 2.6, 3.0, 3.6], caption="표 5. r_half 스윕 (깊이 5 µm, 두께 2.5 µm, 펌프 10 µm, 프로브 5 µm)")
FIG("fig_C.png", "그림 2. C: (좌) 넓은 원판 void 의 신호 대 적층 해석해, (중) r_half 스윕 곡선 — 실선 box, 점선 ellipse, 붉은 점선 해석해 상한, (우) 피크의 단조 수렴.")
P("**의미**: (i) void 를 직렬 저항 면 전도도로 처리하는 방식이 정확하다는 직접 증거. (ii) 유한 void 신호는 r_half 를 줄여 가면 상한에서 단조로 감소하며 시각도 단조로 빨라져, 격자·시간 스텝이 유한 void 에서 비물리적 거동을 만들지 않는다. "
  "(iii) r_half = 20 µm(스폿 반경의 2배)에서 이미 상한의 99.4 %(box)/95 %(ellipse)이므로, '스폿보다 넓은 void 는 적층 해석해로 대체 가능'하다는 실용적 기준을 얻는다. 회귀 테스트로 넣을 만하다(허용: 상한 초과 < 0.5 %, 단조성).")
NOTE("주의: 표 5 의 ellipse 는 VoidSpec 기본값 shape='ellipse' 로 실행한 것이다. 처음에 'ellipsoid' 라고 잘못 적었을 때 코드가 오류 없이 box 로 처리했다(§3 참고).")

H("2.4 D. 깊이–시간 스케일링 — 실행, 스케일 불변 확인", 2)
P("**근거**: [3] ISTFA 2013 §'Resolution and Sensitivity' 4) 식 (5) μ = 2√(αt) (구리 100 ns → 6.3 µm) 및 3) 식 (4) Δt = 0.02x²/α; presets.py 의 τp = 2d²/D. 자기상사 스윕은 모든 길이를 d, 시간을 d²/D 로 잡아 무차원 결과가 같아야 한다.")
T(["d [µm]", "(t_peak − t_c)·D/d²", "피크 ΔT / baseline 피크", "최대 대비", "셀"], [
    ["1", "1.4700", "0.24938", "0.6228", "12 402"], ["2", "1.4700", "0.24938", "0.6228", "11 242"],
    ["4", "1.4700", "0.24938", "0.6228", "9 983"], ["8", "1.4700", "0.24938", "0.6230", "8 008"],
], widths=[2.5, 4.0, 4.0, 3.0, 3.5], caption="표 6. D 자기상사 스윕 (w=3d, 프로브 1.5d, void r_half=2d·두께 0.5d, τp=2d²/D, ellipse, k=0.026) — 계획서와 동일")
T(["d [µm]", "τp", "(t_peak − t_c)·D/d²", "void 피크 [mK / nJ]", "피크 / baseline 피크"], [
    ["1", "17.2 ns", "1.796", "40.1", "0.113"], ["2", "69 ns", "1.606", "35.9", "0.221"], ["5", "431 ns", "1.376", "10.2", "0.220"],
    ["10", "1.72 µs", "1.266", "1.97", "0.132"], ["20", "6.9 µs", "1.656", "0.347", "0.080"], ["40", "27.6 µs", "1.936", "0.0886", "0.071"],
], widths=[2.5, 2.5, 4.0, 4.0, 4.0], caption="표 7. D 실제 형상 스윕 (실리카 외피, w=10 µm 고정, void 크기 ∝ d, t_end = 8 d²/D)")
P("**의미**: 네 자리까지 같은 무차원값 → 자동 격자·시간 스텝이 스케일 불변(회귀 테스트 허용 1e-3 제안). 실제 형상에서 피크 시각 계수는 1.3–1.9 로 코드 진단의 '2d²/D' 는 상한 쪽 근사로 타당하며, [3]의 t = d²/(4α)(계수 0.25)는 열이 도달하기 시작하는 시각이므로 모순이 아니다.")

H("2.5 E. 유한 길이·단열 후면 — 통과", 2)
P("**근거**: [10] Guo 2007 §II.B 식 (2)–(5)의 유한 막대 Green 함수 해(Fo = αt/L² 로 무차원화하면 재료·길이에 무관한 곡선)를, 시뮬레이터가 지원하는 경계(단열 후면·표면 flux)에 맞춰 C&J 형 단열 슬랩 해 "
  "ΔT_s = (qL/k)[Fo + 1/3 − (2/π²)Σ e^{−n²π²Fo}/n²] 로 바꾼 것. 실리카를 k = 10⁻¹² 재질로 두어 측면을 단열하고 w = 1 m 로 균일 flux, square 펄스.")
T(["케이스", "RMS", "최대", "피크 차", "최종 상승 시뮬 / E/(ρcL)", "에너지 오차"], [
    ["τp = 172 µs (침투 ≪ L)", "0.112 %", "1.27 %", "0.008 %", "— (t_end = 3τp 에서 미평형)", "−3.6×10⁻¹⁴"],
    ["τp = 2.76 ms (후면 도달)", "0.167 %", "1.20 %", "0.006 %", "1.47809×10⁻⁴ / 1.47639×10⁻⁴ K (+0.12 %)", "1.6×10⁻¹³"],
], widths=[3.6, 1.6, 1.6, 1.8, 5.4, 3.0], caption="표 8. E 결과 (v6.py; 계획서와 동일)")
FIG("fig_E.png", "그림 3. E: 시뮬레이터와 1D 단열 슬랩 해석해(좌축), 피크 정규화 차이(우축, 녹색). 1.2–1.3 % 스파이크는 t = 0 과 펄스 끝에서만 나타난다(§3 의 §4-2 아티팩트). 오른쪽 후반의 톱니는 성장하는 Δt 블록 경계에서의 보간 잔차로 ±0.2 % 이하.")
P("**의미**: 프리셋 8 에서 GUI 가 경고만 하던 '후면 도달' 영역도 수치적으로 정확하다. 최종 온도가 E/(ρcL) 과 0.12 % 안에서 맞는 것은 에너지 보존과 후면 단열이 함께 맞다는 뜻이다.")

H("2.6 F. 문헌 시나리오 재현 — 문헌과 모순 없음", 2)
P("**F1 근거**: [4] Dillmann 2016 포스터 Table 1 (0.286 mJ, 220 ns, spot size 27.5 µm, peak intensity 3.8×10¹⁶ W/m²), 본문 'Absorption coefficient α = 0.1 for solid', Figure 1 좌상 'Gauss shaped Laser irradiation'과 우하 'Temperaturverlauf und Temperaturgradient über 1.5·τ_Puls'.")
T(["27.5 µm 해석", "흡수 flux q₀", "시뮬 중심 피크", "해석해(C&J 축상)", "1D 극한", "ΔT = 1065 K(용융) 도달"], [
    ["1/e² 반경", "1.09×10¹¹ W/m²", "1439.5 K @ 219 ns", "1442.2 K", "1563 K", "111 ns"],
    ["지름 (w = 13.75 µm)", "4.38×10¹¹ W/m²", "4840.5 K", "4846.2 K", "6253 K", "6.2 ns"],
], widths=[3.0, 2.8, 3.0, 2.8, 2.2, 3.2], caption="표 9. F1 결과 (vF.py, square 220 ns, 흡수율 0.1, 선형 모델; 계획서: 1439 K / 1442 K / 111 ns)")
FIG("fig_F1.png", "그림 4. F1: 선형 모델의 중심 표면 온도(상변화 없음). 포스터 Fig. 1 우하 곡선과 같은 축(0–400 ns). 점선은 C&J 축상 해석해.", 13)
P("**포스터 곡선 판독(내 판독)**: 포스터의 온도 곡선은 약 15 ns 에 1358 K(용융)를 지나 약 40 ns 에 ≈2900 K 로 포화(증발 플래토)한다. 이는 '반경' 해석(용융 111 ns)보다 훨씬 빠르고 '지름' 해석(6 ns)보다는 느리다. "
  "1D 초기 상승 ΔT ∝ q₀√t 로 역산하면 포스터의 유효 흡수 flux 는 ≈3×10¹¹ W/m² (내 계산)로 두 해석의 중간이며, 포스터 Table 1 의 peak intensity 3.8×10¹⁶ W/m² 는 0.286 mJ/220 ns/27.5 µm 와 4자리 이상 어긋난다(단위 오기 가능). "
  "따라서 F1 은 '용융 전 상승 형태·시간 규모가 문헌과 같은 차수'라는 정성 확인에 그친다. 용융 이후는 잠열·증발이 있어 선형 모델과 비교 대상이 아니다.")
P("**F2 근거**: [5] Darif & Semmar 2008 §2 (R = 0.61 solid / 0.73 melt 본문, Table 1 은 0.59 / 0.65; 흡수 길이 6 nm; k = 148, ρ = 2320, c = 710), §3 (시간 0–60 ns, 스텝 2 ns), §4 '멜팅은 gate 형에서 F = 850 mJ/cm² 부터', Figure 5(b) gate 형 표면 온도, Table 1 (E = 3.2×10⁻⁸ J / S = 4×10⁻¹² m² = 800 mJ/cm², Ttrans = 1690 K).")
T(["항목", "시뮬레이터 (1D, w = 1 m)", "1D 해석해", "문헌"], [
    ["표면 피크 상승 (mJ/cm² 당)", "1.7999 K", "1.8032 K", "Fig. 5(b) 800 mJ/cm² 곡선 피크 ≈1650 K → ≈1.70 K/(mJ/cm²) (내 판독)"],
    ["용융 문턱 (ΔT = 1394–1397 K)", "774–776 mJ/cm²", "773 mJ/cm²", "850 mJ/cm² (gate), 1050 (Gaussian)"],
    ["800 mJ/cm² 피크 온도", "1733 K", "1735 K", "≈1650 K (내 판독), 용융 직전"],
], widths=[4.5, 3.5, 2.5, 6.5], caption="표 10. F2 결과 (vF.py; 27 ns gate, R = 0.59; 계획서: 1.799 K, 777 mJ/cm²)")
FIG("fig_F2.png", "그림 5. F2: 시뮬레이터(선형, 상변화 없음)의 Si 표면 온도, [5] Fig. 5(b)와 같은 축(0–60 ns, 200–2000 K)·같은 플루언스. 1690 K 위 부분은 문헌에서는 용융 플래토가 된다.", 13)
P("**의미**: 피크 온도가 5 % 안에서 맞고 문턱은 문헌이 9 % 높다. 문헌 쪽은 부피 열원(6 nm, 무시 가능), 복사·대류(h = 10, 무시 가능), 2 ns 시간 스텝(27 ns 펄스에 대해 거칠어 피크를 낮게 잡음)을 포함하므로 이 차이는 자연스럽다. "
  "문헌 안에서도 R 값(본문 0.61 vs 표 0.59), 문턱 서술이 일관되지 않으므로 더 정밀한 비교는 의미가 없다.")

H("2.7 G. ns-TTR 실측 곡선(Mao 2024 Fig. 6) 대 기준해 — 계획서 개정판(09-18) 재현, Au/SiC 통과", 2)
P("**근거**: [2] §II (펌프 3.2 ns, 355 nm; 프로브 1/e² 반경 3.5 / 8.7 µm, 펌프 62 / 65 µm; 검출기 400 MHz, 오실로스코프 500 MHz), Table I (Au 100 nm 19 300·129·120, Al 80 nm 2 700·897·120, 사파이어 3 980·778, SiC 3 260·690), "
  "§IV.C.1 Fig. 6(a)(b)와 삽입표(Al–Sa TBC 90–94, TC 28.5–30.4; Au–SiC TBC 72.5–74, TC 348.5–349.9); [20] 계획서 개정판 §3-G. 비교 대상은 시뮬레이터가 아니라 A–E 가 쓰는 기준해(refml.py)다.")
P("**방법(계획서 저자 스크립트, 이번에 재실행)**: dig.py 가 Fig. 6 래스터(1280×502 px, CC BY)에서 채도로 피팅 곡선(TR/GO/DL 색선)을, 명도로 측정점(검정) 범위를 뽑고 log 축 눈금 픽셀로 보정한다. "
  "cmp.py 가 refml 의 step 응답에 3.2 ns Gaussian 을 합성곱하고, 선택적으로 400 MHz·500 MHz 1차 저역통과를 차례로 걸며, 논문 시간축의 원점(임의)은 12 ns 이후 로그 RMS 가 최소가 되게 정렬한다. "
  "plot.py 는 겹침 그림을 만든다. 저자 실행값과 이번 재실행값은 소수 셋째 자리까지 같다.")
T(["t (ns, 펄스 중심 기준)", "5", "10", "20", "50", "100", "200", "500", "1000"], [
    ["Al/사파이어 (G 90, K 29.3)", "0.560", "0.299", "0.178", "0.104", "0.072", "0.050", "0.031", "0.022"],
    ["Au/SiC (G 73.6, K 349.9)", "0.549", "0.218", "0.084", "0.042", "0.028", "0.019", "0.011", "0.007"],
], widths=[4.6] + [1.55] * 8, caption="표 11. 기준 곡선 (vG.py, refml 다층 해석해, ΔT/ΔT_max, 대역폭 무제한; 계획서와 동일)")
T(["케이스", "대역폭", "기준해 피크 지연 [ns]", "정렬 t₀ [ns]", "로그 RMS (12 ns 이후)", "논문 피팅 ÷ 기준해 @ 20 / 50 / 100 / 300 / 1000 ns", "계획서 값"], [
    ["Au 100 nm / SiC", "무제한", "1.56", "5.00", "0.030", "1.002 / 1.026 / 1.034 / 1.029 / 1.044", "1.00–1.04"],
    ["Au 100 nm / SiC", "400+500 MHz", "2.30", "3.90", "**0.010**", "1.007 / 1.003 / 1.006 / 1.000 / 1.014", "1.00–1.01, 로그 RMS 1.0 %"],
    ["Al 80 nm / 사파이어", "무제한", "1.50", "8.05", "0.108", "1.022 / 1.107 / 1.109 / 1.123 / 1.113", "20 ns 이후 1.09–1.11(대역폭 포함)"],
    ["Al 80 nm / 사파이어", "400+500 MHz", "2.24", "6.95", "0.082", "1.018 / 1.080 / 1.080 / 1.093 / 1.082", "15 ns 부근 1.05"],
], widths=[2.6, 2.0, 2.0, 1.6, 2.0, 4.4, 2.4], caption="표 12. Fig. 6 디지타이즈 비교 재실행 결과 (mao/cmp.py; 계획서 §3-G 표와 동일)")
FIG("mao/mao_compare_repro.png", "그림 6. G: [2] Fig. 6 의 측정 데이터 범위(회색)·피팅 곡선(주황)과 기준해(녹 점선: 대역폭 무제한, 파랑: 400·500 MHz 저역통과). 아래는 논문 피팅 ÷ 기준해. 계획서 저자의 plot.py 를 이 환경에서 재실행한 그림.", 15.5)
P("**검출기 대역폭 효과(bw.py, 내 계산으로 계획서 값 재확인)**: 1차 저역통과 400 MHz 와 500 MHz 를 차례로 적용.")
T(["곡선", "펄스 FWHM", "피크 높이 변화", "피크 시각 이동", "5·FWHM 시점 변화 (정규화 전)", "계획서 값"], [
    ["Cu 반무한, 10/5 µm (프리셋 0)", "17.24 ns", "−0.13 %", "+0.69 ns", "+0.71 %", "−0.13 % / +0.7 ns / +0.7 %"],
    ["Cu 반무한, 10/5 µm", "3.2 ns", "−3.44 %", "+0.74 ns", "+2.85 %", "−3.4 % / +0.7 ns / +2.8 %"],
    ["Al/사파이어 (Mao)", "3.2 ns", "−2.92 %", "+0.74 ns", "+3.38 %", "—"],
    ["Au/SiC (Mao)", "3.2 ns", "−2.84 %", "+0.74 ns", "+6.49 %", "—"],
], widths=[4.2, 2.0, 2.4, 2.4, 3.4, 2.6], caption="표 13. 측정계 대역폭이 정규화 전 곡선에 주는 영향 (bw.py)")
P("**이 결과가 확인해 주는 것(계획서 §3-G 의 판단에 동의)**: (1) 기준해가 맞게 구현되었다 — 임피던스 재귀·Hankel·Talbot·합성곱을 독립 구현했는데 다른 그룹의 모델 곡선(Au/SiC)과 12 ns–1 µs 전 구간에서 1 % 이내(디지타이즈 오차 1–2 % 수준)로 겹친다. "
  "(2) 시뮬레이터와 기준해가 공유하는 가정(Gaussian 표면 가열, 프로브 가중 표면 온도, 선형 전도, 상수 물성)만으로 실측 ns-TTR 곡선이 수 ns–1 µs 에서 재현된다. "
  "(3) 측정계 응답의 크기를 알게 되었다 — Au/SiC 의 3–4 % 잔차가 대역폭을 넣으면 사라지고, 3.2 ns 펄스에서는 피크가 3.4 % 낮아지고 0.7 ns 늦어진다. 목표 조건(17 ns 이상 펄스, void 피크는 수십 ns–µs)에서는 0.1 % 수준이라 무시할 수 있다.")
P("**Al/사파이어의 9–10 % 어긋남**: 감쇠 모양은 같고 피크 대비 꼬리가 논문 쪽이 높다. 계획서의 민감도 계산(펄스 폭 4 ns → 잔차 0–2 %, Al 두께 100 nm → −3~−5 %, 스폿·피팅값 3종은 영향 미미)으로는 설명되지 않으며, "
  "논문이 이 시편에 실제로 쓴 입력이 본문 값과 다를 가능성이 크나 논문만으로 확정할 수 없다. 따라서 근거로 삼는 것은 Au/SiC 한 건이다.")
P("**말할 수 없는 것**: 시뮬레이터의 수치 구현(FV·CN·격자), void 신호, 유한 반경 원기둥·실리카 외피, 신호 절대값(흡수율·c_tr), Cu 물성·532 nm 광학 입력. 비교는 정규화 곡선이고 시료는 Al·Au 박막이며 펌프 62–65 µm 대 침투 깊이 수 µm 의 사실상 1D 문제다.")
P("**계획서의 두 가지 권고(동의)**: (i) 대역폭 보정은 물리 모델에 고정하지 말고 출력 신호에 거는 선택적 후처리(기본 꺼짐, 입력은 대역폭 MHz 하나)로 둔다 — 장비 고유값이고 A–E 검증은 필터 없는 이상 응답이어야 하므로. "
  "(ii) 시뮬레이터에 z 적층·계면 G·부피 열원을 넣어 Au/SiC 를 직접 재현하는 것은 검증 목적으로는 권하지 않는다 — 그 재현이 시험하는 것은 새로 추가한 코드이고 목표 시뮬레이션은 그중 어느 것도 쓰지 않으며, 공유 부분은 A–E 가 0.05–0.17 % 로 이미 더 엄격히 확인했다. "
  "실제 시편의 라이너/시드층·산화막을 모델에 넣어야 할 때만 적층·G 가 필요해지고, 그때 Au/SiC 가 새 기능의 테스트가 된다.")
NOTE("논문 데이터(github ZZSJ402/DL-TTR, [2] 'Data availability')의 원시 곡선은 이번에도 확인하지 않았다. 확보되면 디지타이즈 오차(1–2 %) 없이 같은 비교를 할 수 있다.")

H("2.8 H·I. 매립 결함 사례와 검출 가능성", 2)
P("[11] Treweek 2024 Fig. 1c·Fig. 2(a): GaN 5 µm 아래 디본드 영역(유효 TBC 4.7–17.8 MW/m²K, §'Results' 위치 2·3)을 3 µm 스폿 FDTR 로 구분. 주파수 영역이라 직접 비교는 불가하지만 C 의 '넓은 void = 열저항층' 관점과 동일하다. "
  "[15] Electronics Cooling 2011 §'Thermoreflectance Imaging': 12-bit CCD 양자화 한계 ΔR/R = 2.44×10⁻⁴, 평균 후 2.5×10⁻⁶; [3] §'Resolution and Sensitivity' 1): 실용 온도 분해능 10–50 mK. "
  "표 7 에 c_tr = −1.5×10⁻⁴ /K 를 곱하면 baseline 피크 10 K 기준 void 차이 신호는 d = 5 µm 에서 |ΔR/R| ≈ 3.3×10⁻⁴, d = 40 µm 에서 ≈ 1.1×10⁻⁴ 로 영상 방식 한계보다 크지만, "
  "(i) ns-TTR 단일 포토다이오드의 잡음은 별개이고 (ii) c_tr 가 10⁻⁵ 규모라면(§4) 여유가 10배 준다.")

# ---------------------------------------------------------------- 3
H("3. 코드에서 확인한 사항")
T(["#", "계획서 지적", "현재 상태 (커밋 0946244)", "근거"], [
    ["4-1", "Gaussian 펄스 에너지 0.0206 % 누락", "**해결됨.** Laser.gauss_norm 이 절단 꼬리 Φ(−t_c/σ) 로 재정규화; f, f_mean 모두 적용. test_energy_conservation 에 abs=0 명시. pytest 통과.", "solver.py Laser.gauss_norm / f / f_mean; git log 0946244"],
    ["4-2", "square 펄스 on/off 순간의 면 온도 보정 q·Δz₀/(2k)·f(tₙ)", "**남아 있음.** record() 가 face_corr * pulse[n](순간값)을 더한다. E 에서 t = 0 오차 +1.27 %(172 µs) / +1.20 %(2.76 ms), 펄스 끝 −0.75 % / −1.01 % 로 정확히 두 지점. Gaussian 은 f 연속이라 무영향. GUI 는 Gaussian 고정.", "solver.py record(): Ts = T2[0] + face_corr * pulse[n] + T0; 그림 3"],
    ["4-3", "VoidSpec.k 기본 1.0 vs GUI 0.026", "변화 없음(문서상 참고). 테스트 make_cfg 도 1.0.", "solver.py VoidSpec, app.py"],
    ["신규", "VoidSpec.shape 문자열 미검증", "shape='ellipsoid' 처럼 'ellipse' 가 아닌 어떤 값이든 조용히 box 로 처리된다(solver.py 547행, solver3d.py 85행의 if/else). 오타가 결과를 바꾸므로 __post_init__ 검증 권장.", "이번 실행에서 box 와 동일한 결과가 나와 발견"],
], widths=[1.0, 4.0, 8.0, 4.0], caption="표 14. 코드 확인 사항")

# ---------------------------------------------------------------- 4
H("4. 펌프 반사율 R 과 열반사 계수 c_tr 의 근거 검토")
P("코드의 정의 위치는 materials.py 의 COPPER_REFLECTIVITY_DEFAULT = 0.60('normal-incidence reflectance of clean Cu near 532 nm (~0.6)'), COPPER_ABSORPTION_DEPTH = 13 nm, COPPER_C_TR_PROBE = −1.5×10⁻⁴ /K('Literature values for Cu near 630 nm are of order −1e-4 to −2e-4 1/K')이다. "
  "구현 스펙 문서(ns-TTR 시뮬레이터 구현 스펙.md §2.2, §2.4)에는 R 의 값·출처가 없고 열반사 계수는 언급이 없다. 즉 두 값 모두 코드 주석 외에 근거 문서가 없어, 수집 자료로 재구성했다.")
H("4.1 펌프 반사율 R(532 nm) 과 흡수 깊이", 2)
P("**근거 자료와 파트**: [6] McPeak 2015 레코드 'Data (tabulated n, k)' 0.52–0.54 µm 행과 '주요 레이저 파장에서의 값' 표; [7] Rakić 1998 BB 레코드 같은 표(0.5216, 0.5324 µm 행); [8] Ordal 1985 레코드 0.517 µm 한 점; "
  "[19] Wikipedia 'Refractive index' 의 α = 4πκ/λ 와 수직 입사 R = [(n−1)²+κ²]/[(n+1)²+κ²]; [9] Barchiesi 2022 §3 결과 요약(30개 시료 평균: 산화막 n² = 8.2+1.0i, 구성 Cu₂O 76 %/CuO 24 %, 박막 Cu n² = −13.3+3.3i vs 벌크 −11.6+1.6i, 632.8 nm).")
T(["데이터셋", "λ", "n", "κ", "R (수직 입사, 벌크)", "1−R", "1/α (흡수 깊이)"], [
    ["McPeak 2015 (열증착 Cu, template-stripped)", "532 nm", "0.945", "2.600", "**0.641**", "0.359", "**16.3 nm**"],
    ["Rakić 1998 (BB 모델 피팅)", "532 nm", "0.808", "2.489", "**0.659**", "0.341", "**17.0 nm**"],
    ["Ordal 1985 (적외선 데이터셋의 끝점)", "517 nm", "1.16", "2.64", "0.601", "0.399", "15.6 nm"],
    ["McPeak 2015", "632.8 nm", "0.109", "3.580", "0.969", "0.031", "14.1 nm"],
    ["Rakić 1998", "632.8 nm", "0.307", "3.435", "0.909", "0.091", "14.7 nm"],
    ["Barchiesi 2022 박막 Cu 평균 (n² = −13.3+3.3i)", "632.8 nm", "0.449", "3.674", "0.885", "0.115", "—"],
], widths=[5.2, 1.6, 1.3, 1.3, 2.8, 1.4, 3.4], caption="표 15. 수집 자료의 구리 광학 상수 (n, κ 는 표 값의 선형 보간; R, 1/α 는 내 계산)")
P("**산화막의 영향(내 계산)**: 공기/산화막/구리 단일막 Fresnel(전달 행렬, 수직 입사). 산화막 굴절률은 [9]의 632.8 nm 값 √(8.2+1.0i) = 2.87+0.17i 를 532 nm 에도 그대로 쓴 경우와 계획서의 가정값 3.1+0.25i 두 가지.")
T(["기판 데이터", "산화막 n", "0 nm", "1 nm", "2 nm", "3 nm", "5 nm", "10 nm"], [
    ["McPeak 2015", "2.87+0.17i", "0.641", "0.623", "0.602", "0.580", "0.531", "0.376"],
    ["McPeak 2015", "3.10+0.25i", "0.641", "0.618", "0.592", "0.564", "0.501", "0.306"],
    ["Rakić 1998", "2.87+0.17i", "0.659", "0.639", "0.618", "0.595", "0.543", "0.379"],
    ["Rakić 1998", "3.10+0.25i", "0.659", "0.634", "0.607", "0.577", "0.510", "0.303"],
], widths=[3.0, 2.4, 1.9, 1.9, 1.9, 1.9, 1.9, 2.1], caption="표 16. R(532 nm) 대 자연 산화막 두께 (내 계산; [9] Appendix B 는 자연 산화막이 수 nm 이고 그레인 크기가 수–80 nm 라고 보고)")
P("**결론**: (1) 깨끗한 구리의 R(532) 는 두 데이터셋이 0.64–0.66 으로 일치한다. 코드 주석의 'clean Cu ~0.6' 은 정확하지 않고, **0.60 은 자연 산화막 약 2 nm(McPeak 기준) 또는 3 nm(Rakić 기준)가 있는 대기 노출 구리에 해당**한다. 산화막 두께를 모르므로 R 의 실용 범위는 0.55–0.66 이다. "
  "(2) 흡수 깊이는 13 nm 가 아니라 **16–17 nm**(두 데이터셋 일치). 이 값은 Δz/δ_abs ≥ 7 경고 판정에만 쓰이므로 결과에는 영향이 없고 문턱이 약간 보수적으로 바뀐다. "
  "(3) R 은 선형 모델에서 진폭 계수(흡수 에너지 ∝ 1−R)일 뿐 시간 파형에는 영향이 없다. R 을 0.60 → 0.64 로 바꾸면 모든 온도가 10 % 낮아진다. "
  "(4) 프로브 632.8 nm 의 R 은 0.89–0.97 로 데이터셋 간 편차가 크며, 신호에 직접 쓰이지 않지만 (1−R) 이 3배까지 달라지므로 프로브 자체 가열을 추정할 때 주의. "
  "(5) [18] AI 보고서의 R(532) = 0.80–0.83 은 표 15 와 맞지 않고, 같은 문서의 '532 nm 에서 강하게 흡수' 서술과도 모순이므로 채택하면 안 된다.")
P("**권고**: materials.py 주석과 GUI 도움말을 '깨끗한 Cu 0.64–0.66 (McPeak 2015 / Rakić 1998), 자연 산화막 1–3 nm 시 0.58–0.62; 기본 0.60' 으로 고치고 흡수 깊이를 16 nm 로 바꾼다. 절대 온도가 중요하면 실험에서 같은 시료의 반사율을 직접 재서 넣는다.")

H("4.2 열반사 계수 c_tr = (dR/dT)/R (632.8 nm)", 2)
P("**수집 자료 전수에서 구리의 열반사 계수 값은 나오지 않는다.** 확인한 파트는 다음과 같다.")
T(["자료", "확인한 파트", "내용", "구리 값?"], [
    ["[12] Wilson 2012", "§1 서론(refs 8–15), §3 Fig. 1–2, §4 결론", "15개 벌크 금속의 dR/dT 를 1.03 µm 에서 측정 — 대상: Al, Au, Bi, Cr, Mo, Nb, Pd, Pt, Re, Rh, Ru, Ta, Ti, V, W, Zr (Fig. 2 라벨). Ta −1.4×10⁻⁴, Ru +1.5×10⁻⁴, Al −6.4×10⁻⁵ /K. 불확도 ~20 %(스폿 크기), Au·Rh 60–80 %. 구리는 없음. 1차 출처로 Rosei & Lynch, PRB 5, 3883 (1972) 'Thermomodulation Spectra of Al, Au, and Cu' 를 인용(ref 9).", "없음"],
    ["[1] Jiang 2018 정리 노트", "§1 기법 개요", "'dR/dT ~ 10⁻⁴ K⁻¹', 선형성은 ΔT ≲ 10 K 에서 성립.", "없음(일반)"],
    ["[14] Yazawa 2013 (EC)", "Introduction; Material Temperature; Illumination Wavelength", "κ 는 10⁻² ~ 10⁻⁵ /K 범위, 조명 파장에 매우 강하게 의존(Au: 470 nm 양의 피크, 500 nm 에서 0, 520 nm 음의 피크). 구리 마이크로 via 는 200 °C 범위에서 κ 가 2.7 % 만 변함(상수 c_tr 가정의 근거). 재료별 곡선(Fig. 2, Raad et al. 2008)과 권장 파장 표(Table 1)는 수집본에 그림이 빠져 있어 읽을 수 없음.", "구리: 온도 의존성만"],
    ["[15] Yazawa 2011 (EC)", "§'Thermoreflectance Imaging'", "계수는 대부분 재료에서 10⁻⁴ ~ 10⁻⁵ /K, 파장·재료·표면 상태 의존, in-situ 보정 필요.", "없음"],
    ["[13] Radue 2018", "서론; 'Experimental Measurement' (80 nm Au/사파이어 보정)", "dR/dT 는 통상 10⁻⁶ ~ 10⁻⁴ /K; Au 는 3.2–3.7×10⁻⁵ /K(펌프 출력에 따라). ps 이후 T_e ≈ T_p 이므로 ns 에서는 평형 계수 사용이 타당.", "없음(Au)"],
    ["[18] AI 보고서", "광학 상수 표", "C_TR = −1.5 ~ −2.0×10⁻⁵ /K, 출처 [cite: 15]는 1차 측정이 아님. 같은 표의 R(532) 도 틀림(4.1).", "10⁻⁵ 주장(근거 약함)"],
    ["[9] Barchiesi 2022", "§3 결과 요약", "632.8 nm 에서 박막 Cu 의 ε″ 이 벌크의 2배 — 표면 상태에 따른 광학 상수 변동의 크기 감각.", "없음(계수 아님)"],
], widths=[2.6, 3.6, 8.8, 2.0], caption="표 17. 열반사 계수 관련 자료 점검")
P("**물리적 판단(내 판단)**: 632.8 nm(1.96 eV)는 구리 d-밴드 → 페르미 준위 간대역 천이 문턱(≈2.1 eV, 표 15 에서 n, κ 가 0.55–0.60 µm 사이에서 급변하는 것으로도 보임) 바로 아래다. "
  "[14]가 보이듯 간대역 천이 근처에서는 계수가 파장에 따라 부호까지 바뀌며 크기가 10⁻⁴ 규모에 이르므로, 코드의 −1.5×10⁻⁴ 은 '그럴듯한 규모'이지만 수집 자료로는 부호도 크기도 확정할 수 없다. "
  "[18]의 10⁻⁵ 주장을 배제할 근거도 없다. 따라서 **c_tr 의 불확도는 최소 10배(10⁻⁵ ~ 2×10⁻⁴ /K)로 두어야 하며, 이는 ΔR/R 절대값의 지배적 불확실 요인**이다.")
P("**영향 범위**: c_tr 는 열 계산에 전혀 들어가지 않는다(app.py 의 ΔR/R 지표와 신호 탭 표시에만 곱해짐). 따라서 §2 의 열 해석 검증 결과는 c_tr 와 무관하며, 검출 가능성 판단(§2.8)만 c_tr 에 비례해 달라진다.")
P("**권고**: (1) 1차 출처 Rosei & Lynch, Phys. Rev. B 5, 3883 (1972)와 그 후속(예: Tessier et al., APL 78, 2267 (2001) — [15] 참고문헌 [7])을 확보해 632.8 nm 구리 값을 확정한다. "
  "(2) 확정 전에는 GUI 도움말에 '문헌 미확인, 10⁻⁵–10⁻⁴ 범위'를 명시하고 c_tr 를 범위 입력(또는 ×10 스케일 토글)으로 두어 검출 여유를 두 값에서 같이 보여 준다. "
  "(3) 실험에서는 온도 조절 스테이지 위의 기준 시료로 ΔR/R 대 ΔT 를 직접 보정한다([14] 'Material Properties' 의 보정 절차).")

# ---------------------------------------------------------------- 5
H("5. 논문 그래프 곡선과 직접 비교할 수 있는 항목")
P("void 가 없는 대상이라도 '시간 곡선(또는 깊이 분포)'이 그림으로 실려 있고 조건을 재현할 수 있는 것을 모두 골랐다. 시뮬레이터 쪽 곡선은 원 그림과 같은 축으로 그려 두었으므로(그림 5, 4, 6) 원 그림을 디지타이즈하거나 겹쳐 놓고 직접 비교하면 된다.")
T(["문헌·그림", "비교할 곡선", "시뮬레이터 설정", "상태·비고"], [
    ["[5] Darif & Semmar 2008 **Fig. 5(b)** (gate 형, F = 800–1000 mJ/cm²) — **1순위**", "Si 표면 온도 vs 시간 (0–60 ns). 용융(1690 K) 아래 구간 전부 비교 가능; 800 mJ/cm² 곡선은 거의 전 구간.", "vF.py F2 (Si, w = 1 m, square 27 ns, R = 0.59, homogeneous_copper=True, copper=SI). 그림 5.", "지금 가능. 내 판독으로 피크 5 % 이내. 문헌 시간 스텝 2 ns 임을 감안."],
    ["[5] **Fig. 6(b)** (t = t_max 의 깊이 프로파일)", "온도 vs 깊이 (0–7 µm)", "같은 실행의 T_axis (store_fields=True 로 스냅샷)", "가능 — 이번에는 표면만 저장했으므로 추가 실행 필요."],
    ["[4] Dillmann 2016 **Fig. 1 우하** (Temperaturverlauf, 0–380 ns)", "구리 표면 온도 vs 시간. 용융(1358 K) 이전 ~15 ns 구간만.", "vF.py F1. 그림 4.", "정성 비교만. 스폿 정의(반경/지름)·peak intensity 가 모호(§2.6)."],
    ["[2] Mao 2024 **Fig. 6(a)** Al/사파이어, **Fig. 6(b)** Au/Ti–SiC", "정규화 ΔT vs t (log–log, 5–1000 ns), 실험점 + 피팅 곡선", "시뮬레이터 불가(z 적층·G·부피 열원 필요). 기준해(refml)로 비교 — 계획서 §3-G, 이번에 재실행(§2.7, 그림 6).", "**완료**: Au/SiC 1 % 이내, Al/사파이어 9–10 %. Fig. 9(GaN 적층)는 층이 많고 G 가 피팅값이라 미실시."],
    ["[17] COMSOL 블로그 **'1 ns' 격자 온도–깊이 그림**", "Au 200 nm 막, 흡수 플루언스 1 J/m², Gaussian 1 ns. t/t_p = −1, −0.5, 0, 0.5 의 온도 vs 깊이(300–302 K).", "Geometry(L = 200 nm) + 금 물성 + w = 1 m + Gaussian 1 ns 로 가능. 최종 ΔT ≈ Φ/(ρc·L) ≈ 2 K.", "가능(저우선). 흡수 깊이 조건(Δz ≫ 15 nm) 경고 예상; 값은 그림에서 판독해야 함."],
    ["[15] Electronics Cooling 2011 **Fig. 1** (via 위 금속 온도, 50 ns 펄스, 0–200 ns)", "온도 상승 vs 지연 (두 via 크기)", "불가 — 줄 발열·via 주변 층 구조 미상.", "정성 참고만(온도 규모 5–15 K, 100 ns 안에 식음)."],
    ["[11] Treweek 2024 Fig. 2(a), Fig. 4", "FDTR 위상 vs 주파수", "불가 — 주기 가열 모드 없음.", "—"],
    ["[3] ISTFA 2013 Fig. 7–8", "ms 시간열 열영상", "불가 — 소자 발열.", "—"],
    ["[10] Guo 2007 Fig. 2 (T* vs Fo)", "정규화 온도 vs Fo", "현 기능으로 불가(등온 끝단 + 부피 열원). §2.5 의 단열 슬랩으로 대체함.", "Dirichlet 경계·부피 열원 추가 시 가능."],
], widths=[4.6, 4.2, 4.4, 3.8], caption="표 18. 곡선 비교 후보")
P("**비교 시 유의점**: (1) 문헌 그림은 절대 온도 K 축이므로 시뮬레이터 ΔT 에 초기온도(293 K)를 더한다. (2) 문헌은 대부분 부피 열원(Beer–Lambert)을 쓰지만 흡수 깊이가 열 확산 길이보다 훨씬 짧으면(Si 6 nm ≪ √(αt) ≈ 1.5 µm) 표면 flux 와 차이가 없다. "
  "(3) Darif Fig. 5(b)의 800 mJ/cm² 곡선은 피크가 용융점 바로 아래라 상변화 영향이 없고, 850 이상은 피크 부근이 플래토로 잘린다.")

# ---------------------------------------------------------------- 6
H("6. 남은 공백과 권장 순서 (계획서 개정판 §7 기준)")
P("1) §4-2 수정: record() 의 face_corr 보정에 순간값 f(tₙ) 대신 스텝 평균 f̄ 를 쓰거나 반-셀 과도 보정으로 교체 → E 의 최대 오차 1.2 % 가 0.2 % 아래로 내려갈 것으로 예상. (§4-1 은 완료됨.)").paragraph_format.left_indent = Cm(0.5)
P("2) VoidSpec.shape 검증(__post_init__ 에서 {'ellipse','box'} 외 값이면 ValueError) — 이번에 새로 발견.").paragraph_format.left_indent = Cm(0.5)
P("3) 기준해 모듈 ttr_sim/analytic_multilayer.py 추가 + pytest: A(프리셋 0·2), B, C, C-r_half 단조성(이번 실행으로 기준값 확보), D 자기상사, E. 스크립트가 그대로 초안이며 전체 3–4분. [계획서 §7-2, 7-3]").paragraph_format.left_indent = Cm(0.5)
P("4) 입력값 정리: 흡수 깊이 16 nm, R 도움말 문구('자연 산화막 포함'), c_tr 범위 입력·'미확인' 표시. [계획서 §7-4]").paragraph_format.left_indent = Cm(0.5)
P("5) 측정계 응답(대역폭) 후처리 옵션 — 신호에 1차 저역통과(또는 실측 임펄스 응답) 컨볼루션, 입력은 대역폭 MHz, 기본 꺼짐, 실측 비교 때만 사용. [계획서 §7-5, §2.7]").paragraph_format.left_indent = Cm(0.5)
P("6) 유한 void 의 독립 교차: 같은 형상(원기둥·실리카·void) 한 케이스를 COMSOL 등 다른 FEM 으로 풀어 비교하고, 격자(dz, dr, Fo 절반)·2D↔3D 에서 void 차이 신호의 변화량을 확인. [계획서 §7-6]").paragraph_format.left_indent = Cm(0.5)
P("7) (보류) z 적층 + 계면 G + 부피 열원 — 검증 목적으로는 권하지 않음(§2.7). 실제 TGV 라이너(Ti 60 nm / 시드 Cu 1 µm, Wang 2026 크리프 논문의 정보)를 모델에 넣어야 할 때만. [계획서 §7-7]").paragraph_format.left_indent = Cm(0.5)
P("8) Rosei & Lynch 1972 등 1차 출처로 c_tr 확정; 실험 시 기준 시료 보정(§4.2).").paragraph_format.left_indent = Cm(0.5)
P("**유한 크기 void 신호를 독립적으로 검증할 실험 자료는 여전히 없다.** 이번 r_half 스윕으로 '해석적 상한과의 연속성'은 확보했으므로, 남은 것은 격자 수렴·2D↔3D 교차와 실험 기준 시료다.")

# ---------------------------------------------------------------- 7
H("7. 이 검토의 한계")
B("디지타이즈한 문헌 곡선은 Mao 2024 Fig. 6 한 건(계획서 저자 스크립트, 오차 1–2 %, 측정점은 피팅 곡선에 가려 범위로만 추출)이다. Darif Fig. 5(b)·Dillmann Fig. 1 은 눈으로 판독했다(±3 % 수준). 표 18 의 1순위 항목은 같은 방식으로 디지타이즈하는 것을 권한다.")
B("[14]·[15] 두 건은 이미지 PDF 를 렌더링해 읽었고, [14]의 Fig. 2 와 Table 1 은 수집본에서 그림이 빠져 있어 내용을 확인하지 못했다.")
B("[1]은 원문이 아니라 NotebookLM 저장본을 정리한 노트다. 식 번호는 노트가 병기한 원문 번호를 따랐다.")
B("solver3d.py(3D 경로)는 이번 실행 검증에 포함하지 않았다(기존 pytest 의 2D↔3D 테스트 통과에 의존).")
B("웹 검색으로 구리 열반사 계수의 1차 값을 찾으려 했으나 수집 자료 밖의 수치는 확인하지 못해 보고서에 넣지 않았다.")

# ---------------------------------------------------------------- 8
H("8. 부록 — 실행 로그 원문")
LOG = open("run_all.log", encoding="utf-8", errors="replace").read()
LOG2 = open("run_extra.log", encoding="utf-8", errors="replace").read() + "\n##### vC2 (ellipse)\n" + open("run_c2e.log", encoding="utf-8", errors="replace").read()
for line in (LOG + "\n" + LOG2).splitlines():
    if not line.strip() or "RuntimeWarning" in line or "Z=Z+1" in line: continue
    par = doc.add_paragraph(); r = par.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(7)
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas"); par.paragraph_format.space_after = Pt(0)
for fn, title in (("mao/cmp.log", "##### mao/cmp.py (계획서 저자 스크립트 재실행)"), ("bw.log", "##### bw.py (대역폭 효과)")):
    for line in [title] + open(fn, encoding="utf-8", errors="replace").read().splitlines():
        if not line.strip() or "RuntimeWarning" in line or "Z=Z+1" in line: continue
        par = doc.add_paragraph(); r = par.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(7)
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas"); par.paragraph_format.space_after = Pt(0)
P("pytest: 35 passed in 160.28 s (tests/test_solver.py).", size=8)

# ---------------------------------------------------------------- 9
H("9. 참고 자료 (수집 폴더 '검증/관련논문', 사용한 파트)")
refs = [
    "[1] P. Jiang, X. Qian, R. Yang, 'Tutorial: Time-domain thermoreflectance (TDTR) for thermal property characterization of bulk and thin film materials', J. Appl. Phys. 124, 161103 (2018) — 수집본: 'tutorial_TR 노트.pdf'(정리 노트). 사용: §1(dR/dT ~10⁻⁴, ns-TTR 과의 차이), §2.3–2.5(층·계면 행렬, 표면 Green 함수, 임피던스 재귀), §3(프로브 가중 w₀), §5(단일 펄스 응답, 기준 곡선), §6(T1–T7).",
    "[2] Y. Mao et al., 'Deep learning-based data processing method for transient thermoreflectance measurements', J. Appl. Phys. 135, 095102 (2024) — 'deep_learning_TR_restored.pdf'. 사용: §II(장치·스폿·식 (1)–(4)), Table I, §IV.C.1 및 Fig. 6(a)(b) 삽입표(디지타이즈 대상, CC BY 4.0), Fig. 9, Table III, Data Availability.",
    "[3] K. Yazawa, D. Kendig, A. Shakouri, 'Time-Resolved Thermoreflectance Imaging for Thermal Testing and Analysis', ISTFA 2013 — 'thermalreflectance_simulation_paper1.pdf'. 사용: §1 Physics of Thermoreflectance, 'Resolution and Sensitivity' 1)–4) (식 (4), 식 (5) μ = 2√(αt), 구리 100 ns → 6.3 µm).",
    "[4] M. Dillmann, B. Braun, M. Kottcke, 'Investigation of Ablation of a Copper Surface Caused By 220 Nanosecond Laser Pulse' / 포스터 'Nanosecond Laser Ablation of a Copper Surface', COMSOL Conf. Munich 2016 — 사용: Table 1, 'Simulation and Results' 글머리(흡수율 0.1/0.2/0.4), Figure 1(Gauss 조사 분포, 온도 곡선).",
    "[5] M. Darif, N. Semmar, 'Numerical Simulation of Si Nanosecond Laser Annealing by COMSOL Multiphysics', COMSOL Conf. Hannover 2008 — 사용: §2 지배식(반사율, 흡수 길이), §3 COMSOL 구현(시간 스텝), §4 결과(문턱 850/1050), Fig. 5(b), Fig. 6, Fig. 8–9, Table 1.",
    "[6] refractiveindex.info, Cu — McPeak et al. 2015 (ACS Photonics 2, 326) 레코드 — 사용: 'Data (tabulated n, k)', '주요 레이저 파장에서의 값', Comments(열증착, template-stripped).",
    "[7] refractiveindex.info, METALS – Copper (= Cu, Rakić et al. 1998 Brendel–Bormann, Appl. Opt. 37, 5271) 레코드 — 사용: 데이터 표, 주요 파장 계산표, 517 nm 기본 표시값(R = 0.6097).",
    "[8] refractiveindex.info, Cu — Ordal et al. 1985 (Appl. Opt. 24, 4493) 레코드 — 사용: 0.517 µm 한 점(R = 0.6012)과 '1 µm 아래 보간 부적합' 주의.",
    "[9] D. Barchiesi et al., 'Performance of Surface Plasmon Resonance Sensors Using Copper–Copper Oxide Films', Photonics 9, 104 (2022) — 사용: §3 결과 요약(Tables A3–A5 평균: Cu n² = −13.3+3.3i, 산화막 8.2+1.0i, Cu₂O 76 %/CuO 24 %), Appendix B(산화막·그레인), §'Discussion' 632.8 nm 유전율.",
    "[10] J. Guo, X. Wang, T. Wang, 'Thermal Characterization of Microscale Conductive and Nonconductive Wires Using Transient Electrothermal Technique', J. Appl. Phys. 101, 063537 (2007) — 사용: §II.B 식 (2)–(5)(유한 막대 Green 함수, T*, Fo 무차원화).",
    "[11] B. Treweek et al. (Sandia), 'Inversion for Thermal Properties with Frequency Domain Thermoreflectance' — 사용: Fig. 1(b)(c), Fig. 2(a)–(c), 'Results'(FEM 스택 130 nm Au / 5 µm GaN / 100 nm 미지층 / 100 µm 다이아몬드, TBC 4.72–17.84 MW/m²K), Fig. 4.",
    "[12] R. B. Wilson, B. A. Apgar, L. W. Martin, D. G. Cahill, 'Thermoreflectance of metal transducers for optical pump-probe studies of thermal properties', Opt. Express 20, 28829 (2012) — 사용: §1 서론(정성 측정 refs 8–11, ref 9 Rosei & Lynch 1972), §3 Fig. 1–2(15개 금속 1.03 µm 값, 불확도), Fig. 5 논의((1/R)dR/dT 기준), §4 결론.",
    "[13] E. L. Radue et al., 'Hot Electron Thermoreflectance Coefficient of Gold' (ACS Photonics 2018) — 'Au_coefficient_paper_restored.pdf'. 사용: 서론(10⁻⁶–10⁻⁴ /K, T_e ≈ T_p 시점), 'Experimental Measurement of dR/dTe'(80 nm Au/사파이어 3.2–3.7×10⁻⁵ /K).",
    "[14] K. Yazawa, D. Kendig, P. E. Raad, P. L. Komarov, A. Shakouri, 'Understanding the Thermoreflectance Coefficient for High Resolution Thermal Imaging of Microelectronic Devices', Electronics Cooling, 2013-03-08 (이미지 PDF) — 사용: Introduction(ΔR/R = κΔT, 10⁻²–10⁻⁵), Material Temperature(구리 via 2.7 %), Material Properties(표면 처리·산화), Illumination Wavelength(Fig. 2 설명), Microscope NA, References [3][4].",
    "[15] K. Yazawa, A. Shakouri, 'Ultrafast Submicron Thermoreflectance Imaging', Electronics Cooling, 2011-03-01 (이미지 PDF) + '정리 노트' — 사용: §'Thermoreflectance Imaging'(10⁻⁴–10⁻⁵, 양자화 한계·평균), Fig. 1(via 과도 곡선), 정리 노트 §4–5.",
    "[16] Stanford NanoHeat Lab, 'Nanosecond Transient Thermoreflectance' (웹 페이지) — 사용: 본문(6 ns Nd:YAG, 스폿 ~3 mm, 1D 다층 해석, 다중 파라미터 피팅), 장치 그림(532 nm 펌프, 658 nm CW 프로브).",
    "[17] V. Paasonen, 'Modeling Ultrafast Heat Transfer with COMSOL Multiphysics', COMSOL Blog, 2026-01-22 — 사용: 'Beyond Fourier's Law', 'Implementing…'(Au 200 nm, Φ_p = 1 J/m², t_p = 1 ns…), 'Results and Discussion'(1 ns 에서 네 모델 동일 → Fourier 충분).",
    "[18] '구리 마이크로필라의 나노초 과도 열반사 시뮬레이션을 위한 다물리적 및 광열 해석 학술 보고서' (AI 생성 보고서) — 사용: 광학 상수 표(R 0.80–0.83, C_TR −1.5~−2.0×10⁻⁵ 주장) — 오류 지적용.",
    "[19] Wikipedia, 'Refractive index' — 사용: 복소 굴절률, α = 4πκ/λ, 수직 입사 반사율 식.",
    "[20] ns-TTR 시뮬레이터 검증 계획서 (2026-09-17, 개정 2026-09-18: §3-G Mao 2024 비교 추가) — §0–§8 전부; 첨부 스크립트 refml.py, v1/v4/v5/v6.py, mao/dig.py·cmp.py·plot.py·dig.npz·mao_fig6.jpg.",
    "(사용하지 않음, 계획서 D 등급 확인) Ramu & Bowers 강화 Fourier FDTR; Pérez-Barrera 2019 경사 재료 엔트로피; Wang 2026 구리 마이크로필라 크리프(치수 50 × 300 µm, Ti/시드 Cu 라이너 정보만); COMSOL Si 웨이퍼 레이저 가열 ×2(+.mph); MATLAB 이방성 판 예제; Barchiesi 2014 Au·Ag 광학상수 피팅; Tao 2020 귀금속 박막 SPR; Brimhall 2009(EUV); globalsino 침투 깊이; refractiveindex.info DB 논문.",
]
for r in refs:
    par = P(r, size=8.5); par.paragraph_format.left_indent = Cm(0.8); par.paragraph_format.first_line_indent = Cm(-0.8)

doc.save(OUT); print("saved", OUT)
