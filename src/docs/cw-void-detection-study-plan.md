# CW 모드 void 검출 가능성 스터디 — 실행 지시서 (새 세션용)

작성 2026-10-26. 이 파일 하나로 스터디를 시작할 수 있도록 결정 사항·케이스·판정 기준·코드 진입점·주의점을 모았다.
스터디 원안은 사용자가 준 지시(아래 §1)이고, §2 이후는 검토 끝에 합의된 내용이다.

새 세션 첫 메시지 예: `docs/cw-void-detection-study-plan.md 를 읽고 1단계를 시작해.`

---

## 0. 세션 규칙

- **커밋하지 않는다.** 변경·결과 파일은 작업 트리에 두고 사용자가 직접 커밋한다 (메모리 `commit-policy`).
- `app.py`(반사형 GUI)는 건드리지 않는다. `app_cw.py` 는 이 스터디에서 바꿀 일이 없다.
- 실행은 `src` 에서 `.venv\Scripts\python.exe` (시스템 python 에는 numpy 가 없다). 콘솔은 cp949 → 스크립트 첫머리에
  `sys.stdout.reconfigure(encoding="utf-8")`.
- 긴 계산은 백그라운드로 돌리고, 케이스별 결과를 파일로 저장해 재실행 없이 보고서를 다시 만들 수 있게 한다.
- 전체 pytest 는 ~11 분. 솔버 코드를 바꾸지 않는 한 돌릴 필요 없다.

## 1. 목적 (사용자 지시 원문 요지)

CW 투과/반사 측정으로 **void 유무를 판단할 수 있는가**. void 의 위치·크기·형태(·개수)에 따라 "어디까지 검출 가능한가"를
케이스별 결과값과 검출 한계로 정리한다. 이론 설명은 최소화.

용어: 깊이 z = 펌프면(z=0) 기준 void **중심** 깊이. 축 이탈 거리 r_off = 비아 축에서 void 중심까지 거리. 이탈 방위각 θ.
"납품값" 이라는 말은 쓰지 않는다 ("최종 산출값" 또는 그냥 k_eff).

## 2. 범위와 순서 (합의)

**단일 void 먼저, 다중 void 는 별도 스터디로 나중에.**

- **1단계 (2-D, 축 위 단일 void)** — 지금 할 일. 2-D 케이스는 각 2 초. 끝나면 **중간 보고서**.
- **2단계 (3-D, 축 밖 단일 void)** — 중간 보고 뒤 사용자가 범위를 정한다. 기본 설정(20 ms 창, Δz 2.5 µm)에서
  축 밖 void 한 케이스가 **45–60 분** 걸린다 (2 ms 창에서 38 분 실측). 축소 설정은 쓰지 않기로 했다.
  먼저 한 케이스를 돌려 시간을 재고 보고한 뒤 진행.
- **다중 void (N 분할, 링/적층/무작위 배치, 시드 3개)** — 이 스터디에서 제외. 나중에 같은 틀로.

## 3. 고정 조건 (모든 케이스 동일, 보고서에 한 번 명시)

GUI 기본값 그대로: 입사 10 mW, R = 0.6 (흡수 4 mW), 플랫 스팟 등가 반경 5.48 µm, 프로브 지름 8 µm 양면, L = 500 µm,
R_cu = 40 µm, 외곽 경계 far, 관측창 20 ms, Δz 2.5 µm, Δt 성장률 1.05, Lock-in 50 Hz–2 kHz 로그 9점,
t* = 20 ms (관측창의 끝), 스텝 절편 = 최대 기울기 접선(피팅 구간 없음), void 충전 k = 0.026 (공기).
3-D 케이스도 같은 설정. 3-D 의 Lock-in 데이터는 스텝 응답 FFT 경로만 있다 (주파수 영역 직접 풀이는 축대칭 전용);
20 ms 창이면 50 Hz 부터 분해되므로 같은 9점을 쓴다. **3-D 케이스의 기준(void 없는) k_eff 도 FFT 경로로 계산**해서
경로 차이(0.05 %)가 Δk/k 에 섞이지 않게 한다.

코드로 쓰면:
```python
from ttr_sim.transmission import cw_config, run_step_response, frequency_response, lockin_dataset
cfg = cw_config(void=VoidSpec(...), f_min=50.0)        # 그 외 기본값 = GUI 기본값
resp = run_step_response(cfg)                            # 2-D 또는 (needs_3d 면) 3-D 자동
fr = frequency_response(cfg, freqs)                      # 2-D 만; 3-D 는 NotImplementedError → fr=None
ds = lockin_dataset(resp, fr, freqs)                     # 1/창 미만 직접 풀이, 이상 스텝 FFT; fr=None 이면 FFT 만
```
`freqs = np.logspace(log10(50), log10(2000), 9)`. 기준 via 는 `baseline_config(cfg)` (같은 격자).

## 4. 케이스마다 기록할 결과값

1. k_eff (위상 기반 유리 껍질 TLM 역산, 반대면): `analysis.invert_tlm(ds.freqs, ds.H_far, rho_cp, L, A, shell=(R_cu, FUSED_SILICA))`
   → void 없는 via 대비 Δk/k [%].
2. 시간 영역 void 신호 (void − 기준): 펌프면 / 반대면 / 양면 차 각각 최대 상대 변화 [%] 와 그때의 절대값 [mK], 그리고
   t* 에서의 값.
3. Lock-in void 신호 (void − 기준): 주파수별 위상 변화 [°] 와 진폭 변화 [%], 펌프면 / 반대면. 최대값과 그 주파수.
4. void 지표 (`analysis.analyze(...)` + `void_indicators`): t_int, t½, R_th(t*), 펌프면 ΔT(t*) — 각각 Δ [%].
5. 보조: ΔT_far/ΔT_pump 비율의 변화, 에너지 보존 오차, 계산 시간, 2-D/3-D, 미지수 수.

저장: `results/cw_void_detect/cases/<id>.json` (수치) + `<id>.npz` (t, dT_pump, dT_far, freqs, H_far, H_pump, 기준 동일),
`results/cw_void_detect/cases.csv` (한 줄 = 한 케이스), `run.log`. 끝난 케이스는 다시 돌리지 않는다 (캐시).

## 5. 프로브 구성별 분류 (계산은 한 번)

- 펌프면 프로브: 펌프면 ΔT 변화, 펌프면 위상·진폭 변화
- 반대면 프로브: k_eff, t_int, t½, 반대면 위상·진폭 변화, 반대면 ΔT 변화
- 양면 프로브: 양면 온도차 R_th(t*), ΔT_far/ΔT_pump 비율

## 6. 검출 판정 기준 (보고서 앞에 한 번 명시; 가정은 가정이라고 쓴다)

| 지표 | 기준 | 가정 |
|---|---|---|
| k_eff | \|Δk/k\| ≥ 3 σ_k. σ_k 는 `analysis.phase_noise_sensitivity` 로 **같은 주파수 세트·같은 데이터 경로**에서 σ_φ = 0.5° 일 때 계산 (2-D 직접 풀이: 0.3 %; 3-D FFT 경로: 다시 계산해 사용) | σ_φ = 0.5° 가정 |
| Lock-in 위상 변화 | 최적 주파수에서 \|Δφ\| ≥ 3 σ_φ, 그 주파수의 절대 진폭이 잡음 바닥 위 | σ_φ 가정 |
| Lock-in 진폭 변화 | ≥ 3 × (비아 간 흡수 편차) | **흡수 편차 1 % 는 근거 없는 가정** — 표에 "R 의존" 표시 |
| ΔT 변화 (펌프면·반대면·양면 차·R_th) | 평균 후 잡음의 3배. 잡음: ΔR/R 1e-6/transient → 6.7 mK (c_tr = 1.5e-4), N 회 평균 시 /√N | 반사형 보고의 잡음값 준용, "R 의존" |
| t_int, t½ | 같은 ΔT 잡음이 만드는 시각 잡음의 3배 (곡선 기울기로 환산) | 동일 |

판정 결과와 함께 **각 지표마다 "검출에 필요한 σ_φ" 와 "필요한 평균 횟수 N"** 을 보고한다 (N = (3·6.7 mK / 신호 mK)²).

## 7. 1단계 케이스 (2-D, 축 위 단일 void)

순서가 중요하다: 크기 사다리 → 기준 void 재선정 → 나머지 계열.

| 계열 | 케이스 | 값 |
|---|---|---|
| A 크기 사다리 (구, 축 위, 중심 깊이 250 µm) | 8 | 지름 5, 8, 12, 16, 20, 25, 30, 40 µm (`VoidSpec(depth=z_c − d/2, thickness=d, r_half=d/2, shape="ellipse")`) |
| 기준 void 선정 | — | A 결과에서 **Δk/k ≈ 2 × 3σ_k (≈ 1.8 %)** 가 되는 지름을 고른다 (보간). 보고서에 선정 근거 기록 |
| B 깊이 (기준 구) | 5 | 중심 깊이 25, 100, 250, 400, 475 µm (양 끝은 void 가 면에 닿지 않게 반경만큼 띄움) |
| C 형태 (기준 부피 동일) | 4 | 구 / 원판 축비 1:4 / **얇은 원판 1:10** / 막대 4:1 (축비 = 두께:지름; 타원체 부피 4/3·π·r_half²·(t/2)) |
| 기준 via | 1 | void 없음, 전 케이스 공통 |

B 에서 확인·보고할 것: **t_int 의 감도가 양 끝 깊이에서 둔감해지는가** (1-D 근사 예측: 감도 ∝ x(1−x), x = z/L).
k_eff 는 깊이에 거의 무관할 것으로 예상 (직렬 저항). 펌프면·양면 지표는 깊이 의존.

적층(축 위 N개) 배치는 다중 void 스터디로 미룬다.

## 8. 2단계 케이스 (3-D, 축 밖 단일 void) — 중간 보고 뒤 결정

r_off = 10, 20, 30 µm (0 은 1단계), θ = 0 고정 (펌프 스팟 원형·중심 정렬이므로 결과 무관 — 보고서에 명시).
각 r_off 에서 검출 가능한 최소 지름을 **이분법**으로 찾는다 (판정 지표: 반대면·양면 구성은 k_eff, 펌프면 구성은 펌프면 ΔT 변화).
r_off + r_half ≤ 40 µm (유리 벽) 제한. 먼저 r_off = 20, 기준 지름 한 케이스로 시간을 재서 보고.
3-D 는 `needs_3d(cfg)` 가 True 면 `run_step_response` 가 자동으로 `solver3d` 를 쓴다; `fit_resolution` 로 미지수를 한도에 맞춘다.

## 9. 보고서

Word(.docx, 편집 가능) — 이전 보고서와 같은 형식 (`studies/multi_void_report.py`, `detection_map_docx.py` 참고,
python-docx, 글꼴 Malgun Gothic). 곡선 비교는 같은 축에 겹쳐 그린다.

1. **요약 (1쪽)**: 프로브 구성 × 지표별 검출 한계 표 한 개 (예: "k_eff 기준: 축 위 void 지름 X µm 이상, r_off=20 µm 에서 Y µm 이상, 깊이 Z 까지").
   가장 유리한 구성·지표 한두 문장.
2. **계열별 결과**: 계열마다 표 1개 + 대표 그래프 1개. 표: 조건값, 지표별 Δ, 판정(가능/불가, 사유), 필요한 σ_φ / N.
3. **부록**: 고정 조건, 판정 기준 근거, 계산 경로 교차검증(직접 풀이 vs 스텝 FFT, 기준 via), 3-D 케이스 계산 시간.

중간 보고(1단계 끝)는 Markdown `results/cw_void_detect/interim.md` 로 충분. 최종 docx 는 2단계까지 끝난 뒤
(2단계를 하지 않기로 하면 1단계만으로).

## 10. 코드 진입점 요약

- `ttr_sim/transmission.py`: `cw_config`, `run_step_response`, `frequency_response`, `lockin_dataset`, `penetration_depth`
- `ttr_sim/analysis.py`: `invert_tlm`(use="phase"), `analyze`, `void_indicators`, `INDICATOR_KEYS`, `phase_noise_sensitivity`, `lockin`,
  `frequency_response_from_step`
- `ttr_sim/solver.py`: `VoidSpec(depth, thickness, r_half, r_center, theta, k, shape)`, `baseline_config`
- `ttr_sim/solver3d.py`: `needs_3d`, `count_unknowns`, `fit_resolution`, `DIRECT_MAX_UNKNOWNS`
- 참고 스터디: `studies/cw_lateral_loss.py`(케이스 루프·csv·md 틀), `studies/cw_phase_noise.py`(σ_k), `studies/cw_spot_shape.py`(3-D 실행 예)
- 이전 결과: `results/cw_lateral_loss/summary.md`, `results/cw_phase_noise/summary.md`, `results/cw_spot_shape/summary.md`

## 11. 알아둘 것 (지난 세션의 교훈)

- 기본 void(원판 r_half 20, 두께 10, 깊이 245 µm)는 Δk/k −1.10 %, 반대면 위상 변화 −0.99° @ 2 kHz, 반대면 ΔT 최대 −4 % @ 0.17 ms.
- 유리 껍질 TLM 의 모델 오차: void 없는 via 가 참값보다 +1…+2 % 높게 나온다 → **Δk/k 는 항상 같은 경로의 기준 via 대비**로 계산.
- 양면 온도차 R_th 는 깊이에 따라 감도가 선형으로 줄어 반대면 쪽 void 를 거의 못 본다 (flux ∝ 1 − z/L).
- `w ** 2` 같은 수식은 바꾸지 말 것 (반사형 결과 비트 단위 보존).
- Streamlit 은 이 스터디와 무관; GUI 를 띄울 필요 없다.
