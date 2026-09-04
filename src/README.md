# ns-TTR Void Detection Simulator

구리 마이크로 실린더(⌀80 μm × 500 μm, fused silica 매립) 내부 void를 레이저 펌프-프로브
thermoreflectance로 검출할 수 있는지 사전 검증하는 시뮬레이터입니다.
단순 계산이 아니라 **결과를 검증할 수 있는 도구**를 목표로 하며, 에너지 보존 / grid convergence /
해석해 비교 / k_void 민감도 테스트를 GUI에서 바로 실행할 수 있습니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

테스트:

```bash
pytest -q
```

Streamlit Community Cloud 배포: 이 디렉터리(`app.py`, `requirements.txt`, `ttr_sim/`)를 GitHub 저장소로 올리고
Community Cloud에서 `app.py`를 진입점으로 지정하면 됩니다. 별도 설치 파일은 필요 없습니다.

## 구조

| 파일 | 내용 |
|---|---|
| `app.py` | Streamlit GUI (설정 → 적용 → 프로그레스 바 → 결과/검증 탭) |
| `ttr_sim/materials.py` | 구리 / fused silica / 공기 물성치 |
| `ttr_sim/presets.py` | void 깊이별 펄스폭·격자 프리셋(§3), k_void 프리셋(§4) |
| `ttr_sim/solver.py` | 축대칭 (r,z) 유한체적 + Crank–Nicolson 솔버, 격자 생성, void 신호 지표 |
| `ttr_sim/analytic.py` | Carslaw & Jaeger 형 semi-infinite 해석해 (Gaussian spot, square/Gaussian 펄스) |
| `ttr_sim/validation.py` | 에너지 보존 / grid convergence / 해석해 비교 / k_void 스윕 |
| `tests/test_solver.py` | pytest 검증 스위트 |

## 물리·수치 모델 요약

* 지배방정식 `ρc ∂T/∂t = ∇·(k∇T)`, 축대칭 (r, z). 내부 열원 없음.
* 레이저: Beer–Lambert 부피 열원을 `Δz ≫ 1/α_abs` 조건에서 표면 flux
  `-k ∂T/∂z|_{z=0} = I₀(1-R) f(t) exp(-2r²/w²)` 로 재표현. 각 환형 면 위에서 Gaussian을 정확 적분.
  Δz/δ_abs < 7 이면 GUI에 경고.
* 경계: 축 대칭, 구리/실리카 계면 온도·flux 연속(면 전도도 = 직렬 저항), 후면 z=500 μm 단열, 외곽 실리카 단열.
  `√(D·t_end)` 가 로드 길이의 80%를 넘으면 경고.
* 격자: 관심 영역(표면 ~ void 아래, 스팟 반경 2배)은 프리셋 Δz(=Δr)로 균일, 그 밖은 기하급수 성장.
  void 경계와 r=40 μm 계면에 격자 면을 정렬. 프리셋 Δz가 스팟보다 거칠면 표면층(~2w)만 Δr 로 세분 후 Δz 로 전이.
* 온도장 표시: 색 범위는 기본적으로 전체 시간에 대해 고정(선형 또는 로그, 로그는 최대값의 1/1000까지 표시)되어
  식는 과정이 그대로 보이며, "스냅샷별 자동" 을 고르면 각 시각의 최대값에 맞춰집니다. 확대 범위도 관측창 전체 기준으로 고정.
* 온도장 스냅샷: 관측 시간창을 균등 분할하여 저장. 개수는 고급 수치 설정의 "온도장 스냅샷 개수" 슬라이더(12~100)로 지정.
  검증용 재계산(grid convergence, k_void 스윕)은 스냅샷을 저장하지 않음.
* 시간적분: Crank–Nicolson. Δt 는 Fo = D_cu·Δt/Δz² (기본 0.5) 로 결정 → 펄스당 200 스텝.
  행렬은 한 번만 LU 분해(SuperLU)하고 매 스텝 재사용.
* 표면 온도는 셀 중심값에 `q·Δz/(2k)` 를 더해 면 온도로 재구성(2차 정확). 프로브 가중(1/e² 반경) 평균을 신호로 사용.
* void: 축상 원판 또는 축대칭 링. 깊이·두께·반경 위치·반경 반폭은 μm 단위 숫자로 직접 입력 (깊이 프리셋을 바꾸면
  대표 깊이에 맞는 기본값으로 다시 채워짐). 숫자 입력창들은 하나의 form 안에 있어 값을 타이핑한 뒤 **적용 (재계산)** 을
  누르면 그대로 반영됨 — Enter 키 전달에 의존하지 않으므로 한글 IME 상태에서도 값이 되돌아가지 않음.
  k_void 는 §4의 4개 이산값 또는 "직접 입력 (숫자)" 옵션으로 지정.
  유한체적 정식화이므로 k_void→0 도 특이성 없음. 즉 floor 값은 순수한 모델링 선택이며, 민감도 탭에서 0.026 대비 차이를
  확인해야 함 (직접 입력한 값은 민감도 스윕에 자동 포함).

## 검증 결과 (기본 설정, 균질 구리 모드)

* 에너지 보존 오차: ~1e-14 (모든 경계 단열 + 보존적 이산화 + 직접해)
* 해석해 대비 baseline RMS 편차: 프리셋 0~6 에서 < 2 %, 프리셋 8(400 μm) 은 후면 도달로 편차 증가(경고 표시)
