ns-TTR 시뮬레이터 검증 스크립트 (2026-09-17)

사용법: 이 파일들을 저장소의 src 폴더(ttr_sim 패키지가 보이는 위치)에 복사한 뒤
    python v1.py   # A: 다층 해석해 대비 baseline, B: w0 불변성
    python v4.py   # C: 넓은 void 극한 = 구리/공기/구리 적층 해석해
    python v5.py   # D: 깊이-시간 스케일링(자기상사 스윕, 실제 형상 스윕)
    python v6.py   # E: 유한 길이 단열 슬랩(후면 경계)
refml.py 는 기준해(임피던스 재귀 + Hankel 적분 + Talbot 역 Laplace) 모듈.
필요 패키지: numpy, scipy (시뮬레이터와 동일). 전체 실행 3~4분.
기대 결과는 'ns-TTR 시뮬레이터 검증 계획서.pdf' §3 참조.
