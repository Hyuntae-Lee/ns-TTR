# ns-TTR Simulator — 데스크톱 빌드

`../src` 의 Streamlit 앱(`app.py`, `ttr_sim/`)을 **수정 없이** 묶어 Windows 독립 실행 프로그램으로 만듭니다.
PyInstaller(one-folder) + pywebview(Edge WebView2 네이티브 창) 구성입니다.

## 빌드

`build.bat` 더블클릭 (또는 PowerShell 에서 `.\build.ps1`).

결과: `dist\ns-TTR Simulator\ns-TTR Simulator.exe` (폴더 전체 약 290 MB). 배포할 때는 `dist\ns-TTR Simulator` 폴더를
통째로 zip 으로 묶어 전달합니다. 대상 PC에 Python 설치는 필요 없습니다 (Windows 10/11 기본 탑재 WebView2 사용).

첫 실행 시 `.venv` 를 만들고 `requirements-build.txt` 를 설치합니다. 런타임 패키지 버전은 `requirements-runtime.txt`
에 `src/.venv` 와 동일하게 고정되어 있으므로, `src` 쪽 패키지를 올리면 이 파일도 같이 갱신하세요.
`src/app.py` 나 `ttr_sim` 을 고친 뒤에는 `build.bat` 만 다시 실행하면 됩니다.

## 동작 방식 (`launcher.py`)

* exe 를 실행하면 빈 로컬 포트를 고르고, 같은 exe 를 `--serve <port> <pid>` 로 한 번 더 띄워 Streamlit 서버를 돌립니다
  (Streamlit 은 메인 스레드에서만 시작 가능하고, pywebview 창도 메인 스레드를 써야 하므로 프로세스 두 개로 분리).
* 창에는 먼저 로딩 화면이 뜨고, 서버 health check 가 응답하면 앱으로 전환됩니다.
* 창을 닫으면 서버가 종료됩니다. GUI 프로세스가 비정상 종료되어도 서버가 스스로 종료합니다.
* 로그: `%LOCALAPPDATA%\ns-TTR Simulator\logs\gui.log`, `server.log`

소스에서 바로 실행 (빌드 없이 확인):

```powershell
.venv\Scripts\python.exe launcher.py
```

## 파일

| 파일 | 내용 |
|---|---|
| `launcher.py` | 창 + 서버 프로세스 관리 |
| `ns_ttr.spec` | PyInstaller 설정 (`../src/app.py` 를 데이터로, `ttr_sim` 을 모듈로 포함) |
| `build.bat` | 더블클릭용 빌드 (`build.ps1` 호출) |
| `build.ps1` | 빌드 스크립트 |
| `requirements-runtime.txt` | 앱 런타임 패키지 (src/.venv 와 동일 버전) |
| `requirements-build.txt` | 런타임 + PyInstaller + pywebview |
