"""Desktop launcher for the ns-TTR Simulator Streamlit app.

The same executable runs in two roles:

* GUI process (default): picks a free localhost port, starts a copy of itself in server mode, shows a
  native pywebview (Edge WebView2) window with a splash page, switches it to the app once the server
  answers its health check, and stops the server when the window is closed.
* Server process (``--serve PORT PARENT_PID``): runs Streamlit's server on its main thread (Streamlit
  installs signal handlers, which only works there) and exits by itself if the GUI process disappears.

The app itself (``src/app.py`` + ``src/ttr_sim``) is bundled unchanged.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP_NAME = "ns-TTR Simulator"
FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    APP_SCRIPT = Path(sys._MEIPASS) / "app" / "app.py"
else:  # running from source: use the app in ../src directly
    APP_SCRIPT = Path(__file__).resolve().parent.parent / "src" / "app.py"

LOG_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "logs"

# Mirrors src/.streamlit/config.toml, plus the settings a bundled single-user app needs.
STREAMLIT_FLAGS = {
    "global_developmentMode": False,   # frozen builds are otherwise detected as dev mode, which forbids server.port
    "server_headless": True,
    "server_address": "127.0.0.1",
    "server_fileWatcherType": "none",
    "server_runOnSave": False,
    "browser_gatherUsageStats": False,
    "client_toolbarMode": "minimal",
    "theme_base": "light",
    "theme_primaryColor": "#d62728",
}

SPLASH_HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin:0; height:100vh; display:flex; align-items:center; justify-content:center;
       font-family:'Segoe UI',sans-serif; background:#fff; color:#31333f; }}
.box {{ text-align:center; }} h1 {{ font-weight:600; font-size:22px; margin:0 0 12px; }}
.spin {{ width:28px; height:28px; margin:0 auto 16px; border:3px solid #eee; border-top-color:#d62728;
        border-radius:50%; animation:s 0.9s linear infinite; }} @keyframes s {{ to {{ transform:rotate(360deg); }} }}
p {{ color:#808495; font-size:13px; margin:0; }}
</style></head><body><div class="box"><div class="spin"></div><h1>{APP_NAME}</h1>
<p id="msg">Starting the simulation engine...</p></div></body></html>"""


def _redirect_output(role: str) -> None:
    """Windowed builds have no console (sys.stdout is None), so send all output to a log file."""
    if FROZEN:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOG_DIR / f"{role}.log", "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log


# ----------------------------------------------------------------------------- server role
def _exit_when_parent_dies(parent_pid: int) -> None:
    import ctypes
    SYNCHRONIZE, INFINITE = 0x00100000, 0xFFFFFFFF
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
    if handle:
        kernel32.WaitForSingleObject(handle, INFINITE)
    os._exit(0)


def run_server(port: int, parent_pid: int) -> None:
    _redirect_output("server")
    threading.Thread(target=_exit_when_parent_dies, args=(parent_pid,), daemon=True).start()

    from streamlit.web import bootstrap

    flags = dict(STREAMLIT_FLAGS, server_port=port)
    bootstrap.load_config_options(flags)
    bootstrap.run(str(APP_SCRIPT), False, [], flags)


# ----------------------------------------------------------------------------- GUI role
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server_command(port: int) -> list[str]:
    args = ["--serve", str(port), str(os.getpid())]
    if FROZEN:
        return [sys.executable, *args]
    return [sys.executable, str(Path(__file__).resolve()), *args]


def _wait_until_ready(url: str, proc: subprocess.Popen, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "_stcore/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


def run_gui() -> None:
    _redirect_output("gui")
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True   # Plotly "download plot as png" and st.download_button

    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    proc = subprocess.Popen(_server_command(port), creationflags=subprocess.CREATE_NO_WINDOW)

    window = webview.create_window(APP_NAME, html=SPLASH_HTML, width=1600, height=1000,
                                   min_size=(1000, 700), text_select=True)

    def load_app() -> None:
        if _wait_until_ready(url, proc):
            window.load_url(url)
        else:
            msg = f"The simulation engine failed to start. See the log in {LOG_DIR}".replace("\\", "\\\\")
            window.evaluate_js(f"document.getElementById('msg').textContent = '{msg}'")

    try:
        webview.start(load_app, private_mode=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "--serve":
        run_server(int(sys.argv[2]), int(sys.argv[3]))
    else:
        run_gui()


if __name__ == "__main__":
    main()
