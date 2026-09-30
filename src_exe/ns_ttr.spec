# PyInstaller spec for the ns-TTR Simulator desktop build.  Build with:  .\build.ps1
# Produces a one-folder app in dist\ns-TTR Simulator\ (run "ns-TTR Simulator.exe" inside it).
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

import sys

SRC = Path(SPECPATH).parent / "src"          # the Streamlit app, bundled unchanged
sys.path.insert(0, str(SRC))                 # so collect_submodules("ttr_sim") below can import it

datas = [(str(SRC / "app.py"), "app"), (str(SRC / "void_editor.html"), "app")]
binaries = []
hiddenimports = []

# Streamlit loads its frontend (static/), protobufs and runtime modules dynamically, and reads its own
# package metadata at import time.
for pkg in ("streamlit",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
datas += copy_metadata("streamlit")

# app.py is shipped as a data file, so PyInstaller cannot see its imports; list them here.  ttr_sim is
# compiled into the bundle and found by app.py's `import ttr_sim` through the frozen importer.
hiddenimports += collect_submodules("ttr_sim")
assert "ttr_sim.solver" in hiddenimports, "ttr_sim not found under ../src"
hiddenimports += [
    "numpy", "pandas", "scipy.interpolate", "scipy.sparse", "scipy.sparse.linalg",
    "plotly.graph_objects", "streamlit.components.v1",
]
hiddenimports += collect_submodules("plotly.validators")   # imported lazily by plotly.graph_objects

a = Analysis(
    ["launcher.py"],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ns-TTR Simulator",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ns-TTR Simulator", upx=False)
