import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["DejaVu Sans", "Malgun Gothic"]; plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9; plt.rcParams["mathtext.fontset"] = "dejavusans"
A = np.load("vA.npz"); C = np.load("vC2.npz"); Ce = np.load("vC2e.npz"); V4 = np.load("v4.npz")
F1 = np.load("vF1.npz"); F2 = np.load("vF2.npz"); G = np.load("vG.npz")

# ---- A: baseline vs multilayer analytic
fig, ax = plt.subplots(2, 2, figsize=(7.5, 5), sharex="col", gridspec_kw=dict(height_ratios=[2, 1]))
for j, (pi, lab) in enumerate(((0, "프리셋 0 (d=1 μm, τp=17 ns)"), (2, "프리셋 2 (d=5 μm, τp=431 ns)"))):
    t, s, r = A[f"A{pi}_t"] * 1e9, A[f"A{pi}_sim"], A[f"A{pi}_ref"]
    ax[0, j].plot(t, s, "b-", lw=1.2, label="시뮬레이터"); ax[0, j].plot(t, r, "r--", lw=1, label="해석해 (Laplace–Hankel)")
    ax[0, j].set_title(lab); ax[0, j].set_ylabel("ΔT_probe [K]"); ax[0, j].legend(); ax[0, j].set_xscale("log")
    ax[1, j].plot(t, (s - r) / r.max() * 100, "k-", lw=1); ax[1, j].axhline(0, color="gray", lw=0.5)
    ax[1, j].set_ylabel("차이 / 피크 [%]"); ax[1, j].set_xlabel("t [ns]"); ax[1, j].set_ylim(-0.3, 0.3)
fig.tight_layout(); fig.savefig("fig_A.png", dpi=160); plt.close(fig)

# ---- C: wide void vs layered analytic, r_half sweep
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
t = V4["t"] * 1e9
ax[0].plot(t, (V4["sim_v"] - V4["sim_b"]) * 1e3, "b-", label="시뮬레이터 (원판 void, r_half=40 μm)")
ax[0].plot(t, (V4["ref_v"] - V4["ref_b"]) * 1e3, "r--", label="적층 해석해 Cu 5 μm/공기 2.5 μm/Cu")
ax[0].set_xlabel("t [ns]"); ax[0].set_ylabel("void 신호 ΔT_void - ΔT_base [mK]"); ax[0].legend(fontsize=7); ax[0].set_title("C: 넓은 void 극한")
for key, D, ls, name in (("box", C, "-", "원판(box)"), ("ellipse", Ce, "--", "타원체(ellipse)")):
    for rh, col in ((40, "k"), (20, "C0"), (10, "C1"), (5, "C2"), (2.5, "C3")):
        tt, sig = D[f"{key}_{rh:g}"]
        ax[1].plot(tt * 1e9, sig * 1e3, ls, color=col, lw=1, label=f"{name} r_half={rh:g} μm" if key == "box" else None)
ax[1].plot(C["t_ref"] * 1e9, C["sref"] * 1e3, "r:", lw=1.5, label="적층 해석해 (상한)")
ax[1].set_xlabel("t [ns]"); ax[1].set_ylabel("void 신호 [mK]"); ax[1].legend(fontsize=6); ax[1].set_title("r_half 스윕 (실선 box, 점선 ellipse)")
rhs = [40, 20, 10, 5, 2.5]
pk_box = [C[f"box_{r:g}"][1].max() * 1e3 for r in rhs]; pk_el = [Ce[f"ellipse_{r:g}"][1].max() * 1e3 for r in rhs]
ax[2].plot(rhs, pk_box, "s-", label="원판(box)"); ax[2].plot(rhs, pk_el, "o--", label="타원체(ellipse)")
ax[2].axhline(C["sref"].max() * 1e3, color="r", ls=":", label="적층 해석해 상한 16.21 mK")
ax[2].set_xscale("log"); ax[2].set_xticks([2.5, 5, 10, 20, 40]); ax[2].set_xticklabels(["2.5", "5", "10", "20", "40"]); ax[2].tick_params(axis="x", which="minor", labelbottom=False); ax[2].set_xlabel("r_half [μm]"); ax[2].set_ylabel("void 신호 피크 [mK]"); ax[2].legend(fontsize=7); ax[2].set_title("피크의 단조 수렴")
fig.tight_layout(); fig.savefig("fig_C.png", dpi=160); plt.close(fig)

# ---- E: insulated slab
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for j, (pi, lab) in enumerate(((6, "τp = 172 μs"), (8, "τp = 2.76 ms (후면 도달)"))):
    t, s, r = A[f"E{pi}_t"] * 1e3, A[f"E{pi}_sim"], A[f"E{pi}_ref"]
    ax[j].plot(t, s * 1e3, "b-", label="시뮬레이터"); ax[j].plot(t, r * 1e3, "r--", label="1D 단열 슬랩 해석해")
    ax2 = ax[j].twinx(); err = (s - r) / r.max() * 100; ax2.plot(t, err, "g-", lw=0.7, alpha=0.8); ax2.set_ylabel("차이/피크 [%]", color="g"); ax2.set_ylim(-1.5, 1.5)
    ax[j].set_xlabel("t [ms]"); ax[j].set_ylabel("ΔT 표면 [mK]"); ax[j].set_title("E: " + lab); ax[j].legend(fontsize=7, loc="center right")
    i = np.argmax(np.abs(err)); print(f"E{pi}: max |err| {abs(err[i]):.2f} % at t = {t[i]*1e3:.4g} us (tau_p={lab}); err at first step {err[1]:.2f} %, err just after pulse end:",
                                     err[np.searchsorted(A[f'E{pi}_t'], {6: 172e-6, 8: 2.76e-3}[pi]) + np.array([-1, 0, 1])])
fig.tight_layout(); fig.savefig("fig_E.png", dpi=160); plt.close(fig)

# ---- F1: Cu 220 ns (Dillmann) — same axes as the poster's Fig. 1 (0-400 ns, 0-4000 K)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for key, lab, col in (("r", "27.5 μm = 1/e² 반경 (q0 = 1.09×10¹¹ W/m²)", "C0"), ("d", "27.5 μm = 지름 (q0 = 4.38×10¹¹ W/m²)", "C1")):
    t, s, r = F1["t" + key] * 1e9, F1["sim" + key] + 293.15, F1["ref" + key] + 293.15
    ax.plot(t, s, "-", color=col, label="시뮬레이터, " + lab); ax.plot(t, r, ":", color=col, lw=1)
ax.axhline(1358, color="k", ls="--", lw=0.8); ax.text(300, 1400, "Cu 용융 1358 K", fontsize=8)
ax.axhline(2835, color="k", ls="-.", lw=0.8); ax.text(300, 2880, "Cu 비등 2835 K", fontsize=8)
ax.set_xlim(0, 400); ax.set_ylim(0, 5000); ax.set_xlabel("t [ns]"); ax.set_ylabel("중심 표면 온도 [K] (선형 모델, 상변화 없음)")
ax.set_title("F1: Cu, 220 ns square, 0.286 mJ, 흡수율 0.1 (Dillmann 2016 Fig. 1 축)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("fig_F1.png", dpi=160); plt.close(fig)

# ---- F2: Si (Darif & Semmar Fig. 5b axes: 0-60 ns, 200-2000 K)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
t, s = F2["t"] * 1e9, F2["sim"]
for Fl, col in ((800, "C0"), (850, "C1"), (900, "C2"), (950, "C3"), (1000, "C4")):
    T = 293 + s * Fl; ax.plot(t, T, color=col, label=f"F = {Fl} mJ/cm²")
ax.axhline(1690, color="k", ls="--", lw=0.8); ax.text(40, 1710, "Si 용융 1690 K (Darif 표 1)", fontsize=8)
ax.set_xlim(0, 60); ax.set_ylim(200, 2000); ax.set_xlabel("t [ns]"); ax.set_ylabel("표면 온도 [K]")
ax.set_title("F2: Si, 27 ns gate, R=0.59, 1D (Darif & Semmar 2008 Fig. 5(b) 축)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("fig_F2.png", dpi=160); plt.close(fig)

# ---- G: Mao 2024 Fig. 6 axes (log-log, 5-1000 ns, 0.01-1)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for j, (key, lab) in enumerate((("al", "(a) Al 80 nm / 사파이어, G=90 MW/m²K, K=29.3 W/mK"), ("au", "(b) Au 100 nm / SiC, G=73.6 MW/m²K, K=349.9 W/mK"))):
    ax[j].plot(G["t"] * 1e9, G[key], "b-"); ax[j].set_xscale("log"); ax[j].set_yscale("log"); ax[j].set_xlim(5, 1500); ax[j].set_ylim(0.005, 1.2)
    ax[j].set_xlabel("펄스 중심 기준 t [ns]"); ax[j].set_ylabel("정규화 ΔT"); ax[j].set_title(lab, fontsize=8); ax[j].grid(True, which="both", lw=0.3)
fig.suptitle("G: 다층 해석해 기준 곡선 (Mao et al. 2024 Fig. 6 조건, 시뮬레이터 미실행)", fontsize=9)
fig.tight_layout(); fig.savefig("fig_G.png", dpi=160); plt.close(fig)
print("figures done")
