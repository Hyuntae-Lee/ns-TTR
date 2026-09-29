# Optical inputs: R(532), R(632.8), absorption depth from the collected n,k data; thin-oxide Fresnel model (my calculation)
import numpy as np, json
def R_bulk(n, k):
    return ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)
def R_film(n_ox, d, n_cu, lam):
    """air / oxide (thickness d) / Cu, normal incidence, transfer matrix."""
    n0 = 1.0; n1 = n_ox; n2 = n_cu
    r01 = (n0 - n1) / (n0 + n1); r12 = (n1 - n2) / (n1 + n2)
    beta = 2 * np.pi * n1 * d / lam
    r = (r01 + r12 * np.exp(2j * beta)) / (1 + r01 * r12 * np.exp(2j * beta))
    return abs(r) ** 2
data = {  # n, k at 532 and 632.8 nm (linear interpolation of the tabulated records in the collected PDFs)
    "McPeak 2015": {532: (0.9453, 2.5997), 632.8: None},
    "Rakic 1998 BB": {532: (0.8076, 2.4886), 632.8: None},
}
# McPeak table around 0.63-0.64 um: (0.63: 0.110069919, 3.546563885), (0.64: 0.107194012, 3.666809315)
f = (632.8 - 630) / 10.0
data["McPeak 2015"][632.8] = (0.110069919 + f * (0.107194012 - 0.110069919), 3.546563885 + f * (3.666809315 - 3.546563885))
# Rakic BB around 0.62767 (0.31902, 3.3821) and 0.64071 (0.28853, 3.5153)
f = (632.8 - 627.67) / (640.71 - 627.67)
data["Rakic 1998 BB"][632.8] = (0.31902 + f * (0.28853 - 0.31902), 3.3821 + f * (3.5153 - 3.3821))
out = {}
for name, d in data.items():
    for lam, (n, k) in d.items():
        alpha = 4 * np.pi * k / (lam * 1e-9)
        out[f"{name} @{lam}"] = dict(n=round(n, 4), k=round(k, 4), R=round(R_bulk(n, k), 4), depth_nm=round(1e9 / alpha, 1), one_minus_R=round(1 - R_bulk(n, k), 4))
print(json.dumps(out, indent=1))
# Ordal 1985 single point 517 nm: n=1.16, k=2.64
print("Ordal 517 nm R =", round(R_bulk(1.16, 2.64), 4))
# Oxide layer effect at 532 nm (pump). Oxide index at 532 nm is NOT in the collected data; Barchiesi 2022 gives
# n^2 = 8.2 + 1.0i at 632.8 nm for the natural oxide (76 % Cu2O / 24 % CuO). Assume the same n at 532 nm (Cu2O
# bandgap ~2.1-2.5 eV so absorption rises toward 532 nm; treated as a sensitivity range).
n_ox_633 = np.sqrt(8.2 + 1.0j)
print("oxide n at 632.8 nm (Barchiesi 2022 mean) =", n_ox_633)
lam = 532e-9
rows = []
for name in data:
    n, k = data[name][532]; n_cu = n + 1j * k
    for n_ox in (n_ox_633, 3.1 + 0.25j):
        vals = [R_film(n_ox, d * 1e-9, n_cu, lam) for d in (0, 1, 2, 3, 5, 10)]
        rows.append((name, f"{n_ox.real:.2f}+{n_ox.imag:.2f}i", [round(v, 3) for v in vals]))
print("R(532) vs oxide thickness 0,1,2,3,5,10 nm:")
for r in rows: print("  ", r)
# Barchiesi's own Cu at 632.8: n^2 = -13.3+3.3i (thin films, mean) vs bulk -11.6+1.6i
for label, eps in (("Barchiesi thin-film Cu", -13.3 + 3.3j), ("Barchiesi bulk Cu [29]", -11.6 + 1.6j)):
    nn = np.sqrt(eps); print(label, "n,k =", round(nn.real, 3), round(nn.imag, 3), "R(632.8) =", round(R_bulk(nn.real, nn.imag), 4))
# probe R(632.8) with oxide
lam = 632.8e-9
for name in data:
    n, k = data[name][632.8]; n_cu = n + 1j * k
    vals = [R_film(n_ox_633, d * 1e-9, n_cu, lam) for d in (0, 2, 5)]
    print(name, "R(632.8) oxide 0/2/5 nm:", [round(v, 4) for v in vals])
