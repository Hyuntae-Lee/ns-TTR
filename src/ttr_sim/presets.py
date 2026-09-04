"""Depth-dependent pulse/grid presets (spec §3) and k_void presets (spec §4).

The pulse width and grid spacing are always set as a pair from the depth preset:
    tau_p = 2 * d^2 / D_th        (factor 2 safety margin)
    dz    = sqrt(D_th * tau_p) / 10  (10 cells across the thermal penetration length)
"""
from dataclasses import dataclass
import math

D_TH_PRESET = 1.16e-4  # m^2/s, copper diffusivity used in the spec table


@dataclass(frozen=True)
class DepthPreset:
    label: str
    depth_range: str
    d_rep: float          # representative void depth, m
    warning: str = ""

    @property
    def tau_p(self) -> float:
        return 2.0 * self.d_rep ** 2 / D_TH_PRESET

    @property
    def dz(self) -> float:
        return math.sqrt(D_TH_PRESET * self.tau_p) / 10.0

    @property
    def display(self) -> str:
        return f"{self.label}  |  τp={fmt_time(self.tau_p)}, Δz={self.dz * 1e6:.2f} μm"


DEPTH_PRESETS = [
    DepthPreset("표면 극근접 (~2μm)", "~2 μm", 1e-6),
    DepthPreset("얕은 void (2~5μm)", "2~5 μm", 2e-6),
    DepthPreset("서브표면 (5~10μm)", "5~10 μm", 5e-6),
    DepthPreset("중간 깊이 (10~20μm)", "10~20 μm", 10e-6),
    DepthPreset("중간 깊이 (20~40μm)", "20~40 μm", 20e-6, "반경 한계(40 μm) 부근: 측면 실리카 계면의 영향이 커집니다."),
    DepthPreset("깊은 편 (40~100μm)", "40~100 μm", 40e-6),
    DepthPreset("깊음 (100~200μm)", "100~200 μm", 100e-6),
    DepthPreset("매우 깊음 (200~400μm)", "200~400 μm", 200e-6),
    DepthPreset(
        "최대 깊이 (400~500μm)", "400~500 μm", 400e-6,
        "⚠ 후면(z=500 μm) 단열 경계조건에 근접합니다. 열 침투 깊이가 로드 길이에 가까워져 결과 신뢰도가 낮습니다.",
    ),
]


@dataclass(frozen=True)
class KVoidPreset:
    label: str
    k: float
    description: str
    warning: str = ""


KVOID_PRESETS = [
    KVoidPreset("실제 공기값 (0.026)", 0.026, "물리적으로 가장 정확. 유한체적 정식화에서는 특이성 없이 계산됩니다."),
    KVoidPreset("보수적 최소값 (0.1)", 0.1, "공기값에 가까우면서 안정성 확보 시도."),
    KVoidPreset("수치 안정 표준값 (1)", 1.0, "일반적으로 안정적으로 수렴."),
    KVoidPreset(
        "기존 관습값 (10)", 10.0, "과거 세션에서 쓰였던 값 — 리뷰에서 '임의적'으로 지적됨.",
        "이 값은 실제 공기 열전도율(0.026 W/m·K)보다 약 400배 큽니다. 결과를 실제값(0.026)과 비교해보세요.",
    ),
]


def fmt_time(t: float) -> str:
    """Human-readable time with SI prefix."""
    if t == 0:
        return "0 s"
    a = abs(t)
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "μs"), (1e-9, "ns"), (1e-12, "ps")):
        if a >= scale:
            return f"{t / scale:.3g} {unit}"
    return f"{t:.3g} s"


def fmt_length(x: float) -> str:
    a = abs(x)
    if a >= 1e-3:
        return f"{x * 1e3:.3g} mm"
    if a >= 1e-6:
        return f"{x * 1e6:.3g} μm"
    return f"{x * 1e9:.3g} nm"
