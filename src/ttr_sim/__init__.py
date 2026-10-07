"""ns-TTR void detection simulator: axisymmetric Crank-Nicolson heat diffusion."""
from .materials import Material, COPPER, FUSED_SILICA, AIR
from .presets import DEPTH_PRESETS, KVOID_PRESETS, DepthPreset, KVoidPreset
from .solver import (
    BottomStack, Geometry, VoidSpec, Laser, Numerics, SimConfig, SimResult, run_simulation, build_grid,
    random_voids,
)
from .analytic import surface_step_response, analytic_probe_signal

__all__ = [
    "Material", "COPPER", "FUSED_SILICA", "AIR",
    "DEPTH_PRESETS", "KVOID_PRESETS", "DepthPreset", "KVoidPreset",
    "BottomStack", "Geometry", "VoidSpec", "Laser", "Numerics", "SimConfig", "SimResult",
    "run_simulation", "build_grid", "random_voids",
    "surface_step_response", "analytic_probe_signal",
]
