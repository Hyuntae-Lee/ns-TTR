"""ns-TTR void detection simulator: axisymmetric Crank-Nicolson heat diffusion."""
from .materials import Material, COPPER, FUSED_SILICA, AIR
from .presets import DEPTH_PRESETS, KVOID_PRESETS, DepthPreset, KVoidPreset
from .solver import (
    Geometry, VoidSpec, Laser, Numerics, SimConfig, SimResult, run_simulation, build_grid,
)
from .analytic import surface_step_response, analytic_probe_signal

__all__ = [
    "Material", "COPPER", "FUSED_SILICA", "AIR",
    "DEPTH_PRESETS", "KVOID_PRESETS", "DepthPreset", "KVoidPreset",
    "Geometry", "VoidSpec", "Laser", "Numerics", "SimConfig", "SimResult",
    "run_simulation", "build_grid",
    "surface_step_response", "analytic_probe_signal",
]
