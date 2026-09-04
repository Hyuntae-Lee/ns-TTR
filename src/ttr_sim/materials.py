"""Material property tables (spec §2.4)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    k: float        # thermal conductivity, W/(m K)
    rho: float      # density, kg/m^3
    cp: float       # specific heat, J/(kg K)

    @property
    def alpha(self) -> float:
        """Thermal diffusivity, m^2/s."""
        return self.k / (self.rho * self.cp)

    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity, J/(m^3 K)."""
        return self.rho * self.cp


COPPER = Material("Copper", k=398.0, rho=8960.0, cp=385.0)          # alpha = 1.15e-4
FUSED_SILICA = Material("Fused silica", k=1.38, rho=2200.0, cp=740.0)  # alpha = 8.5e-7
AIR = Material("Air (void)", k=0.026, rho=1.2, cp=1005.0)

# Optical constants of copper at a green (~532 nm) pump, used for the flux-BC validity check.
COPPER_REFLECTIVITY_DEFAULT = 0.60
COPPER_ABSORPTION_DEPTH = 13e-9   # 1/alpha_abs, m
