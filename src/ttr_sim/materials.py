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

# Experimental wavelengths: pulsed pump at 532 nm, CW probe at 632.8 nm.
PUMP_WAVELENGTH_NM = 532.0
PROBE_WAVELENGTH_NM = 632.8

# Optical constants of copper at the 532 nm pump (absorption of the heating pulse).
COPPER_REFLECTIVITY_DEFAULT = 0.60      # normal-incidence reflectance of clean Cu near 532 nm (~0.6)
COPPER_ABSORPTION_DEPTH = 13e-9         # 1/alpha_abs at 532 nm, m; used for the surface-flux validity check

# Thermoreflectance coefficient (dR/dT)/R of copper at the 632.8 nm probe, 1/K.  Literature values for
# Cu near 630 nm are of order -1e-4 to -2e-4 1/K (negative: reflectance drops as T rises).  Surface
# condition changes this by tens of percent, so calibrate against a reference of known temperature
# when absolute dR/R matters.
COPPER_C_TR_PROBE = -1.5e-4
