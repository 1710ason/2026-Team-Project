from dataclasses import dataclass

@dataclass
class HardwareConstants:
    """
    Data Class to hold physical parameters of the Transformer Core.
    
    This ensures that if we change the core (e.g., from 0.3mm to 0.1mm steel),
    we only update this file.
    """
    N1: int       # Number of Primary Turns (Excitation)
    N2: int       # Number of Secondary Turns (Sensing)
    Ae: float     # Effective Cross-Sectional Area [m^2]
    Lm: float     # Mean Magnetic Path Length [m]
    Rsense: float # Shunt Resistor Value [Ohms]
    Density: float # Material Density [kg/m^3]

    @classmethod
    def default_hwr90_32(cls):
        """Factory method returning the standard config for HWR90/32 core."""
        return cls(
            N1=37,
            N2=20,
            Ae=1.058e-3,
            Lm=0.3,
            Rsense=21,
            Density=7650.0
        )
