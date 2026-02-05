import numpy as np

class Magnetics:
    """
    Library of magnetic formulas. 
    Decouples 'Physics' from 'Data Processing'.
    """

    @staticmethod
    def calculate_H_field(v_shunt, hardware_params):
        """
        Applies Ampere's Law to calculate Magnetic Field Strength (H).
        
        Formula: H(t) = (N1 * I(t)) / Lm
        Where: I(t) = v_shunt(t) / R_sense
        
        Returns:
            H_field array [A/m]
        """
        current = v_shunt / hardware_params.Rsense
        h_field = (hardware_params.N1 * current) / hardware_params.Lm
        return h_field

    @staticmethod
    def calculate_B_field_scaling(integrated_voltage, hardware_params):
        """
        Applies scaling factors for Faraday's Law after integration.
        
        Formula: B(t) = - (1 / (N2 * Ae)) * Integrated_Volts
        
        Returns:
            B_field array [Tesla]
        """
        b_field = - (1 / (hardware_params.N2 * hardware_params.Ae)) * integrated_voltage
        return b_field

    @staticmethod
    def calculate_core_loss_density(H_field, B_field, time, frequency, density):
        """
        Calculates specific core loss (W/kg) using the B-H loop area method.
        
        Logic:
            Energy per unit volume (J/m^3) = Integral(H dB)
            Power per unit mass (W/kg) = (Total Energy / Total Time) / Density
            
        This method is superior to H*dB/dt because integration naturally filters 
        high-frequency measurement noise.
        """
        # 1. Calculate the total energy density over the captured duration (J/m^3)
        # We use np.trapz(y, x) which computes Integral(y dx)
        total_energy_vol = np.trapz(H_field, B_field)
        
        # 2. Get the actual duration of the captured trace
        duration = time[-1] - time[0]
        if duration <= 0:
            return 0.0
        
        # 3. Average Power Density (W/m^3) = Total Energy / Total Time
        # This is mathematically equivalent to (Energy_per_cycle * frequency)
        avg_power_vol = total_energy_vol / duration
        
        # 4. specific_loss (W/kg) = (W/m^3) / (kg/m^3)
        specific_loss = abs(avg_power_vol) / density
        return specific_loss
