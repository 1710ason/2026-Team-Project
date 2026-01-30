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
        Calculates specific core loss (W/kg).
        
        Logic:
            Energy per unit volume (J/m^3) = Area of B-H loop = Integral(H dB)
            Power per unit mass (W/kg) = (Energy_per_volume * Frequency) / Density
            
        Args:
            H_field: Array of H values [A/m]
            B_field: Array of B values [T]
            time: Array of time values [s]
            frequency: Fundamental frequency of the waveform [Hz]
            density: Material density [kg/m^3]
        
        Returns:
            specific_loss (W/kg)
        """
        # Calculate dB/dt to use as the differential
        db_dt = np.gradient(B_field, time)
        
        # Power density (W/m^3) = H * dB/dt
        p_density = H_field * db_dt
        
        # Average power density over the whole cycle
        # We integrate p_density over time for a full cycle and then divide by period,
        # or just take the mean of the instantaneous power density.
        avg_power_vol = np.mean(p_density)
        
        # specific_loss (W/kg) = (W/m^3) / (kg/m^3)
        specific_loss = avg_power_vol / density
        return abs(specific_loss)
