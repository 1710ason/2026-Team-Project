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

    @staticmethod
    def calculate_hysteresis_params(H, B):
        """
        Extracts key magnetic parameters from the B-H loop.
        """
        # 1. Peak Values
        H_peak = np.max(np.abs(H))
        B_peak = np.max(np.abs(B))
        
        # 2. Coercivity (Hc): H value where B crosses 0
        # We find indices where signs change
        sign_changes = np.where(np.diff(np.signbit(B)))[0]
        Hc_values = []
        for idx in sign_changes:
            # Linear interpolation for better accuracy
            b1, b2 = B[idx], B[idx+1]
            h1, h2 = H[idx], H[idx+1]
            if b2 != b1:
                h_zero = h1 - b1 * ((h2 - h1) / (b2 - b1))
                Hc_values.append(abs(h_zero))
        Hc = np.mean(Hc_values) if Hc_values else 0.0

        # 3. Remanence (Br): B value where H crosses 0
        sign_changes_h = np.where(np.diff(np.signbit(H)))[0]
        Br_values = []
        for idx in sign_changes_h:
            h1, h2 = H[idx], H[idx+1]
            b1, b2 = B[idx], B[idx+1]
            if h2 != h1:
                b_zero = b1 - h1 * ((b2 - b1) / (h2 - h1))
                Br_values.append(abs(b_zero))
        Br = np.mean(Br_values) if Br_values else 0.0
        
        # 4. Relative Amplitude Permeability (mu_r)
        # mu = B / H, mu_r = mu / mu_0
        mu_0 = 4 * np.pi * 1e-7
        if H_peak > 0:
            mu_amp = (B_peak / H_peak) / mu_0
        else:
            mu_amp = 0.0
            
        return {
            "B_peak": B_peak,
            "H_peak": H_peak,
            "Hc": Hc,
            "Br": Br,
            "mu_r": mu_amp
        }
