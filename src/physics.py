import numpy as np
from scipy import integrate

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
        H = np.asarray(H_field, dtype=float)
        B = np.asarray(B_field, dtype=float)
        t = np.asarray(time, dtype=float)

        if not (len(H) == len(B) == len(t)):
            raise ValueError("H_field, B_field, and time must have the same length")
        if len(t) < 2:
            return 0.0
        if np.any(np.diff(t) <= 0):
            raise ValueError("time must be strictly increasing")

        # 1. Calculate the total energy density over the captured duration (J/m^3)
        # Fix for NumPy 2.x: np.trapz is removed, use scipy.integrate.trapezoid
        total_energy_vol = integrate.trapezoid(H, B)
        
        # 2. Get the actual duration of the captured trace
        duration = t[-1] - t[0]
        if duration <= 0:
            return 0.0
        
        # 3. Average Power Density (W/m^3) = Total Energy / Total Time
        # This is mathematically equivalent to (Energy_per_cycle * frequency)
        avg_power_vol = total_energy_vol / duration
        
        # 4. specific_loss (W/kg) = (W/m^3) / (kg/m^3)
        specific_loss = abs(avg_power_vol) / density
        return specific_loss

    @staticmethod
    def calculate_differential_permeability(H_field, B_field, time_array):
        """
        Calculates Relative Differential Permeability: mu_diff(t) = (dB/dt) / (dH/dt) / mu_0
        """
        mu_0 = 4 * np.pi * 1e-7

        H = np.asarray(H_field, dtype=float)
        B = np.asarray(B_field, dtype=float)
        t = np.asarray(time_array, dtype=float)

        if not (len(H) == len(B) == len(t)):
            raise ValueError("H_field, B_field, and time_array must have the same length")
        if len(t) < 3:
            return np.full_like(H, np.nan, dtype=float)
        if np.any(np.diff(t) <= 0):
            raise ValueError("time_array must be strictly increasing")

        edge_order = 2 if len(t) >= 3 else 1
        dB_dt = np.gradient(B, t, edge_order=edge_order)
        dH_dt = np.gradient(H, t, edge_order=edge_order)

        with np.errstate(divide='ignore', invalid='ignore'):
            mu_diff = dB_dt / dH_dt

        mu_diff_r = mu_diff / mu_0

        # Mask singularities where dH/dt is near zero.
        # Use a scale-aware threshold so masking works across different amplitudes.
        slope_scale = np.nanmax(np.abs(dH_dt))
        threshold = max(1e-7, slope_scale * 1e-3)
        mu_diff_r[np.abs(dH_dt) <= threshold] = np.nan
        mu_diff_r[~np.isfinite(mu_diff_r)] = np.nan

        return mu_diff_r

    @staticmethod
    def calculate_hysteresis_params(H, B):
        """
        Extracts key magnetic parameters from the B-H loop.
        """
        # 1. Peak Values
        H_peak = np.max(np.abs(H))
        B_peak = np.max(np.abs(B))
        
        # 2. Coercivity (Hc): H value where B crosses 0
        sign_changes = np.where(np.diff(np.signbit(B)))[0]
        Hc_values = []
        for idx in sign_changes:
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
        
        # 4. Relative Amplitude Permeability (mu_amp)
        mu_0 = 4 * np.pi * 1e-7
        mu_amp = (B_peak / H_peak) / mu_0 if H_peak > 0 else 0.0
            
        return {
            "B_peak": B_peak,
            "H_peak": H_peak,
            "Hc": Hc,
            "Br": Br,
            "mu_amp": mu_amp
        }
