import numpy as np
from scipy import integrate, signal

class SignalProcessor:
    """
    Static utility class for Digital Signal Processing.
    Handles integration, filtering, and drift correction.
    """

    @staticmethod
    def remove_dc_offset(data_array):
        """
        Removes the DC component (mean) from a raw signal.
        
        Why: Oscilloscopes often have a slight DC bias that ruins integration.
        Logic: signal = signal - mean(signal)
        """
        return data_array - np.mean(data_array)

    @staticmethod
    def integrate_cumulative(y_data, x_time):
        """
        Performs cumulative trapezoidal integration.
        
        Args:
            y_data: Array of values to integrate (e.g., Voltage)
            x_time: Time array corresponding to y_data
        Returns:
            Integrated array (same length as input).
        """
        # initial=0 ensures the output array has the same length as input
        return integrate.cumulative_trapezoid(y_data, x_time, initial=0)


    @staticmethod
    def apply_drift_correction(integrated_signal):
        """
        CRITICAL FUNCTION: Removes linear drift from integration.
        
        Why: Numerical integration of AC noise creates a 'random walk' or slope.
             This prevents the B-H loop from closing (it looks like a spiral).
        
        Logic:
            1. Use scipy.signal.detrend(data, type='linear') to remove the slope.
            2. Re-center the result around 0 (remove new mean).
        """
        detrended = signal.detrend(integrated_signal, type='linear')
        return detrended - np.mean(detrended)
