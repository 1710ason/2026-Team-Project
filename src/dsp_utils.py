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

    @staticmethod
    def smooth_signal(data_array, window_length=51, polyorder=3):
        """
        Applies a Savitzky-Golay filter to smooth the signal while preserving peak shapes.
        
        Why: Raw oscilloscope data contains high-frequency noise. Simple moving averages
             distort the sharp tips of magnetic hysteresis loops (saturation).
             Savitzky-Golay fits a local polynomial, preserving these features better.
        
        Args:
            data_array: Input 1D array.
            window_length: Ideal length of the filter window. The actual window will be 
                           clamped to ~5% of the data length to prevent distortion 
                           at low sampling rates.
            polyorder: Order of the polynomial to fit (default 3 for cubic).
        """
        data = np.asarray(data_array, dtype=float)
        if data.ndim != 1:
            raise ValueError("smooth_signal expects a 1D array")

        n = len(data)
        if n < polyorder + 2:
            return data  # Data too short to smooth effectively

        # Adaptive Window: Don't let the window exceed 5% of total points 
        # (or at least enough points to fit the polynomial)
        adaptive_limit = max(polyorder + 2, n // 20)
        actual_window = min(window_length, adaptive_limit, n)
        
        # Ensure window_length is odd
        if actual_window % 2 == 0:
            actual_window = max(polyorder + 1, actual_window - 1)
            if actual_window % 2 == 0:
                actual_window += 1

        # Guard against edge cases where adjustments push window out of valid range.
        if actual_window > n:
            actual_window = n if n % 2 == 1 else n - 1
        if actual_window <= polyorder:
            return data
        
        return signal.savgol_filter(data, actual_window, polyorder)
