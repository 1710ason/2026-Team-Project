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
        adaptive_limit = max(polyorder + 2, n // 20) #Finding 5%
        actual_window = min(window_length, adaptive_limit, n) #Capping at 5%
        
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

    @staticmethod
    def cut_to_integer_cycles(reference_signal, *args):
        """
        Slices all input arrays to contain only complete cycles.
        
        Why: Calculating average power over 2.7 cycles introduces significant error 
             because the partial 0.7 cycle might be high-loss or low-loss.
             We must integrate over exactly N cycles.
        
        Args:
            reference_signal: Signal to use for sync (e.g., H-field).
            *args: Other arrays to slice (B, Time, etc.)
            
        Returns:
            Tuple of sliced arrays (ref_sliced, arg1_sliced, arg2_sliced...)
        """
        data = np.asarray(reference_signal)
        
        # Find zero crossings (where sign changes)
        # signbit is True for negative, False for positive
        sign_changes = np.diff(np.signbit(data))
        crossing_indices = np.where(sign_changes)[0]
        
        if len(crossing_indices) < 2:
            # Fewer than 1 full half-cycle found, cannot slice properly. Return original.
            return (reference_signal,) + args
            
        # Filter for Positive-Going Crossings (Slope > 0)
        # We look at the value after the crossing to see if it's positive
        # indices point to the element BEFORE the crossing
        
        pos_slope_indices = []
        for idx in crossing_indices:
            if idx + 1 < len(data):
                if data[idx] < data[idx+1]: # Positive slope
                    pos_slope_indices.append(idx)
                    
        if len(pos_slope_indices) < 2:
            # Fewer than 1 full cycle found
            return (reference_signal,) + args

        # We need at least 2 positive-going crossings to define 1 full cycle.
        # We start at the first crossing and end at the last crossing.
        start_idx = pos_slope_indices[0]
        end_idx = pos_slope_indices[-1]
        
        # Slice all arrays
        # [start_idx : end_idx] covers exactly N cycles.
        sliced_ref = reference_signal[start_idx : end_idx]
        sliced_args = tuple(a[start_idx : end_idx] for a in args)
        
        return (sliced_ref,) + sliced_args
