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
            1. Remove endpoint-to-endpoint linear drift so corrected[0] == corrected[-1].
            2. Re-center the result around 0 (remove new mean).
        """
        data = np.asarray(integrated_signal, dtype=float)
        if data.ndim != 1:
            raise ValueError("apply_drift_correction expects a 1D array")
        if len(data) < 2:
            return data - np.mean(data)

        drift = np.linspace(0.0, data[-1] - data[0], len(data))
        corrected = data - drift
        return corrected - np.mean(corrected)

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
            return data
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
            raise ValueError("Fewer than 1 full half-cycle found, cannot slice properly to integer cycles.")
            
        # Filter for Positive-Going Crossings (Slope > 0)
        # We look at the value after the crossing to see if it's positive
        # indices point to the element BEFORE the crossing
        
        pos_slope_indices = []
        for idx in crossing_indices:
            if idx + 1 < len(data):
                if data[idx] < data[idx+1]: # Positive slope
                    pos_slope_indices.append(idx)
                    
        if len(pos_slope_indices) < 2:
            raise ValueError("Fewer than 1 full cycle found, cannot slice properly to integer cycles.")

        # We need at least 2 positive-going crossings to define 1 full cycle.
        # We start just before the first crossing and include the point after
        # the last crossing so the final cycle has a proper closing boundary.
        start_idx = pos_slope_indices[0]
        end_idx = pos_slope_indices[-1]
        stop_idx = min(end_idx + 2, len(data))
        
        # Slice all arrays
        # [start_idx : stop_idx] covers exactly N cycles and includes the
        # sample immediately after the final crossing for a complete endpoint.
        sliced_ref = reference_signal[start_idx : stop_idx]
        sliced_args = tuple(np.asarray(a)[start_idx : stop_idx] for a in args)
        
        return (sliced_ref,) + sliced_args

    @staticmethod
    def average_cycles(reference_signal, *args, points_per_cycle=1000, close_aux=None):
        """
        Splits a continuous sequence of integer cycles into individual loops
        and computes the average loop and pointwise standard deviation (error).
        
        Args:
            reference_signal: The primary signal (e.g. H_field) to identify cycles.
            *args: Any other arrays aligned with reference_signal to average (e.g. B_field, Time).
            points_per_cycle: Number of points to interpolate each cycle to.
            close_aux: Iterable of booleans matching *args that indicates which
                       auxiliary averaged outputs should be loop-closed.
            
        Returns:
            Tuple: (num_cycles, (ref_avg, ref_std), [(arg1_avg, arg1_std), (arg2_avg, arg2_std)...])
        """
        data = np.asarray(reference_signal)
        
        # Find positive-going zero crossings
        sign_changes = np.diff(np.signbit(data))
        crossing_indices = np.where(sign_changes)[0]
        
        pos_slope_indices = []
        for idx in crossing_indices:
            if idx + 1 < len(data):
                if data[idx] < data[idx+1]:
                    pos_slope_indices.append(idx)
                    
        if len(pos_slope_indices) < 2:
            raise ValueError("Fewer than 1 cycle found, cannot compute average.")

        num_cycles = len(pos_slope_indices) - 1
        if num_cycles < 1:
            raise ValueError("Fewer than 1 cycle found, cannot compute average.")

        if close_aux is None:
            close_aux = [True] * len(args)
        else:
            close_aux = list(close_aux)
            if len(close_aux) != len(args):
                raise ValueError("close_aux length must match number of auxiliary signals.")
            
        common_phase = np.linspace(0, 1, points_per_cycle)
        
        # Preallocate arrays
        ref_cycles = np.zeros((num_cycles, points_per_cycle))
        args_cycles = [np.zeros((num_cycles, points_per_cycle)) for _ in args]
        
        for i in range(num_cycles):
            start = pos_slope_indices[i]
            end = pos_slope_indices[i+1]
            # Use end - start + 1 to include the closing point in the phase mapping
            local_phase = np.linspace(0, 1, end - start + 1)
            
            ref_cycles[i] = np.interp(common_phase, local_phase, data[start : end + 1])
            for j, arg in enumerate(args):
                args_cycles[j][i] = np.interp(common_phase, local_phase, np.asarray(arg)[start : end + 1])
                
        # Calculate mean and std along the cycle dimension
        ref_avg = np.mean(ref_cycles, axis=0)
        ref_std = np.std(ref_cycles, axis=0)
        
        # Enforce mathematical loop closure (Periodic Boundary Condition)
        ref_avg[-1] = ref_avg[0]
        
        args_stats = []
        for ac, should_close in zip(args_cycles, close_aux):
            avg_arr = np.mean(ac, axis=0)
            if should_close:
                avg_arr[-1] = avg_arr[0] # Force closure for periodic signals only
            args_stats.append((avg_arr, np.std(ac, axis=0)))
            
        return num_cycles, (ref_avg, ref_std), args_stats
