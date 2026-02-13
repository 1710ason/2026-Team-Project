import numpy as np
from scipy import integrate, signal

# src/dsp_utils.py
import numpy as np
from scipy import integrate, signal

class SignalProcessor:
    ...


    @staticmethod
    def moving_average_time(x: np.ndarray, t: np.ndarray, window_s: float, mode: str = "reflect") -> np.ndarray:
        """
        Centered moving average where the window is defined in SECONDS.
        This makes smoothing consistent across files with different sample rates.
        """
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)

        dt = np.median(np.diff(t))
        if dt <= 0:
            return x.copy()

        M = int(round(window_s / dt))
        if M <= 1:
            return x.copy()

        if M % 2 == 0:
            M += 1

        pad = M // 2
        x_pad = np.pad(x, pad_width=pad, mode=mode)
        kernel = np.ones(M, dtype=float) / M
        y = np.convolve(x_pad, kernel, mode="valid")
        return y
    
    
    @staticmethod
    def moving_average(x: np.ndarray, window: int, mode: str = "reflect") -> np.ndarray:
        """
        Centered moving average (zero-phase) using symmetric padding.

        Args:
            x: input signal (1D array)
            window: number of samples in the averaging window (must be >= 1)
            mode: padding mode for np.pad. 'reflect' is usually good for waveforms.

        Returns:
            Smoothed signal (same length as x)

        Notes:
            - Centered MA avoids time-lag/phase shift compared to a causal MA.
            - window should be odd for perfectly symmetric centering (we enforce it).
        """
        x = np.asarray(x, dtype=float)

        if window <= 1:
            return x.copy()

        # enforce odd window for true centering
        if window % 2 == 0:
            window += 1

        pad = window // 2
        x_pad = np.pad(x, pad_width=pad, mode=mode)

        kernel = np.ones(window, dtype=float) / window
        y = np.convolve(x_pad, kernel, mode="valid")  # returns len(x)
        return y


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
