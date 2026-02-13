import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from .config import HardwareConstants
from .dsp_utils import SignalProcessor
from .physics import Magnetics

class TransformerCoreAnalyzer:
    """
    Main controller class. 
    Instantiate this once with hardware config, then process multiple files.
    """

    def __init__(self, config: HardwareConstants):
        self.config = config

    def load_data(self, filepath):
        """
        Reads CSV and maps columns to standard names: Time, Ch1_Voltage, Ch2_Voltage.
        Cleans common oscilloscope export quirks (units row, trailing comma/Unnamed cols).
        """
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV {filepath}: {e}")

        # Drop empty/extra columns caused by trailing commas (e.g. "Unnamed: 3")
        df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

        # Strip whitespace/trailing commas from headers
        df.columns = [str(c).strip().rstrip(",") for c in df.columns]

        # Standardize column names (case-insensitive and partial matching)
        mapping = {}
        target_cols = ['Time', 'Ch1_Voltage', 'Ch2_Voltage']

        for col in df.columns:
            c_low = str(col).lower()
            if 'time' in c_low or 'second' in c_low or 'x-axis' in c_low:
                mapping[col] = 'Time'
            elif 'ch1' in c_low or 'current' in c_low or 'shunt' in c_low or 'channel a' in c_low or 'input' in c_low or str(col) == '1':
                mapping[col] = 'Ch1_Voltage'
            elif 'ch2' in c_low or 'volt' in c_low or 'sec' in c_low or 'channel b' in c_low or 'output' in c_low or str(col) == '2':
                mapping[col] = 'Ch2_Voltage'

        # If we didn't find clear matches, assume columns 0, 1, 2
        if len(set(mapping.values())) < 3:
            print(f"  [Notice] Unclear headers in {os.path.basename(filepath)}. Using index-based mapping (0,1,2).")
            df = df.iloc[:, :3].copy()  # keep only first 3 cols to be safe
            df.columns = target_cols
        else:
            df = df.rename(columns=mapping)
            df = df[target_cols].copy()

        # ---- CRITICAL: Handle Units & Convert Time ----
        # Check first row for units (e.g., "(ms)", "(V)")
        time_scaling_factor = 1.0
        if not df.empty:
            first_row = df.iloc[0].astype(str).tolist()
            # Check if Time column (index 0) has 'ms' in the units row
            if 'ms' in str(first_row[0]).lower():
                time_scaling_factor = 1e-3
                print(f"  [Info] Detected 'ms' time units in {os.path.basename(filepath)}. Converting to seconds.")
            
            # If any value in the first row looks like a unit, drop the row
            if any(x for x in first_row if "(" in x or ")" in x):
                df = df.iloc[1:].reset_index(drop=True)

        # Convert to numeric, drop any rows that didn't convert
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        
        # Apply Time Scaling
        df['Time'] = df['Time'] * time_scaling_factor

        return df

    def analyze_waveform(self, time_col, ch1_volts, ch2_volts, frequency):
        """
        Executes the full pipeline on a single dataset.
        
        Flow:
            1. DSP: Smooth raw signals (Current & Voltage) to remove quantization noise.
            2. Physics: Convert Ch1 -> Current -> H_field.
            3. DSP: Remove DC from Ch2 (induced voltage).
            4. DSP: Integrate Ch2 to get raw Flux.
            5. DSP: Apply Drift Correction (Detrending) to Flux.
            6. Physics: Scale Flux -> B_field (Tesla).
            7. Physics: Calculate Loss (W/kg).
            8. Physics: Calculate Differential Permeability.
            
        Returns:
            H_field, B_field, specific_loss, mu_r
        """
        
        # Step 0: Pre-processing (Smoothing)
        # Use a window relative to frequency if possible, or a safe default
        # For now, default 51 is used inside smooth_signal
        ch1_smooth = SignalProcessor.smooth_signal(ch1_volts)
        ch2_smooth = SignalProcessor.smooth_signal(ch2_volts)
        
        # Step 1: H-Field calculation from Shunt Voltage
        H = Magnetics.calculate_H_field(ch1_smooth, self.config)
        
        # Step 2: Clean induced voltage
        v_sec_clean = SignalProcessor.remove_dc_offset(ch2_smooth)
        
        # Step 3 & 4: Integrate and Correct Drift to get B-Field
        flux_raw = SignalProcessor.integrate_cumulative(v_sec_clean, time_col)
        flux_corrected = SignalProcessor.apply_drift_correction(flux_raw)
        
        # Step 5: Convert Flux to Magnetic Flux Density (B)
        B = Magnetics.calculate_B_field_scaling(flux_corrected, self.config)
        
        # Step 6: Calculate Specific Core Loss
        loss = Magnetics.calculate_core_loss_density(H, B, time_col, frequency, self.config.Density)
        
        # Step 7: Calculate Differential Permeability
        mu_r = Magnetics.calculate_differential_permeability(H, B, time_col)
        
        return H, B, loss, mu_r

    def save_results(self, H, B, loss, mu_diff_r, filename, output_dir="output"):
        """
        Generates plots of the B-H Loop and Permeability, saving them to files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Plot 1: B-H Loop
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(H, B, 'b-')
        plt.title(f"B-H Loop: {filename}\nLoss: {loss:.4f} W/kg")
        plt.xlabel("H (A/m)")
        plt.ylabel("B (T)")
        plt.grid(True)
        
        # Plot 2: Permeability vs H
        plt.subplot(1, 2, 2)
        # Filter NaNs for plotting (matplotlib handles NaNs by breaking the line, which is desired)
        plt.plot(H, mu_diff_r, 'r-')
        plt.title(f"Rel. Differential Permeability ($\\mu_{{diff,r}}$)")
        plt.xlabel("H (A/m)")
        plt.ylabel("$\\mu_{{diff,r}}$")
        plt.grid(True)
        
        # Set reasonable y-limits to avoid auto-scaling on noise spikes
        valid_mu = mu_diff_r[~np.isnan(mu_diff_r)]
        if len(valid_mu) > 0:
            plt.ylim(0, np.nanpercentile(valid_mu, 98) * 1.2) 
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{filename}_analysis.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Result saved: {plot_path}")
