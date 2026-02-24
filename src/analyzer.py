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
            
            # 1. Time / X-Axis
            # Must contain 'time' or be a specific time unit, but NOT 'secondary'
            if ('time' in c_low or 'x-axis' in c_low or 
                (any(unit in c_low for unit in ['second', 'sec', '(s)', '(ms)']) and 'secondary' not in c_low)):
                mapping[col] = 'Time'
            
            # 2. Ch1_Voltage (Primary / Current / Shunt)
            elif any(k in c_low for k in ['ch1', 'primary', 'shunt', 'channel a', 'input', 'current']) or str(col) == '1':
                mapping[col] = 'Ch1_Voltage'
                
            # 3. Ch2_Voltage (Secondary / Induced / Volt)
            elif any(k in c_low for k in ['ch2', 'secondary', 'induced', 'channel b', 'output', 'volt']) or str(col) == '2':
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
        
        # Step 6: Truncate to Integer Cycles
        # We use H as the reference for zero crossings to ensure we integrate over exactly N cycles.
        # This prevents errors from partial cycles (e.g. 2.7 cycles).
        H_cut, B_cut, time_cut = SignalProcessor.cut_to_integer_cycles(H, B, time_col)
        
        # Step 7: Calculate Specific Core Loss (using truncated data)
        loss = Magnetics.calculate_core_loss_density(H_cut, B_cut, time_cut, frequency, self.config.Density)
        
        # Step 8: Calculate Differential Permeability (using truncated data)
        mu_r = Magnetics.calculate_differential_permeability(H_cut, B_cut, time_cut)
        
        return H_cut, B_cut, loss, mu_r

    def save_results(self, H, B, loss, mu_diff_r, filename, output_dir="output"):
        """
        Generates plots of the B-H Loop and permeability metrics, saving them to files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.figure(figsize=(15, 5))

        # Plot 1: B-H Loop
        plt.subplot(1, 3, 1)
        plt.plot(H, B, 'b-')
        plt.title(f"B-H Loop: {filename}\nLoss: {loss:.4f} W/kg")
        plt.xlabel("H (A/m)")
        plt.ylabel("B (T)")
        plt.grid(True)
        
        # Plot 2: Permeability vs H
        plt.subplot(1, 3, 2)
        plt.plot(H, mu_diff_r, 'r-')
        plt.title("Rel. Differential Permeability vs H")
        plt.xlabel("H (A/m)")
        plt.ylabel("$\\mu_{{diff,r}}$")
        plt.grid(True)
        
        # Set reasonable y-limits to avoid auto-scaling on noise spikes
        valid_mu = mu_diff_r[~np.isnan(mu_diff_r)]
        if len(valid_mu) > 0:
            upper = max(1.0, np.nanpercentile(valid_mu, 98) * 1.2)
            plt.ylim(0, upper)

        # Plot 3: Permeability around the hysteresis curve (color-mapped on B-H)
        plt.subplot(1, 3, 3)
        scatter = plt.scatter(
            H,
            B,
            c=mu_diff_r,
            cmap="viridis",
            s=6,
            linewidths=0
        )
        plt.title("Permeability Around Hysteresis Loop")
        plt.xlabel("H (A/m)")
        plt.ylabel("B (T)")
        plt.grid(True)
        cbar = plt.colorbar(scatter)
        cbar.set_label("$\\mu_{{diff,r}}$")
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{filename}_analysis.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Result saved: {plot_path}")
