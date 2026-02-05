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
        """
        # Some scopes have metadata lines, let's try to find the header row
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV {filepath}: {e}")

        # Standardize column names (case-insensitive and partial matching)
        mapping = {}
        target_cols = ['Time', 'Ch1_Voltage', 'Ch2_Voltage']
        
        for col in df.columns:
            c_low = str(col).lower()
            if 'time' in c_low or 'second' in c_low or 'x-axis' in c_low:
                mapping[col] = 'Time'
            elif 'ch1' in c_low or 'current' in c_low or 'shunt' in c_low or col == '1':
                mapping[col] = 'Ch1_Voltage'
            elif 'ch2' in c_low or 'volt' in c_low or 'sec' in c_low or col == '2':
                mapping[col] = 'Ch2_Voltage'

        # If we didn't find clear matches, assume columns 0, 1, 2
        if len(set(mapping.values())) < 3:
            print(f"  [Notice] Unclear headers in {os.path.basename(filepath)}. Using index-based mapping (0,1,2).")
            new_cols = list(df.columns)
            new_cols[0] = 'Time'
            new_cols[1] = 'Ch1_Voltage'
            new_cols[2] = 'Ch2_Voltage'
            df.columns = new_cols
        else:
            df = df.rename(columns=mapping)
            
        return df[target_cols]  # Return only the needed columns

    def analyze_waveform(self, time_col, ch1_volts, ch2_volts, frequency):
        """
        Executes the full pipeline on a single dataset.
        
        Flow:
            1. Physics: Convert Ch1 -> Current -> H_field.
            2. DSP: Remove DC from Ch2 (induced voltage).
            3. DSP: Integrate Ch2 to get raw Flux.
            4. DSP: Apply Drift Correction (Detrending) to Flux.
            5. Physics: Scale Flux -> B_field (Tesla).
            6. Physics: Calculate Loss (W/kg).
            
        Returns:
            H_field, B_field, specific_loss
        """
        
        # Step 1: H-Field calculation from Shunt Voltage
        H = Magnetics.calculate_H_field(ch1_volts, self.config)
        
        # Step 2: Clean induced voltage
        v_sec_clean = SignalProcessor.remove_dc_offset(ch2_volts)
        
        # Step 3 & 4: Integrate and Correct Drift to get B-Field
        flux_raw = SignalProcessor.integrate_cumulative(v_sec_clean, time_col)
        flux_corrected = SignalProcessor.apply_drift_correction(flux_raw)
        
        # Step 5: Convert Flux to Magnetic Flux Density (B)
        B = Magnetics.calculate_B_field_scaling(flux_corrected, self.config)
        
        # Step 6: Calculate Specific Core Loss
        loss = Magnetics.calculate_core_loss_density(H, B, time_col, frequency, self.config.Density)
        
        return H, B, loss

    def save_results(self, H, B, loss, filename, output_dir="output"):
        """
        Generates a plot of the B-H Loop and saves metrics to a file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.figure(figsize=(8, 6))
        plt.plot(H, B)
        plt.title(f"B-H Loop: {filename}\nSpecific Loss: {loss:.4f} W/kg")
        plt.xlabel("H (A/m)")
        plt.ylabel("B (T)")
        plt.grid(True)
        
        plot_path = os.path.join(output_dir, f"{filename}_bh_loop.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Result saved: {plot_path}")
