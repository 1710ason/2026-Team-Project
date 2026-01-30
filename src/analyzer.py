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
        Reads CSV using pandas. Expects columns: Time, Ch1_Voltage, Ch2_Voltage.
        
        Args:
            filepath: Path to the CSV file.
        Returns:
            pandas DataFrame.
        """
        # Load data, assuming headers exist. Skip header if necessary.
        df = pd.read_csv(filepath)
        # Rename columns to standard names for internal consistency if needed
        # Expected: Time, Ch1_Voltage, Ch2_Voltage
        return df

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
