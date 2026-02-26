import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import HardwareConstants
from src.analyzer import TransformerCoreAnalyzer
from src.physics import Magnetics


def _save_batch_summary(results, output_dir="output"):
    """
    Saves consolidated metrics and trend plots required by the client.
    """
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)
    summary_df = pd.DataFrame(results).sort_values(by=["frequency_hz", "file"])
    summary_csv_path = os.path.join(output_dir, "batch_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # Plot required client outputs vs frequency
    metrics = [
        ("specific_loss_w_per_kg", "Hysteresis Loss vs Frequency", "Specific Loss (W/kg)", "loss_vs_frequency.png"),
        ("b_peak_t", "Max Flux Excursion vs Frequency", "B_peak (T)", "max_flux_excursion_vs_frequency.png"),
        ("mu_amp_relative", "Amplitude Permeability vs Frequency", "Relative Amplitude Permeability", "mu_amp_vs_frequency.png"),
    ]

    for metric_key, title, ylabel, filename in metrics:
        grouped = summary_df.groupby("frequency_hz", as_index=False)[metric_key].mean()

        plt.figure(figsize=(7, 4.5))
        plt.plot(grouped["frequency_hz"], grouped[metric_key], marker="o", linestyle="-")
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def main():
    # 1. Initialize Configuration (HWR90/32 Core)
    hw_config = HardwareConstants.default_hwr90_32()
    
    # 2. Initialize the Analyzer
    analyzer = TransformerCoreAnalyzer(hw_config)
    
    # 3. Locate data files
    data_path = "data/*.csv"
    data_files = glob.glob(data_path)
    
    if not data_files:
        print("No CSV files found in 'data/' folder. Please add experimental data.")
        return

    print(f"Found {len(data_files)} files. Starting batch processing...\n")
    batch_results = []

    # 4. Batch Process
    for file_path in data_files:
        filename = os.path.basename(file_path)
        print(f"--- Analyzing: {filename} ---")
        
        try:
            # Load Data
            df = analyzer.load_data(file_path)
            
            # Infer frequency from filename using Regex
            # Example filename: 'data_400Hz_1.5T.csv' or 'GOSS_50Hz_TEST.csv'
            match = re.search(r"(\d+)Hz", filename, re.IGNORECASE)
            if match:
                frequency = int(match.group(1))
                print(f"  [Info] Extracted Frequency: {frequency}Hz")
            else:
                frequency = 50
                print(f"  [Info] Frequency not found in filename. Defaulting to {frequency}Hz.")
            
            # Execute Pipeline
            # Note: We assume the CSV has 'Time', 'Ch1_Voltage', 'Ch2_Voltage' columns after load_data standardization
            H, B, loss, mu_diff_r = analyzer.analyze_waveform(
                df['Time'].values, 
                df['Ch1_Voltage'].values, 
                df['Ch2_Voltage'].values,
                frequency
            )
            
            mag_params = Magnetics.calculate_hysteresis_params(H, B)
            
            # Save Visualizations & Output Metrics
            analyzer.save_results(H, B, loss, mu_diff_r, filename.replace(".csv", ""))
            
            print(f"Results for {filename}:")
            print(f"  Specific Loss: {loss:.4f} W/kg")
            print(f"  B_peak:        {mag_params['B_peak']:.4f} T")
            print(f"  H_peak:        {mag_params['H_peak']:.4f} A/m")
            print(f"  Coercivity:    {mag_params['Hc']:.4f} A/m")
            print(f"  Remanence:     {mag_params['Br']:.4f} T")
            print(f"  Rel. Amp Perm: {mag_params['mu_amp']:.2f}")
            print("-" * 30 + "\n")

            # Collect per-file outputs for required frequency-dependent summaries
            batch_results.append({
                "file": filename,
                "frequency_hz": frequency,
                "specific_loss_w_per_kg": float(loss),
                "b_peak_t": float(mag_params["B_peak"]),
                "h_peak_a_per_m": float(mag_params["H_peak"]),
                "max_flux_excursion_tpp": float(2.0 * np.max(np.abs(B))),
                "coercivity_a_per_m": float(mag_params["Hc"]),
                "remanence_t": float(mag_params["Br"]),
                "mu_amp_relative": float(mag_params["mu_amp"]),
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}\n")

    _save_batch_summary(batch_results, output_dir="output")
    print("Saved batch summary:")
    print("  output/batch_summary.csv")
    print("  output/loss_vs_frequency.png")
    print("  output/max_flux_excursion_vs_frequency.png")
    print("  output/mu_amp_vs_frequency.png")

if __name__ == "__main__":
    main()
