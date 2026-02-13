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
    def save_comparison_plots(self, time, debug, filename, output_dir="output"):
        """
        Saves before vs after moving-average comparison plots.
        Produces:
          1) Induced voltage (clean vs smoothed)
          2) B(t) before vs after
          3) B-H loop before vs after
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        t = time
        H = debug["H"]
        v0 = debug["v_sec_clean"]
        v1 = debug["v_sec_smooth"]
        B0 = debug["B_before"]
        B1 = debug["B_after"]

        loss0 = debug["loss_before"]
        loss1 = debug["loss_after"]

        # --- 1) Voltage comparison ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t, v0, label="Vsec (DC removed)")
        ax.plot(t, v1, label="Vsec (moving average)")
        ax.set_title(f"Induced voltage smoothing: {filename}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{filename}_vsec_compare.png"), dpi=200)
        plt.close(fig)

        # --- 2) B(t) comparison ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t, B0, label=f"B before (loss={loss0:.4f} W/kg)")
        ax.plot(t, B1, label=f"B after  (loss={loss1:.4f} W/kg)")
        ax.set_title(f"B-field comparison: {filename}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("B (T)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{filename}_B_compare.png"), dpi=200)
        plt.close(fig)

        # --- 3) B-H loop comparison ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(H, B0, label="Before smoothing")
        ax.plot(H, B1, label="After smoothing")
        ax.set_title(f"B-H loop compare: {filename}\nLoss before={loss0:.4f}, after={loss1:.4f} W/kg")
        ax.set_xlabel("H (A/m)")
        ax.set_ylabel("B (T)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{filename}_BH_compare.png"), dpi=200)
        plt.close(fig)

        
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

    def analyze_waveform(
        self,
        time_col,
        ch1_volts,
        ch2_volts,
        frequency,
        ma_window: int = 1,          # NEW
        return_debug: bool = False   # NEW
    ):
        """
        Full pipeline on a single dataset.

        New:
            - Optional moving-average smoothing on Ch2 before integration.
            - Optional debug return of intermediate arrays for plotting.
        """

        # Step 1: H-field from shunt voltage
        H = Magnetics.calculate_H_field(ch1_volts, self.config)

        # Step 2: remove DC from induced voltage
        v_sec_clean = SignalProcessor.remove_dc_offset(ch2_volts)

        # NEW Step 2b: optional moving average on induced voltage (pre-integration)
        # choose a window in seconds (start small!)
        # 0.0002 s = 0.2 ms ~ 1% of a 50 Hz period
        MA_WINDOW_S = 2e-4
        v_sec_smooth = SignalProcessor.moving_average_time(v_sec_clean, time_col, MA_WINDOW_S)


        # Step 3/4: integrate and drift correct (both paths)
        flux_raw_before = SignalProcessor.integrate_cumulative(v_sec_clean, time_col)
        flux_corr_before = SignalProcessor.apply_drift_correction(flux_raw_before)
        B_before = Magnetics.calculate_B_field_scaling(flux_corr_before, self.config)

        flux_raw_after = SignalProcessor.integrate_cumulative(v_sec_smooth, time_col)
        flux_corr_after = SignalProcessor.apply_drift_correction(flux_raw_after)
        B_after = Magnetics.calculate_B_field_scaling(flux_corr_after, self.config)

        # Step 6: loss both paths
        loss_before = Magnetics.calculate_core_loss_density(H, B_before, time_col, frequency, self.config.Density)
        loss_after  = Magnetics.calculate_core_loss_density(H, B_after,  time_col, frequency, self.config.Density)

        if return_debug:
            debug = {
                "H": H,
                "v_sec_clean": v_sec_clean,
                "v_sec_smooth": v_sec_smooth,
                "B_before": B_before,
                "B_after": B_after,
                "loss_before": loss_before,
                "loss_after": loss_after,
            }
            return H, B_after, loss_after, debug

        # default return keeps old style but returns the "after" pipeline
        return H, B_after, loss_after

    def save_results(self, H, B, loss, filename, output_dir="output"):
        """
        Generates a plot of the B-H Loop and saves metrics to a file.
        Also shows the plot interactively.
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

        plt.show()     # <-- THIS makes it appear interactively

        # Optional: remove close if you want window to stay open
        # plt.close()

        print(f"Result saved: {plot_path}")
