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
        
        H_cut, B_cut, time_cut = self._prepare_cut_waveforms(time_col, ch1_volts, ch2_volts)
        
        # Step 7: Calculate Average Cycle and Error
        time_rel = time_cut - time_cut[0]
        num_cycles, (H_avg, H_std), args_stats = SignalProcessor.average_cycles(
            H_cut, B_cut, time_rel, points_per_cycle=1000, close_aux=[True, False]
        )
        B_avg, B_std = args_stats[0]
        time_avg, time_std = args_stats[1]
        
        # Step 8: Calculate Specific Core Loss (using averaged data)
        loss = Magnetics.calculate_core_loss_density(H_avg, B_avg, time_avg, frequency, self.config.Density)
        
        # Step 9: Calculate Differential Permeability (using averaged data)
        mu_r = Magnetics.calculate_differential_permeability(H_avg, B_avg, time_avg)
        
        return {
            "H_avg": H_avg,
            "B_avg": B_avg,
            "H_std": H_std,
            "B_std": B_std,
            "loss": loss,
            "mu_r": mu_r,
            "num_cycles": num_cycles,
            "time_avg": time_avg
        }

    def _prepare_cut_waveforms(self, time_col, ch1_volts, ch2_volts):
        """
        Runs per-waveform conditioning and returns integer-cycle-trimmed H, B, and time.
        """
        # Per-file smoothing is intentionally skipped so grouped smoothing happens
        # after cycle pooling/averaging across all files in a condition.
        H = Magnetics.calculate_H_field(ch1_volts, self.config)

        # Cut first, then remove DC and integrate on the exact integer-cycle window.
        # This avoids biasing the integral with partial-cycle mean offsets.
        H_cut, v_sec_cut, time_cut = SignalProcessor.cut_to_integer_cycles(H, ch2_volts, time_col)
        v_sec_clean = SignalProcessor.remove_dc_offset(v_sec_cut)
        flux_raw = SignalProcessor.integrate_cumulative(v_sec_clean, time_cut)
        flux_corrected = SignalProcessor.apply_drift_correction(flux_raw)
        B_cut = Magnetics.calculate_B_field_scaling(flux_corrected, self.config)
        return H_cut, B_cut, time_cut

    def _extract_cycles(self, h_cut, b_cut, time_cut, points_per_cycle=1000):
        """
        Splits trimmed traces into phase-aligned cycles and interpolates each to a common grid.
        """
        h_data = np.asarray(h_cut)
        b_data = np.asarray(b_cut)
        t_rel = np.asarray(time_cut) - time_cut[0]

        sign_changes = np.diff(np.signbit(h_data))
        crossing_indices = np.where(sign_changes)[0]
        pos_slope_indices = [
            idx for idx in crossing_indices if idx + 1 < len(h_data) and h_data[idx] < h_data[idx + 1]
        ]

        if len(pos_slope_indices) < 2:
            raise ValueError("Fewer than 1 cycle found for cycle extraction.")

        num_cycles = len(pos_slope_indices) - 1
        common_phase = np.linspace(0, 1, points_per_cycle)
        h_cycles = np.zeros((num_cycles, points_per_cycle))
        b_cycles = np.zeros((num_cycles, points_per_cycle))
        t_cycles = np.zeros((num_cycles, points_per_cycle))

        for i in range(num_cycles):
            start = pos_slope_indices[i]      # crossing between start and start+1
            end = pos_slope_indices[i + 1]    # crossing between end and end+1

            # Interpolate exact positive-going H=0 crossing points for boundaries.
            dh_start = h_data[start + 1] - h_data[start]
            dh_end = h_data[end + 1] - h_data[end]
            if dh_start == 0 or dh_end == 0:
                raise ValueError("Degenerate zero-crossing slope encountered during cycle extraction.")

            alpha_start = -h_data[start] / dh_start
            alpha_end = -h_data[end] / dh_end
            alpha_start = float(np.clip(alpha_start, 0.0, 1.0))
            alpha_end = float(np.clip(alpha_end, 0.0, 1.0))

            t_start = t_rel[start] + alpha_start * (t_rel[start + 1] - t_rel[start])
            t_end = t_rel[end] + alpha_end * (t_rel[end + 1] - t_rel[end])
            if t_end <= t_start:
                raise ValueError("Invalid cycle boundary timing encountered during extraction.")

            b_start = b_data[start] + alpha_start * (b_data[start + 1] - b_data[start])
            b_end = b_data[end] + alpha_end * (b_data[end + 1] - b_data[end])

            # Build per-cycle vectors with exact crossing boundaries included.
            t_segment = np.concatenate(([t_start], t_rel[start + 1 : end + 1], [t_end]))
            h_segment = np.concatenate(([0.0], h_data[start + 1 : end + 1], [0.0]))
            b_segment = np.concatenate(([b_start], b_data[start + 1 : end + 1], [b_end]))

            local_phase = (t_segment - t_start) / (t_end - t_start)
            h_cycles[i] = np.interp(common_phase, local_phase, h_segment)
            b_cycles[i] = np.interp(common_phase, local_phase, b_segment)
            t_cycles[i] = np.interp(common_phase, local_phase, t_segment - t_start)

        return h_cycles, b_cycles, t_cycles

    def analyze_waveform_group(self, waveforms, frequency, points_per_cycle=1000):
        """
        Pools cycles from multiple files of the same test condition and averages once.
        """
        h_cycle_list = []
        b_cycle_list = []
        t_cycle_list = []

        for i, (time_col, ch1_volts, ch2_volts) in enumerate(waveforms, start=1):
            try:
                h_cut, b_cut, time_cut = self._prepare_cut_waveforms(time_col, ch1_volts, ch2_volts)
                h_cycles, b_cycles, t_cycles = self._extract_cycles(
                    h_cut, b_cut, time_cut, points_per_cycle=points_per_cycle
                )
                h_cycle_list.append(h_cycles)
                b_cycle_list.append(b_cycles)
                t_cycle_list.append(t_cycles)
            except Exception as e:
                print(f"  [Skip] waveform #{i} in group failed preprocessing/cycle extraction: {e}")

        if not h_cycle_list:
            raise ValueError("No valid waveforms available for grouped analysis.")

        h_all = np.vstack(h_cycle_list)
        b_all = np.vstack(b_cycle_list)
        t_all = np.vstack(t_cycle_list)

        h_avg_raw = np.mean(h_all, axis=0)
        b_avg_raw = np.mean(b_all, axis=0)
        t_avg = np.mean(t_all, axis=0)

        h_std_raw = np.std(h_all, axis=0)
        b_std_raw = np.std(b_all, axis=0)

        # Apply SG smoothing after pooling/averaging.
        h_avg = SignalProcessor.smooth_signal(h_avg_raw)
        b_avg = SignalProcessor.smooth_signal(b_avg_raw)
        h_std = SignalProcessor.smooth_signal(h_std_raw)
        b_std = SignalProcessor.smooth_signal(b_std_raw)

        # Enforce periodic closure for magnetic variables only.
        h_avg[-1] = h_avg[0]
        b_avg[-1] = b_avg[0]

        loss = Magnetics.calculate_core_loss_density(h_avg, b_avg, t_avg, frequency, self.config.Density)
        mu_r = Magnetics.calculate_differential_permeability(h_avg, b_avg, t_avg)

        return {
            "H_avg": h_avg,
            "B_avg": b_avg,
            "H_std": h_std,
            "B_std": b_std,
            "loss": loss,
            "mu_r": mu_r,
            "num_cycles": int(h_all.shape[0]),
            "num_files": len(h_cycle_list),
            "time_avg": t_avg,
        }

    def save_results(self, result_dict, filename, output_dir="output"):
        """
        Generates plots of the B-H Loop and permeability metrics, saving them to files.
        """
        H = result_dict["H_avg"]
        B = result_dict["B_avg"]
        H_std = result_dict["H_std"]
        B_std = result_dict["B_std"]
        loss = result_dict["loss"]
        mu_diff_r = result_dict["mu_r"]
        num_cycles = result_dict["num_cycles"]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.figure(figsize=(15, 5))

        # Plot 1: B-H Loop
        plt.subplot(1, 3, 1)
        plt.plot(H, B, 'b-', label='Average Loop')
        plt.fill_between(H, B - B_std, B + B_std, color='b', alpha=0.2, label='±1 Std Dev')
        plt.title(f"B-H Loop: {filename}\nLoss: {loss:.4f} W/kg ({num_cycles} cycles avg)")
        plt.xlabel("H (A/m)")
        plt.ylabel("B (T)")
        plt.legend()
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
