import os
import glob
from src.config import HardwareConstants
from src.analyzer import TransformerCoreAnalyzer

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

    # 4. Batch Process
    for file_path in data_files:
        filename = os.path.basename(file_path)
        print(f"--- Analyzing: {filename} ---")
        
        try:
            # Load Data
            df = analyzer.load_data(file_path)
            
            # Infer frequency from filename if possible
            # Example filename: 'data_400Hz_1.5T.csv'
            fname_lower = filename.lower()
            frequency = 50
            if "400hz" in fname_lower: 
                frequency = 400
            elif "100hz" in fname_lower: 
                frequency = 100
            else:
                print(f"  [Info] Frequency not found in filename. Defaulting to {frequency}Hz.")
            
            # Execute Pipeline
            # Note: We assume the CSV has 'Time', 'Ch1_Voltage', 'Ch2_Voltage' columns after load_data standardization
            H, B, loss = analyzer.analyze_waveform(
                df['Time'].values, 
                df['Ch1_Voltage'].values, 
                df['Ch2_Voltage'].values,
                frequency
            )
            
            # Calculate detailed magnetic properties
            from src.physics import Magnetics
            mag_params = Magnetics.calculate_hysteresis_params(H, B)
            
            # Save Visualizations & Output Metrics
            analyzer.save_results(H, B, loss, filename.replace(".csv", ""))
            
            print(f"Results for {filename}:")
            print(f"  Specific Loss: {loss:.4f} W/kg")
            print(f"  B_peak:        {mag_params['B_peak']:.4f} T")
            print(f"  H_peak:        {mag_params['H_peak']:.4f} A/m")
            print(f"  Coercivity:    {mag_params['Hc']:.4f} A/m")
            print(f"  Remanence:     {mag_params['Br']:.4f} T")
            print(f"  Rel. Perm.:    {mag_params['mu_r']:.2f}")
            print("-" * 30 + "\n")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}\n")

if __name__ == "__main__":
    main()
