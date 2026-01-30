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
        print(f"Analyzing: {filename}")
        
        try:
            # Load Data
            df = analyzer.load_data(file_path)
            
            # Infer frequency from filename if possible, otherwise default to 50Hz
            # Example filename: 'data_400Hz_1.5T.csv'
            frequency = 50
            if "400Hz" in filename: frequency = 400
            elif "100Hz" in filename: frequency = 100
            
            # Execute Pipeline
            # Note: We assume the CSV has 'Time', 'Ch1_Voltage', 'Ch2_Voltage' columns
            H, B, loss = analyzer.analyze_waveform(
                df['Time'].values, 
                df['Ch1_Voltage'].values, 
                df['Ch2_Voltage'].values,
                frequency
            )
            
            # Save Visualizations & Output Metrics
            analyzer.save_results(H, B, loss, filename.replace(".csv", ""))
            
            print(f"Success: Loss = {loss:.4f} W/kg\n")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}\n")

if __name__ == "__main__":
    main()
