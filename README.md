# Magnetic Core Analysis Project

This repository contains the software pipeline for characterizing transformer core losses (50Hz - 400Hz).

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

1. Place your oscilloscope CSV data in the `data/` directory.
2. Ensure CSV files have headers roughly matching: `Time`, `Ch1_Voltage` (Current Shunt), `Ch2_Voltage` (Induced Voltage).
3. Run the main analysis script:

```bash
python main.py
```

## Directory Structure

- `src/`: Core logic (Physics, DSP, Config).
- `data/`: Raw experimental CSV data.
- `output/`: Generated plots (B-H Loops, Permeability) and analysis results.

## Core Features

- **Drift Correction**: Uses linear detrending to prevent B-H loop "spiraling".
- **Advanced Smoothing**: Implements **Savitzky-Golay filtering** (adaptive window) to remove quantization noise while preserving saturation tips.
- **Physics Engine**: 
    - **Core Loss**: Area of B-H loop ($W/kg$).
    - **Amplitude Permeability ($\mu_{amp}$)**: Slope to peak flux ($B_{peak}/H_{peak}$).
    - **Differential Permeability ($\mu_{diff}$)**: Instantaneous time-derivative ($dB/dt / dH/dt$).
- **OOP Design**: Easy for team members to extend (e.g., adding new core types in `config.py`).

---

## Developer Onboarding Guide

Welcome to the team! This codebase is designed to handle the precision-heavy task of converting electrical waveforms into magnetic data. 

### 1. The Core Architecture
We use **Separation of Concerns**. As a developer, you don't need to touch the physics math if you're fixing a UI bug, and you don't need to touch the DSP logic if you're updating core hardware specs.

*   **`src/config.py`**: The "Blueprint." Contains the physical dimensions of the HWR90/32 core (N1=37, N2=20, etc.).
*   **`src/dsp_utils.py`**: The "Cleaning Lab." Handles **Savitzky-Golay smoothing**, integration, and drift correction.
*   **`src/physics.py`**: The "Science." Implementations of Ampere's Law, Faraday's Law, and Permeability calculations.
*   **`src/analyzer.py`**: The "Orchestrator." Loads data, runs the pipeline, and generates plots.

### 2. The Execution Flow
When you run `python3 main.py`, the system follows this path:

1.  **Initialization**: `main.py` loads the `HardwareConstants` from the config.
2.  **Discovery**: Scans `data/` for `.csv` files.
3.  **The Pipeline (`analyze_waveform`)**:
    *   **Step A (Smoothing)**: Apply Savitzky-Golay filter to raw Voltage/Current signals to remove noise without distorting saturation tips.
    *   **Step B (Physics)**: Convert `Ch1` -> $H$ (Ampere's Law) and Center `Ch2` (remove DC).
    *   **Step C (Flux)**: Integrate `Ch2` and apply **Drift Correction** (Linear Detrending) to close the loop.
    *   **Step D (B-Field)**: Scale Flux -> $B$ (Faraday's Law).
    *   **Step E (Analysis)**: Calculate **Core Loss** (Loop Area) and **Permeability** ($\mu_{amp}$, $\mu_{diff}$).
4.  **Finalization**: Results (B-H Loop + Permeability Plot) are saved in `output/`.

### 3. Usage Examples

#### Scenario A: Standard Batch Processing
Drop CSV files in `data/` and run:
```bash
python3 main.py
```

#### Scenario B: Using the Analyzer in your own script
```python
from src.config import HardwareConstants
from src.analyzer import TransformerCoreAnalyzer

# 1. Load the specific core specs
my_core = HardwareConstants.default_hwr90_32()

# 2. Start the analyzer
analyzer = TransformerCoreAnalyzer(my_core)

# 3. Process data manually
# Returns H, B, Loss, Differential Permeability
df = analyzer.load_data("data/my_experiment.csv")
H, B, loss, mu_diff = analyzer.analyze_waveform(df['Time'], df['Ch1'], df['Ch2'], frequency=50)

print(f"The specific core loss is {loss:.4f} W/kg")
```

### 4. Key Developer Tips
*   **Permeability**: We calculate two types. **Amplitude Permeability** ($\mu_{amp}$) is the effective value at the peak (printed to console). **Differential Permeability** ($\mu_{diff}$) is the instantaneous slope (plotted).
*   **Smoothing**: If you see "rounded" saturation tips, the Savitzky-Golay window might be too large (check `src/dsp_utils.py`).
*   **Units**: Always work in SI units ($m^2$, $T$, $A/m$). The code converts input units (like ms) internally.
