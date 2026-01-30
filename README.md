# Magnetic Core Analysis Project

This repository contains the software pipeline for characterizing transformer core losses (50Hz - 400Hz).

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

1. Place your oscilloscope CSV data in the `data/` directory.
2. Ensure CSV files have headers: `Time`, `Ch1_Voltage`, `Ch2_Voltage`.
3. Run the main analysis script:

```bash
python main.py
```

## Directory Structure

- `src/`: Core logic (Physics, DSP, Config).
- `data/`: Raw experimental CSV data.
- `output/`: Generated plots and analysis results.

## Core Features

- **Drift Correction**: Uses linear detrending to prevent B-H loop "spiraling".
- **OOP Design**: Easy for team members to extend (e.g., adding new core types in `config.py`).
- **Standardized Physics**: Centralized formulas for B, H, and Core Loss calculations.

---

## Developer Onboarding Guide

Welcome to the team! This codebase is designed to handle the precision-heavy task of converting electrical waveforms into magnetic data. 

### 1. The Core Architecture
We use **Separation of Concerns**. As a developer, you don't need to touch the physics math if you're fixing a UI bug, and you don't need to touch the DSP logic if you're updating core hardware specs.

*   **`src/config.py`**: The "Blueprint." Contains the physical dimensions of the HWR90/32 core.
*   **`src/dsp_utils.py`**: The "Cleaning Lab." This is where we handle the noise and the integration "drift" mentioned in the project brief.
*   **`src/physics.py`**: The "Science." Contains pure mathematical implementations of Ampere's and Faraday's laws.
*   **`src/analyzer.py`**: The "Orchestrator." It knows the *order* in which to call the cleaning and science functions.

### 2. The Execution Flow
When you run `python3 main.py`, the system follows this linear path:

1.  **Initialization**: `main.py` loads the `HardwareConstants` from the config.
2.  **Discovery**: It scans the `data/` folder for any `.csv` files.
3.  **The Pipeline (`analyze_waveform`)**:
    *   **Step A (H-Field)**: Convert `Ch1_Voltage` (across a $1\Omega$ resistor) directly into Magnetizing Force ($H$) using Ampere’s Law.
    *   **Step B (Cleaning)**: The secondary voltage ($Ch2$) is centered (DC offset removed) to prepare for integration.
    *   **Step C (Integration)**: We integrate the voltage to get the magnetic flux. 
    *   **Step D (Drift Correction)**: **This is critical.** Integration naturally "drifts" upwards or downwards. We use `scipy.signal.detrend` to flatten it so the B-H loop stays periodic.
    *   **Step E (B-Field)**: The cleaned flux is scaled by the core's area ($A_e$) and turns ($N_2$) to get Magnetic Flux Density ($B$ in Teslas).
    *   **Step F (Loss Calculation)**: We calculate the area of the resulting B-H loop to find the power loss ($W/kg$).
4.  **Finalization**: The results are saved as plots in the `output/` folder.

### 3. Usage Examples

#### Scenario A: Standard Batch Processing
If you just have new data files, drop them in `data/` and run the automation:
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
df = analyzer.load_data("data/my_experiment.csv")
H, B, loss = analyzer.analyze_waveform(df['Time'], df['Ch1'], df['Ch2'], frequency=50)

print(f"The core loss for this sample is {loss} W/kg")
```

#### Scenario C: Adding a new core type
```python
# Inside src/config.py, add a new factory method:
@classmethod
def new_large_core(cls):
    return cls(N1=500, N2=500, Ae=1.2e-3, Lm=0.45, Rsense=1.0, Density=7650.0)
```

### 4. Key Developer Tips
*   **Drift Correction**: If you see the B-H loops looking like spirals instead of ovals, the issue is in `dsp_utils.py`. The detrending logic is what keeps the "loops" closed.
*   **Column Names**: The `load_data` function in `analyzer.py` currently assumes your CSV has headers named `Time`, `Ch1_Voltage`, and `Ch2_Voltage`.
*   **Units**: Always work in SI units ($m^2$, $T$, $A/m$). The code converts these internally.
