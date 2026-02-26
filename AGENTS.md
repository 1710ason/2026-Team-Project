# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Batch entry point. Scans `data/*.csv`, infers frequency from filename, runs analysis, and writes outputs.
- `src/`: Core package.
- `src/config.py`: Hardware constants and default core configuration.
- `src/dsp_utils.py`: Signal conditioning (smoothing, DC removal, integration, drift correction).
- `src/physics.py`: Magnetic field, flux density, loss, and permeability calculations.
- `src/analyzer.py`: Orchestrates loading, analysis pipeline, and plotting.
- `data/`: Input CSV waveforms (kept out of source logic).
- `output/`: Generated plots/results (artifacts, not hand-edited).
- `Docs/`: Project notes and supplementary documentation.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create and activate local environment.
- `pip install -r requirements.txt`: Install pinned runtime dependencies.
- `python main.py`: Run batch analysis on all CSV files in `data/`.
- `python -m src.analyzer` is not the intended entrypoint; use `main.py` for full workflow.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear, typed-friendly function signatures where practical.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and descriptive constant names in `config.py`.
- Keep analysis steps modular: DSP operations stay in `dsp_utils.py`, physics formulas in `physics.py`, orchestration in `analyzer.py`.
- Prefer small, deterministic functions over inline logic in `main.py`.

## Testing Guidelines
- No formal test suite exists yet; new features should add `pytest` tests under `tests/`.
- Name test files `test_<module>.py` and test functions `test_<behavior>()`.
- For numerical logic, include tolerance-based assertions (for example with `numpy.isclose`).
- Before opening a PR, run at least: `python main.py` on a representative CSV and verify `output/*_analysis.png` is produced.

## Commit & Pull Request Guidelines
- Match repository history: use concise, imperative subjects with optional prefixes such as `fix:`, `feat:`, `refactor:`, or `build:`.
- Keep commits focused; separate physics changes, parsing changes, and plotting/UI changes when possible.
- PRs should include:
- A short problem/solution summary.
- Linked issue (if available).
- Validation notes (sample dataset used, expected metric/plot impact).
- Screenshot(s) of updated output plots when visualization changes.
