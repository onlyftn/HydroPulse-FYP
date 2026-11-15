# Cleanup record

This folder is a cleaned copy of the uploaded HydroPulse ZIP. The original archive was not changed.

**Removed from this copy (preserved in the uploaded original):**

- `models/xgboost/` (six JSON files): experimental models, not part of the reported Linear Regression/LSTM/GRU pipeline.
- `results/xgboost_*.csv` (three CSV files): results from the excluded XGBoost experiments.
- `data/raw/vibration data.csv` and `data/raw/vibration data of G4 15-05-2022 to 10-05-2022.csv`: not among the six G1 monthly files used in the documented modelling pipeline.
- `dashboard/maintenance_db.py`: no imports from the final three-page dashboard; the maintenance pages were already excluded before upload.
- `digital_twin/live_state.db`: local generated SQLite state, exported to `data/synthetic/simulation_history.csv` and recreated when the scheduler runs.
- Stored execution outputs in the seven Jupyter notebooks: source, markdown and cell ordering preserved; outputs removed to avoid large embedded figures and private execution traces.

**Retained on purpose:** all six models, all four scalers, six original G1 CSVs, prepared inputs, test results, simulation scripts, GLB and STEP source 3D assets, and other training notebooks. The 3D STEP model is kept as a source asset even though only the GLB is read by the active dashboard.

**Known review items:**

- The historical digital-twin notebook (`notebooks/07_digital_twin.ipynb`) contains some older `hydrotwin_*` module names; these were deliberately not rewritten because this cleanup aims to preserve the submitted project logic.
- The original historical CSV permissions have not been verified; the repository should be private before sharing with the supervisor.
- Exact dependency versions for the original training environment were not included; `requirements.txt` provides inferred compatible-version ranges and needs testing on the user's machine.
