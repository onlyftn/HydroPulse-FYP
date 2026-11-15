# Synthetic simulation data

`simulation_history.csv` contains **343 saved simulation cycles** exported from the original local `digital_twin/live_state.db` in the uploaded project. The export contains the synthetic sensor measurements and predictions that were actually saved by the user's prototype. No new observations were invented.

- The timestamps refer to prototype runs in 2026, **not** to operational plant measurements.
- These readings are synthetic and must **not** be used as the November 2020 independent test dataset.
- The original live SQLite database was intentionally excluded because it contains machine-specific scheduler state. Running `digital_twin/live_scheduler.py` initializes a new database locally.
- `digital_twin/g1_integrated_replay.csv` is a **historical-data-derived replay/prediction table**, not synthetic readings collected by a real sensor.
- No primary sensor dataset was collected by this project.
