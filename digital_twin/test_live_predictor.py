
from pathlib import Path

import pandas as pd

from live_sensor_simulator import LiveSensorSimulator
from live_predictor import LivePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "prepared"
    / "hydrotwin_cleaned_data.csv"
)

# Initialize simulator and trained models
simulator = LiveSensorSimulator(DATA_PATH)
predictor = LivePredictor(PROJECT_ROOT)

# Current Malaysia time aligned to 10-minute interval
scheduled_time = pd.Timestamp.now(
    tz="Asia/Kuala_Lumpur"
).floor("s")

# Use GRU for both TGB and UGB
result = predictor.predict_cycle(
    simulator,
    scheduled_time,
    selected_model="GRU"
)

print("\nHYDROTWIN LIVE PREDICTION TEST")
print("Simulation time:", result["timestamp"])
print("Forecast target:", result["forecast_timestamp"])
print("Global model:", result["selected_model"])

for component in ["TGB", "UGB"]:

    state = result["components"][component]

    print(f"\n{component}")
    print("Simulated vibration:", state["vibration"])
    print("Temperature:", state["temperature"])
    print("Selected model:", state["selected_model"])
    print("Next 10-minute prediction:", state["predicted_vibration"])
    print("All model predictions:", state["all_predictions"])
