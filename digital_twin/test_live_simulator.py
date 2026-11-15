
from pathlib import Path

import pandas as pd

from live_sensor_simulator import LiveSensorSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "prepared"
    / "hydrotwin_cleaned_data.csv"
)

# Initialize simulator
simulator = LiveSensorSimulator(DATA_PATH)

# Use current Malaysia time
scheduled_time = pd.Timestamp.now(
    tz="Asia/Kuala_Lumpur"
).floor("s")

# Generate simulated readings
reading = simulator.advance(scheduled_time)

print("SIMULATED SENSOR READING")
print(reading)

print("\nTGB input shape:")
print(simulator.get_model_window("TGB").shape)

print("\nUGB input shape:")
print(simulator.get_model_window("UGB").shape)
