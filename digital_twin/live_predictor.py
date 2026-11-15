
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


class LivePredictor:

    MODEL_NAMES = ["Linear Regression", "LSTM", "GRU"]

    def __init__(self, project_root):

        root = Path(project_root)

        self.models = {}
        self.scalers = {}

        for component in ["TGB", "UGB"]:

            name = component.lower()

            # Load previously trained models
            self.models[component] = {
                "Linear Regression": joblib.load(
                    root / "models" / "linear_regression"
                    / f"{name}_linear_regression.pkl"
                ),
                "LSTM": load_model(
                    root / "models" / "lstm"
                    / f"{name}_lstm.keras",
                    compile=False
                ),
                "GRU": load_model(
                    root / "models" / "gru"
                    / f"{name}_gru.keras",
                    compile=False
                )
            }

            # Reuse training-only scalers
            scaler_dir = root / "models" / "lstm" / "scalers"

            self.scalers[component] = {
                "feature": joblib.load(
                    scaler_dir / f"{name}_feature_scaler.pkl"
                ),
                "target": joblib.load(
                    scaler_dir / f"{name}_target_scaler.pkl"
                )
            }

        print("All six trained models loaded successfully.")

    def predict_component(self, component, window):

        component = component.upper()
        window = np.asarray(window, dtype=float)

        if component not in self.models:
            raise ValueError("Unknown component.")

        if window.shape != (36, 4):
            raise ValueError("Expected input shape (36, 4).")

        if not np.isfinite(window).all():
            raise ValueError("Invalid sensor readings detected.")

        # Linear Regression: raw flattened inputs
        lr_pred = self.models[component][
            "Linear Regression"
        ].predict(window.reshape(1, -1))

        predictions = {
            "Linear Regression": float(lr_pred[0])
        }

        # LSTM and GRU: scaled sequential inputs
        feature_scaler = self.scalers[component]["feature"]
        target_scaler = self.scalers[component]["target"]

        scaled_window = feature_scaler.transform(
            window
        ).reshape(1, 36, 4)

        for model_name in ["LSTM", "GRU"]:

            scaled_pred = self.models[component][
                model_name
            ].predict(scaled_window, verbose=0)

            original_pred = target_scaler.inverse_transform(
                scaled_pred.reshape(-1, 1)
            )

            predictions[model_name] = float(
                original_pred[0, 0]
            )

        return predictions

    def predict_cycle(
        self,
        simulator,
        scheduled_time,
        selected_model="GRU"
    ):

        if selected_model not in self.MODEL_NAMES:
            raise ValueError("Invalid forecasting model.")

        # One synchronized sensor update for both bearings
        reading = simulator.advance(scheduled_time)

        current_time = pd.Timestamp(reading["timestamp"])
        forecast_time = current_time + pd.Timedelta(minutes=10)

        result = {
            "timestamp": current_time.isoformat(),
            "forecast_timestamp": forecast_time.isoformat(),
            "selected_model": selected_model,
            "simulation_type": "Synthetic Sensor Simulation",
            "unit_speed": reading["unit_speed"],
            "generator_power": reading["generator_power"],
            "components": {}
        }

        for component in ["TGB", "UGB"]:

            window = simulator.get_model_window(component)

            predictions = self.predict_component(
                component, window
            )

            prefix = component.lower()

            result["components"][component] = {
                "vibration": reading[f"{prefix}_vibration"],
                "temperature": reading[f"{prefix}_temperature"],
                "selected_model": selected_model,
                "predicted_vibration": predictions[selected_model],
                "all_predictions": predictions
            }

        return result
