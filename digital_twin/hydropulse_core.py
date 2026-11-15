
import pandas as pd


class HydroTwinGeneratingUnit:

    MODEL_COLUMNS = {
        "TGB": {
            "Linear Regression": "tgb_lr_prediction",
            "LSTM": "tgb_lstm_prediction",
            "GRU": "tgb_gru_prediction"
        },

        "UGB": {
            "Linear Regression": "ugb_lr_prediction",
            "LSTM": "ugb_lstm_prediction",
            "GRU": "ugb_gru_prediction"
        }
    }

    def __init__(self, data):

        self.data = (
            data.copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        self.data["timestamp"] = pd.to_datetime(
            self.data["timestamp"]
        )

        if self.data.empty:
            raise ValueError("Digital twin dataset is empty.")

        if self.data["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps detected.")

        self.position = 0

        # Models selected using September validation
        self.selected_models = {
            "TGB": "GRU",
            "UGB": "LSTM"
        }

    def get_bearing_state(self, component, row, next_row):

        prefix = component.lower()

        # Next-timestamp model predictions
        if next_row is not None:

            predictions = {
                model: float(next_row[column])
                for model, column in
                self.MODEL_COLUMNS[component].items()
            }

        else:

            predictions = {
                model: None
                for model in self.MODEL_COLUMNS[component]
            }

        selected_model = self.selected_models[component]

        return {
            "component": component,

            "vibration": float(
                row[f"{prefix}_vibration"]
            ),

            "temperature": float(
                row[f"{prefix}_temperature"]
            ),

            "selected_model": selected_model,

            "next_prediction": predictions[selected_model],

            "all_predictions": predictions
        }

    def get_state(self):

        row = self.data.iloc[self.position]

        current_time = row["timestamp"]

        next_row = None
        forecast_time = None

        if self.position + 1 < len(self.data):

            candidate = self.data.iloc[self.position + 1]

            expected_time = (
                current_time + pd.Timedelta(minutes=10)
            )

            if candidate["timestamp"] == expected_time:

                next_row = candidate
                forecast_time = expected_time

        return {
            "unit_id": "G1",

            "timestamp": current_time,

            "forecast_timestamp": forecast_time,

            "replay_position": self.position,

            "total_records": len(self.data),

            "generator_power": float(
                row["generator_power"]
            ),

            "unit_speed": float(
                row["unit_speed"]
            ),

            "TGB": self.get_bearing_state(
                "TGB", row, next_row
            ),

            "UGB": self.get_bearing_state(
                "UGB", row, next_row
            )
        }

    def next_step(self):

        if self.position + 1 >= len(self.data):
            return None

        self.position += 1

        return self.get_state()

    def reset(self):

        self.position = 0

        return self.get_state()

    def select_model(self, component, model):

        component = component.upper()

        if component not in self.MODEL_COLUMNS:
            raise ValueError("Unknown component.")

        if model not in self.MODEL_COLUMNS[component]:
            raise ValueError("Unknown model.")

        self.selected_models[component] = model
