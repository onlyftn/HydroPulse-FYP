
from collections import deque

import numpy as np
import pandas as pd


class LiveSensorSimulator:

    SENSOR_COLUMNS = [
        "tgb_vibration",
        "tgb_temperature",
        "ugb_vibration",
        "ugb_temperature",
        "unit_speed",
        "generator_power"
    ]

    MODEL_FEATURES = {
        "TGB": [
            "tgb_vibration",
            "tgb_temperature",
            "unit_speed",
            "generator_power"
        ],
        "UGB": [
            "ugb_vibration",
            "ugb_temperature",
            "unit_speed",
            "generator_power"
        ]
    }

    def __init__(self, data_path, seed=42):

        self.rng = np.random.default_rng(seed)

        data = pd.read_csv(
            data_path,
            parse_dates=["timestamp"]
        )

        # Only use June-August training data
        data = data[
            (data["timestamp"].dt.year == 2020)
            & (data["timestamp"].dt.month.isin([6, 7, 8]))
        ].sort_values("timestamp").reset_index(drop=True)

        if len(data) < 37:
            raise ValueError("Insufficient training data.")

        self.data = data

        values = data[self.SENSOR_COLUMNS].to_numpy(
            dtype=float
        )

        continuous = data["timestamp"].diff().eq(
            pd.Timedelta(minutes=10)
        )

        pair_indices = np.where(
            continuous.to_numpy()
        )[0]

        self.previous_states = values[pair_indices - 1]
        self.next_states = values[pair_indices]

        self.minimums = values.min(axis=0)
        self.maximums = values.max(axis=0)

        self.scales = np.maximum(
            self.previous_states.std(axis=0),
            1e-6
        )

        segment_ids = (~continuous).cumsum()

        valid_segments = [
            group
            for _, group in data.groupby(segment_ids)
            if len(group) >= 36
        ]

        if not valid_segments:
            raise ValueError("No valid 36-step segment.")

        selected = valid_segments[
            self.rng.integers(len(valid_segments))
        ]

        start = self.rng.integers(
            len(selected) - 36 + 1
        )

        initial_window = selected.iloc[
            start:start + 36
        ][self.SENSOR_COLUMNS].to_numpy(dtype=float)

        self.history = deque(
            initial_window,
            maxlen=36
        )

        self.last_timestamp = None

    def advance(self, timestamp):

        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is None:
            raise ValueError(
                "Timezone-aware timestamp required."
            )

        timestamp = timestamp.tz_convert(
            "Asia/Kuala_Lumpur"
        )

        if (
            self.last_timestamp is not None
            and timestamp <= self.last_timestamp
        ):
            raise ValueError(
                "Timestamp must move forward."
            )

        current = np.asarray(
            self.history[-1],
            dtype=float
        )

        distances = np.mean(
            (
                (self.previous_states - current)
                / self.scales
            ) ** 2,
            axis=1
        )

        k = min(50, len(distances))

        nearest = np.argpartition(
            distances,
            k - 1
        )[:k]

        chosen = self.rng.choice(nearest)

        change = (
            self.next_states[chosen]
            - self.previous_states[chosen]
        )

        new_values = np.clip(
            current + change,
            self.minimums,
            self.maximums
        )

        # Preserve discrete vibration values
        new_values[0] = round(new_values[0])
        new_values[2] = round(new_values[2])

        self.history.append(new_values)

        self.last_timestamp = timestamp

        reading = {
            "timestamp": timestamp.isoformat()
        }

        reading.update(
            dict(zip(
                self.SENSOR_COLUMNS,
                new_values.tolist()
            ))
        )

        return reading

    def get_model_window(self, component):

        component = component.upper()

        if component not in self.MODEL_FEATURES:
            raise ValueError("Unknown component.")

        indices = [
            self.SENSOR_COLUMNS.index(column)
            for column in self.MODEL_FEATURES[component]
        ]

        window = np.asarray(
            self.history,
            dtype=float
        )

        return window[:, indices]