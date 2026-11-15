
import json
import sqlite3

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = (
    PROJECT_ROOT / "digital_twin" / "live_state.db"
)

MODELS = ["Linear Regression", "LSTM", "GRU"]

COMPONENTS = ["TGB", "UGB"]


def get_verified_predictions(
    limit=200,
    tolerance_seconds=60
):
    """
    Compare each 10-minute forecast with the
    next simulated sensor observation.

    Reject pairs separated by missing cycles
    or excessive scheduler timing differences.
    """

    columns = [
        "Component",
        "Model",
        "Forecast Time",
        "Observation Time",
        "Predicted",
        "Actual",
        "Absolute Error",
        "Signed Error"
    ]

    if not DB_PATH.exists():
        return pd.DataFrame(columns=columns)

    with sqlite3.connect(DB_PATH, timeout=10) as conn:

        rows = conn.execute(
            """
            SELECT
                scheduled_at,
                forecast_at,
                results_json
            FROM prediction_cycles
            ORDER BY scheduled_at DESC
            LIMIT ?
            """,
            (limit + 1,)
        ).fetchall()

    rows.reverse()

    verified = []

    # Compare each prediction with the following cycle
    for index in range(len(rows) - 1):

        previous = rows[index]
        following = rows[index + 1]

        forecast_time = datetime.fromisoformat(
            previous[1]
        )

        observation_time = datetime.fromisoformat(
            following[0]
        )

        timing_error = abs(
            (
                observation_time - forecast_time
            ).total_seconds()
        )

        # Avoid matching predictions across
        # missing cycles or scheduler interruptions.
        if timing_error > tolerance_seconds:
            continue

        predicted_results = json.loads(previous[2])

        actual_results = json.loads(following[2])

        for component in COMPONENTS:

            predictions = (
                predicted_results["components"][component]
                ["all_predictions"]
            )

            actual = float(
                actual_results["components"][component]
                ["vibration"]
            )

            for model in MODELS:

                predicted = float(predictions[model])

                if not (
                    np.isfinite(actual)
                    and np.isfinite(predicted)
                ):
                    continue

                verified.append({
                    "Component": component,
                    "Model": model,
                    "Forecast Time": forecast_time,
                    "Observation Time": observation_time,
                    "Predicted": predicted,
                    "Actual": actual,
                    "Absolute Error": abs(
                        actual - predicted
                    ),
                    "Signed Error": predicted - actual
                })

    return pd.DataFrame(
        verified,
        columns=columns
    )


def summarize_verified_predictions(verified):

    if verified.empty:
        return pd.DataFrame()

    summary = (
        verified
        .groupby(["Component", "Model"])
        .agg(
            Samples=("Absolute Error", "count"),
            MAE=("Absolute Error", "mean"),
            RMSE=(
                "Signed Error",
                lambda x: np.sqrt(
                    np.mean(np.square(x))
                )
            ),
            Mean_Error=("Signed Error", "mean")
        )
        .reset_index()
    )

    return summary.sort_values(
        ["Component", "RMSE"]
    ).reset_index(drop=True)