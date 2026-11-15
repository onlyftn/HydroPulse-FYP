from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "prepared"
    / "hydrotwin_cleaned_data.csv"
)


def calculate_operating_ranges():

    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    # Training data only: June to August 2020
    train = data[
        (data["timestamp"].dt.year == 2020)
        & (data["timestamp"].dt.month.isin([6, 7, 8]))
    ]

    if train.empty:
        raise ValueError("Training data not found.")

    ranges = {}

    for component in ["TGB", "UGB"]:

        column = f"{component.lower()}_vibration"

        ranges[component] = {
            "lower": float(train[column].quantile(0.05)),
            "upper": float(train[column].quantile(0.95))
        }

    return ranges