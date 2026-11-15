import argparse
import json
import sqlite3

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from apscheduler.schedulers.blocking import BlockingScheduler

from live_sensor_simulator import LiveSensorSimulator
from live_predictor import LivePredictor

from datetime import datetime
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "prepared"
    / "hydrotwin_cleaned_data.csv"
)

DB_PATH = (
    PROJECT_ROOT / "digital_twin" / "live_state.db"
)

TIMEZONE = "Asia/Kuala_Lumpur"


def get_connection():
    return sqlite3.connect(DB_PATH, timeout=30)


def initialize_database():
    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_cycles (
                scheduled_at TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                forecast_at TEXT NOT NULL,
                results_json TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS simulator_state (
                id INTEGER PRIMARY KEY,
                history_json TEXT NOT NULL,
                rng_json TEXT NOT NULL,
                last_timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_health (
                id INTEGER PRIMARY KEY,
                heartbeat_at TEXT NOT NULL,
                next_run_at TEXT
            )
        """)


def restore_simulator(simulator):
    with get_connection() as conn:

        saved = conn.execute("""
            SELECT history_json, rng_json, last_timestamp
            FROM simulator_state
            WHERE id = 1
        """).fetchone()

    if saved is None:
        print("Starting a new synthetic simulation.")
        return

    history, rng_state, last_timestamp = saved

    simulator.history = deque(
        [
            np.asarray(row, dtype=float)
            for row in json.loads(history)
        ],
        maxlen=36
    )

    simulator.rng.bit_generator.state = json.loads(
        rng_state
    )

    simulator.last_timestamp = (
        pd.Timestamp(last_timestamp)
        if last_timestamp else None
    )

    print("Previous simulation state restored.")


def save_result(result, started_at, completed_at, simulator):

    history_json = json.dumps(
        [row.tolist() for row in simulator.history]
    )

    rng_json = json.dumps(
        simulator.rng.bit_generator.state
    )

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO prediction_cycles (
                scheduled_at,
                started_at,
                completed_at,
                forecast_at,
                results_json
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            result["timestamp"],
            started_at.isoformat(),
            completed_at.isoformat(),
            result["forecast_timestamp"],
            json.dumps(result, allow_nan=False)
        ))

        conn.execute("""
            INSERT OR REPLACE INTO simulator_state (
                id, history_json, rng_json, last_timestamp
            )
            VALUES (1, ?, ?, ?)
        """, (
            history_json,
            rng_json,
            result["timestamp"]
        ))


def run_scheduled_cycle(simulator, predictor):

    started = pd.Timestamp.now(tz=TIMEZONE)

    scheduled_time = started.floor("s")

    with get_connection() as conn:

        existing = conn.execute("""
            SELECT 1 FROM prediction_cycles
            WHERE scheduled_at = ?
        """, (scheduled_time.isoformat(),)).fetchone()

    if existing:
        print("Cycle already saved:", scheduled_time)
        return

    result = predictor.predict_cycle(
        simulator,
        scheduled_time,
        selected_model="GRU"
    )

    completed = pd.Timestamp.now(tz=TIMEZONE)

    save_result(result, started, completed, simulator)

    print("\nPrediction cycle completed", flush=True)
    print("Scheduled:", result["timestamp"], flush=True)
    print("Started:", started.isoformat(), flush=True)
    print("Completed:", completed.isoformat(), flush=True)
    print("Forecast target:", result["forecast_timestamp"], flush=True)
    print("Both TGB and UGB updated.", flush=True)

def save_scheduler_heartbeat(scheduler):

    current_time = pd.Timestamp.now(
        tz=TIMEZONE
    )

    prediction_job = scheduler.get_job(
        "hydrotwin_live_cycle"
    )

    next_run = (
        prediction_job.next_run_time.isoformat()
        if prediction_job is not None
        else None
    )

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO scheduler_health (
                id,
                heartbeat_at,
                next_run_at
            )
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                heartbeat_at = excluded.heartbeat_at,
                next_run_at = excluded.next_run_at
        """, (
            current_time.isoformat(),
            next_run
        ))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run one test without saving or scheduling"
    )
    args = parser.parse_args()

    initialize_database()

    simulator = LiveSensorSimulator(DATA_PATH)
    restore_simulator(simulator)

    predictor = LivePredictor(PROJECT_ROOT)

    if args.test:

        test_time = pd.Timestamp.now(
            tz=TIMEZONE
        ).floor("s")

        result = predictor.predict_cycle(
            simulator,
            test_time,
            selected_model="GRU"
        )

        print("\nMANUAL SCHEDULER TEST")
        print("Simulation time:", result["timestamp"])
        print("Forecast target:", result["forecast_timestamp"])

        for component in ["TGB", "UGB"]:
            print(
                component,
                result["components"][component][
                    "predicted_vibration"
                ]
            )

        print("Test complete. No database cycle saved.")
        return

    scheduler = BlockingScheduler(
        timezone=TIMEZONE
    )

    from datetime import datetime
    from zoneinfo import ZoneInfo

    scheduler.add_job(
    lambda: run_scheduled_cycle(
        simulator,
        predictor
    ),
    trigger="interval",
    minutes=10,
    next_run_time=datetime.now(ZoneInfo(TIMEZONE)),
    id="hydrotwin_live_cycle",
    max_instances=1,
    coalesce=True,
    misfire_grace_time=40
    )

    scheduler.add_job(
        lambda: save_scheduler_heartbeat(scheduler),
        trigger="interval",
        seconds=15,
        next_run_time=datetime.now(ZoneInfo(TIMEZONE)),
        id="hydrotwin_heartbeat",
        max_instances=1,
        coalesce=True
    )
    
    print("\nHydroTwin scheduler running.")
    print("Timezone:", TIMEZONE)
    print("Schedule: :00, :10, :20, :30, :40, :50")
    print("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")


if __name__ == "__main__":
    main()
