"""Read-only access to the separately scheduled HydroTwin simulation database."""

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "digital_twin" / "live_state.db"


_QUERY_COLUMNS = """
    SELECT scheduled_at, started_at, completed_at, forecast_at, results_json
    FROM prediction_cycles
"""


def _decode(row):
    if row is None:
        return None
    return {
        "scheduled_at": row[0],
        "started_at": row[1],
        "completed_at": row[2],
        "forecast_at": row[3],
        "results": json.loads(row[4]),
    }


def _query(sql, parameters=()):
    if not DB_PATH.is_file():
        return []
    # Read-only: the Streamlit UI never modifies the simulation/scheduler state.
    db_uri = DB_PATH.resolve().as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(db_uri, uri=True, timeout=5) as conn:
            return conn.execute(sql, parameters).fetchall()
    except sqlite3.OperationalError as exc:
        # New installation, with an empty database not yet initialized.
        if "no such table" in str(exc).lower():
            return []
        raise


def get_latest_cycle():
    rows = _query(_QUERY_COLUMNS + " ORDER BY scheduled_at DESC LIMIT 1")
    return _decode(rows[0]) if rows else None


def get_recent_cycles(limit=72):
    """Return at most `limit` completed cycles, oldest first."""
    limit = max(1, min(int(limit), 500))
    rows = _query(
        _QUERY_COLUMNS + " ORDER BY scheduled_at DESC LIMIT ?", (limit,)
    )
    return [_decode(row) for row in reversed(rows)]

def get_scheduler_health():

    if not DB_PATH.exists():
        return None

    try:
        with sqlite3.connect(DB_PATH, timeout=5) as conn:

            row = conn.execute("""
                SELECT heartbeat_at, next_run_at
                FROM scheduler_health
                WHERE id = 1
            """).fetchone()

    except sqlite3.OperationalError:
        return None

    if row is None:
        return None

    return {
        "heartbeat_at": row[0],
        "next_run_at": row[1]
    }
