"""HydroPulse G1 — live synthetic simulation monitoring dashboard.

The models execute in a separate scheduler. The dashboard ONLY reads the saved
cycles and changes which stored model forecast it displays. This is not SCADA.
"""

from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "digital_twin"))
from hydropulse_cad_viewer import cad_preview

import plotly.graph_objects as go
import streamlit as st
import numpy as np

from live_data import get_latest_cycle, get_recent_cycles, get_scheduler_health
from prediction_verification import (
    get_verified_predictions,
    summarize_verified_predictions
)
from operating_ranges import calculate_operating_ranges
from shared_sidebar import render_shared_sidebar
from forecast_chart_upgrade import make_hydropulse_forecast_chart

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIGITAL_TWIN_DIR = PROJECT_ROOT / "digital_twin"
GLB_PATH = DIGITAL_TWIN_DIR / "assets" / "HydroTwin_G1_Interactive.glb"
CSS_PATH = Path(__file__).resolve().parent / "styles.css"
TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")
MODEL_NAMES = ["Linear Regression", "LSTM", "GRU"]
REFRESH_INTERVAL = "5s"  # UI refresh only; inference runs in the scheduler.

st.set_page_config(
    page_title="HydroPulse",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_operating_ranges():
    return calculate_operating_ranges()


OPERATING_RANGES = load_operating_ranges()

if CSS_PATH.is_file():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _local_datetime(iso_time):
    if not iso_time:
        return None
    timestamp = datetime.fromisoformat(str(iso_time))
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=TIMEZONE)
    return timestamp.astimezone(TIMEZONE)


def _display_time(iso_time):
    value = _local_datetime(iso_time)
    return value.strftime("%d %b %Y · %H:%M:%S MYT") if value else "—"


def _fmt_number(value, decimals=3):
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _last_cycle_status(cycle):
    """Recent saved data does NOT prove a scheduler process is still running."""
    if not cycle:
        return ("AWAITING SIMULATION", False)
    completed = _local_datetime(cycle.get("completed_at"))
    if completed is None:
        return ("CYCLE TIME UNAVAILABLE", False)
    age = (datetime.now(TIMEZONE) - completed).total_seconds()
    if age < -30:
        return ("CLOCK MISMATCH", False)
    if age > 12 * 60:
        return ("STALE SIMULATION DATA", False)
    return ("RECENT SIMULATION CYCLE", True)


def _scheduler_snapshot():
    """One source of truth for UI state, based on SQLite heartbeat and last cycle.

    A recent heartbeat indicates that the scheduler process is reporting activity,
    not that a completed ML prediction was delivered on time.
    """
    now = datetime.now(TIMEZONE)
    health = get_scheduler_health()
    cycle = get_latest_cycle()

    heartbeat = _local_datetime(health.get("heartbeat_at")) if health else None
    next_run = _local_datetime(health.get("next_run_at")) if health else None
    completed = _local_datetime(cycle.get("completed_at")) if cycle else None

    if heartbeat is None:
        active = False
        health_label = "SCHEDULER NOT CONNECTED"
    else:
        heartbeat_age = (now - heartbeat).total_seconds()
        active = -30 <= heartbeat_age <= 45
        health_label = (
            "SCHEDULER ACTIVE" if active
            else ("SYSTEM CLOCK MISMATCH" if heartbeat_age < -30
                  else "SCHEDULER OFFLINE / DELAYED")
        )

    cycle_age = (now - completed).total_seconds() if completed else None
    cycle_fresh = cycle_age is not None and -30 <= cycle_age <= 12 * 60

    if active and cycle_fresh:
        header_text = "SCHEDULER ACTIVE · FORECAST CURRENT"
        header_tone = "ok"
    elif active:
        header_text = "SCHEDULER ACTIVE · AWAITING NEW FORECAST"
        header_tone = "wait"
    elif cycle is not None:
        header_text = "SCHEDULER OFFLINE · SAVED DATA ONLY"
        header_tone = "offline"
    else:
        header_text = "AWAITING LIVE SIMULATION"
        header_tone = "offline"

    return {
        "now": now, "heartbeat": heartbeat, "next_run": next_run,
        "last_cycle": cycle, "cycle_fresh": cycle_fresh,
        "active": active, "health_label": health_label,
        "header_text": header_text, "header_tone": header_tone,
    }


def metric_card(title, value, note):
    st.markdown(
        f'<div class="ht-card"><div class="ht-card-title">{escape(str(title))}</div>'
        f'<div class="ht-card-value">{escape(str(value))}</div>'
        f'<div class="ht-card-note">{escape(str(note))}</div></div>',
        unsafe_allow_html=True,
    )


def bearing_card(name, selected_model, cycle):
    component = (
        cycle.get("results", {}).get("components", {}).get(name, {})
        if cycle else {}
    )
    # Every scheduler cycle stores all 3 predictions for each component.
    # Switching this global selector does not rerun or retrain models.
    predicted = component.get("all_predictions", {}).get(selected_model)
    name_long = (
        "Turbine Guide Bearing" if name == "TGB" else "Upper Guide Bearing"
    )
    if predicted is None:
        range_status = "Awaiting forecast"
    else:
        lower = OPERATING_RANGES[name]["lower"]
        upper = OPERATING_RANGES[name]["upper"]
        range_status = (
            "Within Typical Range"
            if lower <= float(predicted) <= upper
            else "Outside Typical Range"
        )
    rows = [
        ("Algorithm", selected_model),
        ("Simulated vibration", _fmt_number(component.get("vibration"))),
        ("Temperature", _fmt_number(component.get("temperature"))),
        ("Next +10 min prediction", _fmt_number(predicted, 4)),
        ("Forecast range status", range_status),
    ]
    contents = "".join(
        '<div class="ht-row">'
        f'<span class="ht-key">{escape(label)}</span>'
        f'<span class="ht-val">{escape(value)}</span>'
        '</div>'
        for label, value in rows
    )
    st.markdown(
        f'<div class="ht-bearing">'
        f'<div class="ht-bearing-head {name.lower()}">'
        f'{name} · {name_long}</div>'
        f'{contents}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_cad_figure(component):
    if not GLB_PATH.is_file():
        return None
    if str(DIGITAL_TWIN_DIR) not in sys.path:
        sys.path.insert(0, str(DIGITAL_TWIN_DIR))
    

    figure = cad_preview(
        GLB_PATH,
        selected_component=component,
        housing_opacity=0.24,
    )
    # Preserve the user's chosen title and its subtle opacity.
    figure.update_layout(
        
        title=dict(
            text=(
                "Vertical-Shaft Turbine–Generator"
                "<br>Assembly (Conceptual CAD)"
            ),
            font=dict(
                size=15,
                color="#dcebf6"
            ),
            x=0.02,
            xanchor="left"
        )

    )
    figure.update_layout(
        paper_bgcolor="#0b1725",
        plot_bgcolor="#0b1725",
        font=dict(color="#dcebf6"),
        margin=dict(l=0, r=0, t=80, b=0),
        height=640,
    )
    return figure


@st.fragment(run_every=REFRESH_INTERVAL)
def sidebar_live_status():
    status = _scheduler_snapshot()
    st.caption("SIMULATION STATUS")
    tone = "#39D9A0" if status["active"] else "#F0B44A"
    background = "rgba(37, 163, 122, 0.11)" if status["active"] else "rgba(245,180,93,0.10)"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:9px;'
        f'padding:10px 12px;margin:2px 0 10px;'
        f'background:{background};border:1px solid {tone}55;'
        f'border-radius:10px;color:#F2F8FD;font-size:14px;font-weight:600">'
        f'<span style="height:8px;width:8px;border-radius:50%;'
        f'background:{tone};flex:none"></span>'
        f'{escape("Scheduler Active" if status["active"] else "Scheduler Offline / Delayed")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    cycle = status["last_cycle"]
    if cycle:
        st.caption("Latest saved prediction: " + _display_time(cycle["completed_at"]))
        if not status["cycle_fresh"]:
            st.caption("Awaiting a new completed prediction; displayed sensor data may be old.")
    else:
        st.caption("Waiting for the first saved prediction.")
    st.caption("Synthetic simulation · not connected to plant SCADA")



@st.fragment(run_every=REFRESH_INTERVAL)
def render_live_header():
    status = _scheduler_snapshot()
    tones = {
        "ok": ("#82E4C5", "rgba(31, 137, 105, 0.17)", "#2F705E"),
        "wait": ("#FFD18C", "rgba(166, 110, 33, 0.16)", "#806142"),
        "offline": ("#FFB7A3", "rgba(176, 61, 55, 0.16)", "#86534C"),
    }
    fg, bg, border = tones[status["header_tone"]]
    st.markdown(
        '<div class="ht-header">'
        '<div class="ht-eyebrow">Integrated Hydropower Monitoring</div>'
        '<div class="ht-brand">Hydro<span>Twin</span></div>'
        f'<div class="ht-tag" style="color:{fg};background:{bg};border:1px solid {border}">'
        f'{escape(status["header_text"])}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

@st.fragment(run_every="5s")
def render_live_kpis(selected_model):

    # Read latest completed simulation cycle
    cycle = get_latest_cycle()

    results = cycle["results"] if cycle else {}

    # Format sensor values
    def format_value(value):
        if value is None:
            return "—"
        return f"{float(value):.3f}"

    unit_speed = format_value(
        results.get("unit_speed")
    )

    generator_power = format_value(
        results.get("generator_power")
    )

    # Reuse HydroPulse's existing card styling
    def display_kpi(title, value, note):

        st.markdown(
            f"""
            <div class="ht-card">
                <div class="ht-card-title">
                    {title}
                </div>

                <div class="ht-card-value">
                    {value}
                </div>

                <div class="ht-card-note">
                    {note}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Four generating-unit summary cards
    col1, col2, col3, col4 = st.columns(
        4,
        gap="medium"
    )

    with col1:
        display_kpi(
            "Unit Speed",
            unit_speed,
            "Latest saved simulated reading"
        )

    with col2:
        display_kpi(
            "Generator Power",
            generator_power,
            "Latest saved simulated reading"
        )

    with col3:
        display_kpi(
            "Forecast Interval",
            "10 min",
            "TGB + UGB synchronized"
        )

    with col4:
        display_kpi(
            "Active Model",
            selected_model,
            "Applied to both bearings"
        )

@st.fragment(run_every=REFRESH_INTERVAL)
def render_live_kpis(selected_model):
    cycle = get_latest_cycle()
    result = cycle["results"] if cycle else {}
    columns = st.columns(4, gap="medium")
    with columns[0]:
        metric_card("Unit speed", _fmt_number(result.get("unit_speed")), "Simulated operating reading")
    with columns[1]:
        metric_card("Generator power", _fmt_number(result.get("generator_power")), "Simulated operating reading")
    with columns[2]:
        metric_card("Prediction cadence", "10 min", "Interval from scheduler startup")
    with columns[3]:
        metric_card("Displayed model", selected_model, "Same algorithm for TGB + UGB")
    if cycle:
        st.caption(
            "Last completed cycle: " + _display_time(cycle["completed_at"])
            + "  |  Forecast target: " + _display_time(cycle["forecast_at"])
        )


@st.fragment(run_every=REFRESH_INTERVAL)
def render_live_bearings(selected_model):
    cycle = get_latest_cycle()
    bearing_card("UGB", selected_model, cycle)
    bearing_card("TGB", selected_model, cycle)
    status = _scheduler_snapshot()
    if cycle is None:
        st.info("Waiting for the first scheduled simulation cycle.", icon="ℹ️")
    elif not status["cycle_fresh"]:
        if status["active"]:
            st.warning("Scheduler is active, but the latest saved forecast is old. Awaiting its next completed cycle.")
        else:
            st.warning("Showing previously saved results. Scheduler heartbeat is offline or delayed.")
    st.caption(
        "The selected algorithm is shared by TGB and UGB; "
        "each bearing has its own trained model."
    )



def make_trend_chart(
    component,
    selected_model,
    cycles,
    accent,
    view="Operating range"
):
    return make_hydropulse_forecast_chart(
        component=component,
        selected_model=selected_model,
        cycles=cycles,
        accent=accent,
        operating_ranges=OPERATING_RANGES,
        view=view
    )

@st.fragment(run_every=REFRESH_INTERVAL)
def render_live_charts(selected_model):

    cycles = get_recent_cycles(limit=144)

    left_col, right_col = st.columns(2, gap="medium")

    with left_col:
        st.plotly_chart(
            make_trend_chart(
                "TGB",
                selected_model,
                cycles,
                "#42c8f5",
                view="Zoom into readings"
            ),
            use_container_width=True,
            config={
                "displaylogo": False,
                "displayModeBar": False
            }
        )

    with right_col:
        st.plotly_chart(
            make_trend_chart(
                "UGB",
                selected_model,
                cycles,
                "#ffb45a",
                view="Zoom into readings"
            ),
            use_container_width=True,
            config={
                "displaylogo": False,
                "displayModeBar": False
            }
        )

    st.caption(
        "The charts use a focused vertical scale to highlight "
        "short-term vibration trends. Operating reference levels "
        "are derived from historical training data and do not "
        "represent certified equipment safety limits."
    )


@st.fragment(run_every="1s")
def render_prediction_countdown():
    state = _scheduler_snapshot()
    st.markdown(
        '<div style="color:#EAF5FF;font-size:16px;font-weight:700;'
        'margin:12px 0 5px">Next Prediction</div>',
        unsafe_allow_html=True,
    )
    next_run = state["next_run"]
    now = state["now"]
    if not state["active"]:
        label = "—"
        note = "Start or check live_scheduler.py"
    elif next_run is None:
        label = "—"
        note = "Scheduled time not available yet"
    else:
        remaining = (next_run - now).total_seconds()
        if remaining > 0:
            # Round up so the countdown does not display 00:00 early.
            from math import ceil
            minutes, seconds = divmod(ceil(remaining), 60)
            label = f"{minutes:02d}:{seconds:02d}"
            note = "Time until next scheduled execution"
        elif remaining >= -60:
            label = "DUE"
            note = "Execution pending or in progress"
        else:
            label = "LATE"
            note = "Check for a missed or delayed cycle"
    st.markdown(
        '<div style="padding:2px 0 7px">'
        f'<div style="font-size:43px;font-weight:750;line-height:1.1;'
        f'letter-spacing:-1px;color:#44CCF6;font-variant-numeric:tabular-nums">'
        f'{escape(label)}</div>'
        f'<div style="font-size:12px;color:#AEC5D6;margin-top:5px">'
        f'{escape(note)}</div>'
        '</div>', unsafe_allow_html=True,
    )
    if next_run is not None:
        st.caption("Next scheduled: " + next_run.strftime("%H:%M:%S") + " MYT")
    st.caption("TGB + UGB · synchronized 10-minute forecasting")

@st.fragment(run_every=REFRESH_INTERVAL)
def render_prediction_performance(selected_model):

    verified = get_verified_predictions(
        limit=200,
        tolerance_seconds=60
    )

    if verified.empty:
        st.info(
            "Waiting for consecutive simulation cycles "
            "to verify prediction accuracy."
        )
        return

    summary = summarize_verified_predictions(verified)

    # Show rolling RMSE for the globally selected model
    left, right = st.columns(2, gap="medium")

    for column, component in [
        (left, "TGB"),
        (right, "UGB")
    ]:

        matching = summary[
            (summary["Component"] == component)
            & (summary["Model"] == selected_model)
        ]

        with column:

            if matching.empty:
                st.info(f"No verified {component} predictions yet.")
                continue

            result = matching.iloc[0]

            metric_card(
                f"{component} · {selected_model} Rolling RMSE",
                f"{result['RMSE']:.4f}",
                (
                    f"MAE: {result['MAE']:.4f} | "
                    f"Verified pairs: {int(result['Samples'])}"
                )
            )

    st.markdown(
        '<div class="ht-section-heading">'
        'Simulation Model Performance Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    # Display all three models for both bearings
    comparison = summary[
        [
            "Component",
            "Model",
            "Samples",
            "MAE",
            "RMSE",
            "Mean_Error"
        ]
    ].copy()

    comparison = comparison.rename(
        columns={"Mean_Error": "Mean Error"}
    )

    for column_name in ["MAE", "RMSE", "Mean Error"]:
        comparison[column_name] = comparison[
            column_name
        ].round(4)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Metrics are calculated by comparing saved "
        "10-minute forecasts against subsequent "
        "simulated sensor readings. "
        "These are synthetic simulation results, "
        "not independent real-plant test results. "
        "Smaller RMSE and MAE indicate lower prediction error."
    )

# Shared navigation and monitoring sidebar
global_model, bearing_focus = render_shared_sidebar(
    show_bearing_focus=True
)

# Display shared generating-unit operating parameters
render_live_kpis(global_model)

st.markdown(
    '<div class="ht-section-heading">3D Generating Unit & Bearing Monitoring</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.7, 1], gap="medium")
with left:
    try:
        figure = get_cad_figure(bearing_focus)
        if figure is None:
            st.info(f"CAD file not found: {GLB_PATH}. Place the GLB file there to display the model.")
        else:
            st.plotly_chart(
                figure, use_container_width=True,
                config={"displaylogo": False},
            )
    except Exception as exc:
        st.error(f"CAD viewer could not load: {exc}")
        st.caption("Verify HydroPulse_cad_viewer.py and the GLB asset are in digital_twin/.")
    st.markdown(
        '<div class="ht-footnote">Conceptual CAD geometry, not a plant-verified '
        'mechanical replica. Rotate and zoom to inspect both bearings.</div>',
        unsafe_allow_html=True,
    )
with right:
    render_live_bearings(global_model)

render_live_charts(global_model)

st.markdown(
    '<div class="ht-footnote">HydroPulse prototype · Machine-learning-driven '
    'synthetic sensor simulation · No physical SCADA connection or '
    'verified safety alarm thresholds.</div>',
    unsafe_allow_html=True,
)