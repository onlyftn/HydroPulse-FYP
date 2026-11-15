"""Shared HydroPulse sidebar for all three Streamlit pages.

Uses saved synthetic simulation outputs. It does not execute ML inference.
"""

from datetime import datetime
from html import escape
from math import ceil
from zoneinfo import ZoneInfo

import streamlit as st

from live_data import get_latest_cycle, get_scheduler_health


MYT = ZoneInfo("Asia/Kuala_Lumpur")
MODEL_NAMES = ("Linear Regression", "LSTM", "GRU")


def _as_local(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        return dt.replace(tzinfo=MYT) if dt.tzinfo is None else dt.astimezone(MYT)
    except (ValueError, TypeError):
        return None


def _save_model_selection():
    st.session_state["ht_global_model"] = st.session_state["_ht_model_widget"]


@st.fragment(run_every="1s")
def _live_sidebar_monitor():
    health = get_scheduler_health()
    cycle = get_latest_cycle()
    now = datetime.now(MYT)

    heartbeat = _as_local(health.get("heartbeat_at")) if health else None
    next_run = _as_local(health.get("next_run_at")) if health else None

    active = (
        heartbeat is not None
        and -30 <= (now - heartbeat).total_seconds() <= 45
    )

    st.caption("SIMULATION STATUS")
    if active:
        st.success("Scheduler Active", icon="✅")
    else:
        st.warning("Scheduler not confirmed active", icon="⚠️")

    if cycle:
        completed = _as_local(cycle.get("completed_at"))
        if completed:
            st.caption("Last completed forecast: " + completed.strftime("%d %b %Y · %H:%M:%S MYT"))
        if active and completed and (now - completed).total_seconds() > 12 * 60:
            st.caption("A new forecast is overdue; check the scheduler terminal.")

    st.markdown(
        '<div style="font-weight:700;color:#EAF5FF;font-size:16px;'
        'margin:12px 0 5px">Next Prediction</div>',
        unsafe_allow_html=True,
    )

    if not active:
        display = "—"
        subtext = "Start live_scheduler.py"
    elif next_run is None:
        display = "—"
        subtext = "Awaiting schedule information"
    else:
        remaining = (next_run - now).total_seconds()
        if remaining > 0:
            minutes, seconds = divmod(ceil(remaining), 60)
            display = f"{minutes:02d}:{seconds:02d}"
            subtext = "Time to next scheduled execution"
        else:
            display = "DUE"
            subtext = "Awaiting scheduler update"

    st.markdown(
        '<div style="font-size:40px;font-weight:750;line-height:1.12;'
        'color:#44CCF6;font-variant-numeric:tabular-nums">'
        + escape(display)
        + '</div><div style="color:#AEC5D6;font-size:12px;margin-top:5px">'
        + escape(subtext)
        + '</div>',
        unsafe_allow_html=True,
    )
    if active and next_run:
        st.caption("Next scheduled: " + next_run.strftime("%H:%M:%S MYT"))
    st.caption("TGB + UGB · synchronized 10-minute forecasting")


def render_shared_sidebar(show_bearing_focus=False):
    """Render consistent navigation and controls; return (model, highlighted bearing)."""
    if "ht_global_model" not in st.session_state:
        st.session_state["ht_global_model"] = "GRU"

    with st.sidebar:
        st.markdown(
            '<div style="margin-bottom:10px">'
            '<div style="color:#EAF5FF;font-size:26px;font-weight:800">'
            'Hydro<span style="color:#44CCF6">Pulse</span></div>'
            '<div style="color:#9BB8CA;font-size:11px;letter-spacing:1.2px">'
            'PREDICTIVE MAINTENANCE SYSTEM</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("POWER PLANT")
        st.markdown("**Neelum–Jhelum Hydropower Plant**")
        st.caption("Generating Unit G1 · TGB + UGB")

        st.divider()
        st.caption("NAVIGATION")
        st.page_link("app.py", label="Overview", icon="🏠")
        st.page_link("pages/1_Model_Comparison.py", label="Model Comparison", icon="📊")
        st.page_link("pages/2_Prediction_Analysis.py", label="Prediction Analysis", icon="📈")
        #st.page_link("pages/3_Maintenance_Workspace.py",label="Maintenance Workspace",icon="🛠️")
        #st.page_link("pages/4_Maintenance_History.py", label="Maintenance History", icon="📋")


        st.divider()
        st.caption("GLOBAL FORECASTING MODEL")
        # Rehydrate the widget on every page. The durable key survives page changes.
        st.session_state["_ht_model_widget"] = st.session_state["ht_global_model"]
        st.radio(
            "Choose algorithm for both bearings",
            MODEL_NAMES,
            key="_ht_model_widget",
            on_change=_save_model_selection,
            label_visibility="collapsed",
            help="The same algorithm is displayed for TGB and UGB, using separately trained models.",
        )
        selected_model = st.session_state["ht_global_model"]
        st.caption("Same algorithm · separate trained TGB and UGB models")

        if show_bearing_focus:
            st.caption("3D COMPONENT HIGHLIGHT")
            bearing_focus = st.radio(
                "Highlight component", ["TGB", "UGB"],
                horizontal=True, label_visibility="collapsed",
                key="ht_bearing_focus",
            )
        else:
            bearing_focus = st.session_state.get("ht_bearing_focus", "TGB")

        st.divider()
        st.caption("SIMULATION SCHEDULE · MALAYSIA TIME")
        st.markdown("**Every 10 minutes from scheduler startup**")
        _live_sidebar_monitor()

        st.divider()
        st.caption("Synthetic sensor simulation · No physical plant SCADA connection.")

    return selected_model, bearing_focus