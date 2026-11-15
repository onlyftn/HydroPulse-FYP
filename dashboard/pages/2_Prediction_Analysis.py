from pathlib import Path
import sys
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from live_data import get_latest_cycle, get_recent_cycles
from forecast_chart_upgrade import make_hydropulse_forecast_chart
from operating_ranges import calculate_operating_ranges
from prediction_verification import (
    get_verified_predictions,
    summarize_verified_predictions
)

# --------------------------------------------------
# MALAYSIA TIME FORMATTING
# --------------------------------------------------

MALAYSIA_TZ = ZoneInfo("Asia/Kuala_Lumpur")


def format_malaysia_time(timestamp):

    if not timestamp:
        return "—"

    dt = datetime.fromisoformat(
        str(timestamp).replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=MALAYSIA_TZ)

    dt = dt.astimezone(MALAYSIA_TZ)

    return dt.strftime("%d %b %Y · %H:%M:%S MYT")

st.set_page_config(
    page_title="HydroPulse | Prediction Analysis",
    page_icon="📈",
    layout="wide"
)

# Project paths
DASHBOARD_DIR = Path(__file__).resolve().parents[1]
from shared_sidebar import render_shared_sidebar

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

# Preserve HydroTwin styling
CSS_PATH = DASHBOARD_DIR / "styles.css"

if CSS_PATH.exists():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )

# Page header
st.title("Prediction Analysis")

st.caption(
    "HydroPulse | Neelum–Jhelum Hydropower Plant | "
    "Generating Unit G1"
)

st.info(
    "Real-time synthetic sensor simulation with "
    "next 10-minute vibration forecasts. "
    "No live plant SCADA connection.",
    icon="ℹ️"
)

# Global model selection for both bearings
MODEL_NAMES = ["Linear Regression", "LSTM", "GRU"]

selected_model, _ = render_shared_sidebar()


# --------------------------------------------------
# AUTOMATIC LIVE PREDICTION DISPLAY
# --------------------------------------------------

@st.fragment(run_every="5s")
def render_live_prediction_analysis(selected_model):

    # Read the latest completed simulation cycle
    cycle = get_latest_cycle()

    if cycle is None:
        st.warning(
            "No simulation results available. "
            "Start live_scheduler.py first."
        )
        return

    results = cycle["results"]
    ranges = calculate_operating_ranges()

    st.markdown("### Latest Simulation Cycle")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.caption("LATEST READING TIME")

        st.markdown(
            f"**{format_malaysia_time(cycle['scheduled_at'])}**"
        )

    with col2:
        st.caption("NEXT FORECAST TARGET")

        st.markdown(
            f"**{format_malaysia_time(cycle['forecast_at'])}**"
        )

    # Component forecasts
    st.markdown("### Component Forecasts")

    left, right = st.columns(2, gap="medium")

    for column, component in [
        (left, "TGB"),
        (right, "UGB")
    ]:

        bearing = results["components"][component]

        prediction = bearing["all_predictions"][
            selected_model
        ]

        lower = ranges[component]["lower"]
        upper = ranges[component]["upper"]

        status = (
            "Within Typical Range"
            if lower <= prediction <= upper
            else "Outside Typical Range"
        )

        component_name = (
            "Turbine Guide Bearing"
            if component == "TGB"
            else "Upper Guide Bearing"
        )

        with column:
            with st.container(border=True):

                st.subheader(
                    f"{component} | {component_name}"
                )

                st.metric(
                    "Simulated Vibration",
                    f"{bearing['vibration']:.3f}"
                )

                st.metric(
                    "Next 10-Minute Prediction",
                    f"{prediction:.4f}"
                )

                st.metric(
                    "Temperature",
                    f"{bearing['temperature']:.3f}"
                )

                st.write(
                    f"**Forecasting Model:** {selected_model}"
                )

                st.write(
                    "**Typical Operating Range:** "
                    f"{lower:.1f} – {upper:.1f}"
                )

                if status == "Within Typical Range":
                    st.success(status)
                else:
                    st.warning(status)

    # ---------------------------------------------
    # DETAILED VIBRATION FORECASTING GRAPHS
    # ---------------------------------------------

    st.markdown("### Vibration Forecast Analysis")

    recent_cycles = get_recent_cycles(limit=144)

    chart_left, chart_right = st.columns(2, gap="medium")

    for column, component, accent in [
        (chart_left, "TGB", "#42C8F5"),
        (chart_right, "UGB", "#FFB45A")
    ]:

        with column:

            fig = make_hydropulse_forecast_chart(
                component=component,
                selected_model=selected_model,
                cycles=recent_cycles,
                accent=accent,
                operating_ranges=ranges,
                view="Zoom into readings"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "displayModeBar": False
                }
            )

    st.caption(
        "The solid line represents recorded simulation readings, "
        "while the white diamond indicates the next 10-minute forecast. "
        "A focused vertical scale highlights short-term vibration trends. "
        "Historical operating references are not certified safety limits."
    )

        # ---------------------------------------------
    # LIVE PREDICTION ACCURACY
    # ---------------------------------------------

    st.markdown("### Prediction Accuracy")

    st.caption(
        "Accuracy calculated from completed synthetic "
        "simulation cycles. Lower RMSE and MAE "
        "indicate better forecasting performance."
    )

    verified = get_verified_predictions(
        limit=200,
        tolerance_seconds=60
    )

    if verified.empty:

        st.info(
            "Waiting for consecutive simulation "
            "cycles to verify prediction accuracy."
        )

    else:

        summary = summarize_verified_predictions(verified)

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
                with st.container(border=True):

                    st.markdown(
                        f"#### {component} | {selected_model}"
                    )

                    if matching.empty:
                        st.info("No verified predictions yet.")

                    else:

                        result = matching.iloc[0]

                        metric1, metric2 = st.columns(2)

                        with metric1:
                            st.metric(
                                "RMSE",
                                f"{result['RMSE']:.4f}"
                            )

                        with metric2:
                            st.metric(
                                "MAE",
                                f"{result['MAE']:.4f}"
                            )

                        st.caption(
                            f"Verified forecast pairs: "
                            f"{int(result['Samples'])}"
                        )

        # Compare all models using the same
        # completed synthetic simulation cycles

        st.markdown("#### Simulation Model Comparison")

        comparison = summary[
            ["Component", "Model", "Samples", "RMSE", "MAE"]
        ].copy()

        with st.container(key="ht_simulation_comparison_desktop"):
            st.dataframe(
                comparison,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "RMSE": st.column_config.NumberColumn(
                        "RMSE", format="%.4f"
                    ),
                    "MAE": st.column_config.NumberColumn(
                        "MAE", format="%.4f"
                    )
                }
            )

        # ---------------------------------------------
        # COMPACT MOBILE SIMULATION COMPARISON
        # ---------------------------------------------

        with st.container(key="ht_simulation_comparison_mobile"):

            tgb_tab, ugb_tab = st.tabs(["TGB", "UGB"])

            for tab, component in [
                (tgb_tab, "TGB"),
                (ugb_tab, "UGB")
            ]:

                with tab:

                    component_data = comparison[
                        comparison["Component"] == component
                    ]

                    if component_data.empty:
                        st.info("No verified simulation results yet.")

                    else:
                        selected_mobile_model = st.selectbox(
                            "Select Forecasting Model",
                            component_data["Model"].tolist(),
                            key=f"mobile_simulation_{component}"
                        )

                        record = component_data[
                            component_data["Model"]
                            == selected_mobile_model
                        ].iloc[0]

                        with st.container(border=True):

                            st.markdown(
                                f"**{component} — "
                                f"{selected_mobile_model}**"
                            )

                            st.caption(
                                "Verified forecast pairs: "
                                f"{int(record['Samples'])}"
                            )

                            metric1, metric2 = st.columns(2)

                            with metric1:
                                st.metric(
                                    "RMSE",
                                    f"{record['RMSE']:.4f}"
                                )

                            with metric2:
                                st.metric(
                                    "MAE",
                                    f"{record['MAE']:.4f}"
                                )

                        st.caption(
                            "Evaluation against synthetic "
                            "sensor readings."
                        )


        st.info(
            "These accuracy metrics are calculated "
            "against synthetic sensor readings. "
            "They are separate from the independent "
            "November 2020 test results.",
            icon="ℹ️"
        )

    st.caption(
        "Operating ranges are based on the 5th–95th "
        "percentiles of June–August 2020 training data. "
        "These are statistical references, not "
        "certified safety limits."
    )

    st.caption(
        "The page checks for newly completed "
        "simulation cycles every 5 seconds. "
        "Predictions are generated separately "
        "by the 10-minute scheduler."
    )



# Display the live prediction analysis
render_live_prediction_analysis(selected_model)