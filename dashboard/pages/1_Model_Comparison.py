from pathlib import Path
import sys

DASHBOARD_DIR = Path(__file__).resolve().parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from shared_sidebar import render_shared_sidebar


import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HydroTwin | Model Comparison",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison.csv"
)

CSS_PATH = (
    PROJECT_ROOT
    / "dashboard"
    / "styles.css"
)

if CSS_PATH.exists():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# LOAD FINAL TEST RESULTS
# --------------------------------------------------

@st.cache_data
def load_results():

    data = pd.read_csv(RESULTS_PATH)

    required = [
        "Component", "Model",
        "RMSE", "MAE", "MAPE", "R2"
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return data


if not RESULTS_PATH.exists():
    st.error(
        "Final model comparison file not found."
    )
    st.code(str(RESULTS_PATH))
    st.stop()

results = load_results()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Model Comparison")

st.caption(
    "HydroTwin | Neelum–Jhelum Hydropower Plant "
    "| Generating Unit G1"
)

st.info(
    "Independent November 2020 test evaluation. "
    "These results are separate from live synthetic "
    "simulation performance.",
    icon="📊"
)


# --------------------------------------------------
# BEST TEST RESULTS
# --------------------------------------------------

st.subheader("Best November Test Performance")

left, right = st.columns(2)

for column, component in [
    (left, "TGB"),
    (right, "UGB")
]:

    component_data = results[
        results["Component"] == component
    ]

    winner = component_data.loc[
        component_data["RMSE"].idxmin()
    ]

    with column:
        st.metric(
            label=f"{component} | Lowest Test RMSE",
            value=f"{winner['RMSE']:.4f}",
            help=f"Model: {winner['Model']}"
        )

        st.caption(
            f"Best observed test model: "
            f"**{winner['Model']}**"
        )


# --------------------------------------------------
# RMSE COMPARISON CHARTS
# --------------------------------------------------

st.subheader("RMSE Comparison")

MODEL_COLORS = {
    "Linear Regression": "#94A3B8",
    "LSTM": "#42C8F5",
    "GRU": "#F5B45D"
}


def make_rmse_chart(component):

    data = results[
        results["Component"] == component
    ].copy()

    model_order = [
        "Linear Regression",
        "LSTM",
        "GRU"
    ]

    data["Model"] = pd.Categorical(
        data["Model"],
        categories=model_order,
        ordered=True
    )

    data = data.sort_values("Model")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data["Model"],
            y=data["RMSE"],
            marker_color=[
                MODEL_COLORS[str(model)]
                for model in data["Model"]
            ],
            text=[
                f"{value:.4f}"
                for value in data["RMSE"]
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>"
                "RMSE: %{y:.4f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{component} | Test RMSE",
            font=dict(
                size=18,
                color="#FFFFFF"
            )
        ),
        paper_bgcolor="#0B1725",
        plot_bgcolor="#0B1725",
        font=dict(color="#DCEBF6"),
        xaxis=dict(
            title="Forecasting Model",
            showgrid=False
        ),
        yaxis=dict(
            title="RMSE (dataset scale)",
            gridcolor="#243748",
            rangemode="tozero"
        ),
        height=390,
        margin=dict(
            l=30, r=20, t=65, b=40
        ),
        showlegend=False
    )

    fig.update_yaxes(
        range=[
            0,
            float(data["RMSE"].max()) * 1.20
        ]
    )

    return fig


chart_left, chart_right = st.columns(2)

with chart_left:
    st.plotly_chart(
        make_rmse_chart("TGB"),
        use_container_width=True
    )

with chart_right:
    st.plotly_chart(
        make_rmse_chart("UGB"),
        use_container_width=True
    )


# --------------------------------------------------
# COMPLETE PERFORMANCE METRICS
# --------------------------------------------------

st.subheader("Complete Model Performance")

comparison = results[
    ["Component", "Model", "RMSE", "MAE", "MAPE", "R2"]
].copy()

comparison = comparison.sort_values(
    ["Component", "RMSE"]
)

comparison = comparison.rename(
    columns={
        "MAPE": "MAPE (%)",
        "R2": "R²"
    }
)

with st.container(key="ht_model_performance_desktop"):
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "RMSE": st.column_config.NumberColumn(
                "RMSE", format="%.4f"
            ),
            "MAE": st.column_config.NumberColumn(
                "MAE", format="%.4f"
            ),
            "MAPE (%)": st.column_config.NumberColumn(
                "MAPE (%)", format="%.2f"
            ),
            "R²": st.column_config.NumberColumn(
                "R²", format="%.4f"
            )
        }
    )



# ---------------------------------------------
# COMPACT MOBILE MODEL PERFORMANCE
# ---------------------------------------------

with st.container(key="ht_model_performance_mobile"):

    tgb_tab, ugb_tab = st.tabs(["TGB", "UGB"])

    for tab, component in [
        (tgb_tab, "TGB"),
        (ugb_tab, "UGB")
    ]:

        with tab:

            component_data = comparison[
                comparison["Component"] == component
            ]

            selected_model = st.selectbox(
                "Select Forecasting Model",
                component_data["Model"].tolist(),
                key=f"mobile_model_{component}"
            )

            record = component_data[
                component_data["Model"] == selected_model
            ].iloc[0]

            with st.container(border=True):

                st.markdown(
                    f"**{component} — {selected_model}**"
                )

                st.markdown(
                    f"**RMSE:** {record['RMSE']:.4f}  \n"
                    f"**MAE:** {record['MAE']:.4f}  \n"
                    f"**MAPE (%):** {record['MAPE (%)']:.2f}  \n"
                    f"**R²:** {record['R²']:.4f}"
                )

            st.caption(
                "Independent November 2020 test results."
            )



# --------------------------------------------------
# PROFESSIONAL PERFORMANCE INTERPRETATION
# --------------------------------------------------

st.markdown("## Performance Interpretation")

st.caption(
    "Independent evaluation | November 2020 test dataset | "
    "Next 10-minute vibration forecasting"
)

# Retrieve the best test model for each component
tgb_results = results[
    results["Component"] == "TGB"
]

ugb_results = results[
    results["Component"] == "UGB"
]

tgb_best = tgb_results.loc[
    tgb_results["RMSE"].idxmin()
]

ugb_best = ugb_results.loc[
    ugb_results["RMSE"].idxmin()
]


# --------------------------------------------------
# COMPONENT PERFORMANCE CARDS
# --------------------------------------------------

col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(border=True):

        st.markdown("### Turbine Guide Bearing (TGB)")

        st.metric(
            label="Lowest November Test RMSE",
            value=f"{tgb_best['RMSE']:.4f}"
        )

        st.markdown(
            f"**Best Test Model: {tgb_best['Model']}**"
        )

        st.markdown(
            "LSTM achieved the lowest forecasting error "
            "on the independent November test set. "
            "GRU demonstrated closely comparable "
            "performance, while Linear Regression "
            "recorded a higher RMSE."
        )

with col2:
    with st.container(border=True):

        st.markdown("### Upper Guide Bearing (UGB)")

        st.metric(
            label="Lowest November Test RMSE",
            value=f"{ugb_best['RMSE']:.4f}"
        )

        st.markdown(
            f"**Best Test Model: {ugb_best['Model']}**"
        )

        st.markdown(
            "Linear Regression achieved the lowest "
            "forecasting error on the November test set, "
            "outperforming both recurrent models. "
            "This highlights the importance of "
            "generalization beyond model complexity."
        )


# --------------------------------------------------
# KEY TECHNICAL FINDING
# --------------------------------------------------

st.markdown("### Key Technical Finding")

with st.container(border=True):

    st.markdown("**UGB Distribution Shift**")

    st.markdown(
        "Approximately **43.05% of November UGB "
        "observations** were below the minimum "
        "vibration value represented in the training "
        "targets (11.0)."
    )

    st.markdown(
        "The recurrent models tended to overpredict "
        "these lower readings, contributing to their "
        "higher test errors. This suggests that a "
        "change in the observed vibration distribution "
        "was an important generalization challenge."
    )

    st.caption(
        "This observation does not independently "
        "establish equipment deterioration or failure."
    )


# --------------------------------------------------
# MODEL SELECTION METHODOLOGY
# --------------------------------------------------

st.markdown("### Model Selection and Evaluation")

selection_summary = pd.DataFrame({
    "Component": ["TGB", "UGB"],
    "September Validation Selection": [
        "GRU",
        "LSTM"
    ],
    "November Test Leader": [
        str(tgb_best["Model"]),
        str(ugb_best["Model"])
    ]
})

with st.container(key="ht_model_selection_desktop"):
    st.dataframe(
        selection_summary,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------
# MOBILE MODEL SELECTION CARDS
# ---------------------------------------------

with st.container(key="ht_model_selection_mobile"):

    for record in selection_summary.to_dict("records"):

        with st.container(border=True):

            st.markdown(
                f"### {record['Component']} — Model Selection"
            )

            st.markdown(
                "**September Validation Selection**"
            )

            st.write(
                record["September Validation Selection"]
            )

            st.markdown(
                "**November Test Leader**"
            )

            st.write(
                record["November Test Leader"]
            )


st.info(
    "Model configuration and selection were based on "
    "September validation results. November 2020 "
    "was reserved for independent final evaluation "
    "and was not used for model tuning or "
    "retroactive selection.",
    icon="ℹ️"
)

st.caption(
    "RMSE and other error metrics are reported "
    "in the original dataset vibration scale. "
    "Physical measurement units have not been "
    "independently verified. These results "
    "describe forecasting performance, not "
    "certified equipment safety."
)

render_shared_sidebar()