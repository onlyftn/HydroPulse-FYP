"""HydroPulse forecast chart presentation (synthetic simulation only).

Designed as a drop-in replacement for the existing make_trend_chart() code.
No changes to scheduler, inference, CAD, or global model selection.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import math
import plotly.graph_objects as go

MYT = ZoneInfo("Asia/Kuala_Lumpur")


def _parse_myt(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        return dt.replace(tzinfo=MYT) if dt.tzinfo is None else dt.astimezone(MYT)
    except (TypeError, ValueError):
        return None


def _finite_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def make_hydropulse_forecast_chart(
    component,
    selected_model,
    cycles,
    accent,
    operating_ranges,
    view="Operating range",
    max_readings=12,
):
    """Display uninterrupted recent simulated observations plus one next-step forecast.

    view: 'Operating range' plots the training-based 5-95% reference band;
          'Zoom into readings' emphasizes small changes and explicitly warns
          about truncated/zoomed y-axis.
    """
    if component not in ("TGB", "UGB"):
        raise ValueError("component must be TGB or UGB")
    if view not in ("Operating range", "Zoom into readings"):
        raise ValueError("Unknown view")

    lower = float(operating_ranges[component]["lower"])
    upper = float(operating_ranges[component]["upper"])
    records = []
    for cycle in cycles:
        current_time = _parse_myt(cycle.get("scheduled_at"))
        bearing = cycle.get("results", {}).get("components", {}).get(component, {})
        measured = _finite_float(bearing.get("vibration"))
        if current_time is None or measured is None:
            continue
        forecast = _finite_float(bearing.get("all_predictions", {}).get(selected_model))
        target = _parse_myt(cycle.get("forecast_at"))
        records.append({"time": current_time, "actual": measured,
                        "target": target, "forecast": forecast})
    records.sort(key=lambda r: r["time"])

    # Keep the last uninterrupted series; never imply a continuous trace over downtime.
    recent = []
    if records:
        recent = [records[-1]]
        for prior in reversed(records[:-1]):
            gap = (recent[0]["time"] - prior["time"]).total_seconds()
            if not (510 <= gap <= 690) or len(recent) >= max_readings:
                break
            recent.insert(0, prior)

    fig = go.Figure()
    fig.add_hrect(y0=lower, y1=upper,
                  fillcolor="rgba(45, 184, 137, 0.14)", line_width=0,
                  layer="below")
    fig.add_hline(y=upper, line_color="#F6B65D", line_width=1.8,
                  line_dash="dash",
                  annotation_text=f"Upper typical reference  {upper:g}",
                  annotation_position="top left",
                  annotation_font=dict(color="#F6C079", size=11))
    fig.add_hline(y=lower, line_color="#42CBA5", line_width=1.6, line_dash="dot")

    plotted = []
    x_range = None
    if recent:
        xs = [r["time"] for r in recent]
        ys = [r["actual"] for r in recent]
        plotted.extend(ys)
        last = recent[-1]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers" if len(xs) > 1 else "markers",
            name="Vibration Reading", line=dict(color=accent, width=2.8),
            marker=dict(size=8, color=accent),
            hovertemplate="Simulated reading<br>%{x|%d %b %Y, %H:%M:%S}<br>Vibration: %{y:.3f}<extra></extra>"
        ))
        target = last["target"]
        forecast = last["forecast"]
        valid_target = (target is not None and
                        540 <= (target - last["time"]).total_seconds() <= 660)
        if forecast is not None and valid_target:
            plotted.append(forecast)
            fig.add_vrect(x0=last["time"], x1=target,
                          fillcolor="rgba(221,237,255,0.05)", line_width=0,
                          layer="below")
            fig.add_trace(go.Scatter(
                x=[last["time"], target], y=[last["actual"], forecast],
                mode="lines", name="Next 10-min forecast",
                line=dict(color="#F3F7FB", width=2.8, dash="dash"),
                hovertemplate="%{x|%H:%M:%S}<br>Forecast path: %{y:.3f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=[target], y=[forecast], mode="markers", showlegend=False,
                marker=dict(symbol="diamond", size=13, color="#F3F7FB",
                            line=dict(color="#132436", width=1.3)),
                hovertemplate="Forecast target %{x|%H:%M:%S}<br>Predicted vibration: %{y:.3f}<extra></extra>"
            ))
            fig.add_annotation(x=target, y=forecast,
                               text=f"{forecast:.2f}", showarrow=False,
                               yshift=17, xanchor="center",
                               font=dict(color="#F3F7FB", size=12))

        # Vertical line identifies the latest simulated observation, not necessarily wall-clock now.
        fig.add_shape(type="line", x0=last["time"], x1=last["time"],
                      yref="paper", y0=0, y1=1,
                      line=dict(color="rgba(173,199,219,0.55)", width=1, dash="dot"))
        fig.add_annotation(x=last["time"], y=1.07, xref="x", yref="paper",
                           text="Latest Reading", showarrow=False,
                           font=dict(color="#BDD3E3", size=10))
        if len(recent) == 1:
            fig.add_annotation(x=0.03, y=0.1, xref="paper", yref="paper",
                               text="Trend builds with each 10-minute cycle",
                               showarrow=False, xanchor="left",
                               font=dict(color="#A6BBCB", size=10))
        start = (xs[0] - timedelta(minutes=7) if len(xs)>1
                 else last["time"]-timedelta(minutes=15))
        end = (target + timedelta(minutes=5) if forecast is not None and valid_target
               else last["time"] + timedelta(minutes=15))
        x_range = [start, end]
    else:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text="Waiting for simulation data", showarrow=False,
                           font=dict(color="#AAC2D4", size=13))

        # Calculate the Y-axis range
    if view == "Operating range" or not plotted:
        vals = [lower, upper] + plotted

        y_min = min(vals)
        y_max = max(vals)

        margin = max(
            0.25,
            (y_max - y_min) * 0.09
        )

        y_range = [
            y_min - margin,
            y_max + margin
        ]

    else:
        # Focused vertical scale for recent readings
        y_min = min(plotted)
        y_max = max(plotted)

        spread = y_max - y_min

        margin = max(
            0.12,
            spread * 0.35
        )

        y_range = [
            y_min - margin,
            y_max + margin
        ]

    fig.update_layout(
        title=dict(text=f"{component}  |  Vibration and next forecast",
                   font=dict(color=accent, size=17), x=0.02),
        paper_bgcolor="#0B1725", plot_bgcolor="#0B1725",
        font=dict(color="#DDEDF8", size=11),
        xaxis=dict(title="Malaysia time", tickformat="%H:%M", nticks=6,
                   gridcolor="#243B50", linecolor="#445A6F", showline=True,
                   **({"range": x_range} if x_range else {})),
        yaxis=dict(title="Vibration (dataset scale)", range=y_range,
                   gridcolor="#243B50", zeroline=False),
        legend=dict(orientation="h", x=0, y=-0.34, bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E0EFF9", size=11)),
        height=365, margin=dict(l=43, r=20, b=76, t=88),
        hovermode="x unified", showlegend=True,
    )
    return fig
