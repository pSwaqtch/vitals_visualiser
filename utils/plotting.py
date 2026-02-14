"""Plotly chart builders for sensor data visualization."""

import numpy as np
import plotly.graph_objects as go

COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]

_COMMON_LAYOUT = dict(
    template="plotly_white",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    hoverlabel=dict(namelength=-1),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
        bgcolor="rgba(0,0,0,0)", font=dict(size=11),
    ),
)

_GRID = "rgba(128,128,128,0.15)"


def plot_signals(df, timestamp_col: str | None, signal_cols: list[str]) -> go.Figure:
    """Raw signal chart with range slider."""
    fig = go.Figure()
    x = df[timestamp_col] if timestamp_col else df.index

    for i, col in enumerate(signal_cols):
        fig.add_trace(go.Scattergl(
            x=x, y=df[col], mode="lines", name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        **_COMMON_LAYOUT,
        xaxis=dict(title=None, gridcolor=_GRID,
                   rangeslider=dict(visible=True, thickness=0.04)),
        yaxis=dict(title=None, gridcolor=_GRID, zeroline=False, tickformat=",",
                   fixedrange=False),
        margin=dict(l=60, r=20, t=10, b=10),
        height=480,
        dragmode="select",
    )
    return fig


def plot_peaks(working_data: dict, measures: dict, sample_rate: float) -> go.Figure:
    """Filtered PPG with detected peaks overlay."""
    signal = np.array(working_data["hr"])
    peaklist = np.array(working_data["peaklist"])
    ybeat = np.array(working_data["ybeat"])
    x = np.arange(len(signal)) / sample_rate
    peak_x = peaklist / sample_rate

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=signal, mode="lines", name="Filtered",
        line=dict(color=COLORS[0], width=1),
        hovertemplate="t=%{x:.2f}s  val=%{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=peak_x, y=ybeat, mode="markers", name="Peaks",
        marker=dict(color=COLORS[1], size=7, symbol="circle",
                    line=dict(width=1, color="white")),
        hovertemplate="t=%{x:.2f}s<extra>Peak</extra>",
    ))

    fig.update_layout(
        **_COMMON_LAYOUT,
        xaxis=dict(title="Time (s)", gridcolor=_GRID),
        yaxis=dict(title=None, gridcolor=_GRID, zeroline=False),
        margin=dict(l=50, r=20, t=10, b=40),
        height=350,
        dragmode="select",
    )
    return fig


def plot_rr_tachogram(working_data: dict, sample_rate: float) -> go.Figure:
    """RR interval tachogram — beat-to-beat interval over time."""
    rr = np.array(working_data.get("RR_list_cor", working_data.get("RR_list", [])))
    peaklist = np.array(working_data["peaklist"])

    if len(rr) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough beats", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
        return fig

    # x-axis = time of each beat (use peak positions, skip first to align with RR)
    beat_times = peaklist[1:len(rr) + 1] / sample_rate

    mean_rr = np.mean(rr)

    fig = go.Figure()

    # Mean line
    fig.add_hline(y=mean_rr, line_dash="dash", line_color="rgba(128,128,128,0.4)",
                  annotation_text=f"mean {mean_rr:.0f} ms",
                  annotation_position="top right",
                  annotation_font_size=10, annotation_font_color="gray")

    fig.add_trace(go.Scattergl(
        x=beat_times, y=rr, mode="lines+markers", name="RR interval",
        line=dict(color=COLORS[2], width=1.5),
        marker=dict(size=4, color=COLORS[2]),
        hovertemplate="t=%{x:.1f}s<br>RR=%{y:.0f} ms<extra></extra>",
    ))

    fig.update_layout(
        **_COMMON_LAYOUT,
        xaxis=dict(title="Time (s)", gridcolor=_GRID),
        yaxis=dict(title="RR (ms)", gridcolor=_GRID, zeroline=False),
        margin=dict(l=50, r=20, t=10, b=40),
        height=300,
    )
    return fig


def plot_poincare(working_data: dict, measures: dict) -> go.Figure:
    """Poincare plot (RR_n vs RR_n+1) with SD1/SD2 ellipse."""
    rr = np.array(working_data.get("RR_list_cor", working_data.get("RR_list", [])))
    if len(rr) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Not enough beats", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
        return fig

    rr_n = rr[:-1]
    rr_n1 = rr[1:]
    mean_rr = np.mean(rr)

    fig = go.Figure()

    # Identity line
    lo, hi = rr.min() * 0.9, rr.max() * 1.1
    fig.add_trace(go.Scattergl(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="rgba(128,128,128,0.3)", dash="dash", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # SD1/SD2 ellipse
    sd1 = measures.get("sd1", 0)
    sd2 = measures.get("sd2", 0)
    if sd1 > 0 and sd2 > 0:
        theta = np.linspace(0, 2 * np.pi, 100)
        cos45, sin45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
        ex, ey = sd2 * np.cos(theta), sd1 * np.sin(theta)
        rx = mean_rr + ex * cos45 - ey * sin45
        ry = mean_rr + ex * sin45 + ey * cos45
        fig.add_trace(go.Scatter(
            x=rx, y=ry, mode="lines", name=f"SD1={sd1:.0f} SD2={sd2:.0f}",
            line=dict(color=COLORS[2], width=1.5), fill="toself",
            fillcolor="rgba(0,204,150,0.1)", hoverinfo="skip",
        ))

    fig.add_trace(go.Scattergl(
        x=rr_n, y=rr_n1, mode="markers", name="RR intervals",
        marker=dict(color=COLORS[0], size=5, opacity=0.6,
                    line=dict(width=0.5, color="white")),
        hovertemplate="RR_n=%{x:.0f}ms<br>RR_n+1=%{y:.0f}ms<extra></extra>",
    ))

    fig.update_layout(
        **_COMMON_LAYOUT,
        xaxis=dict(title="RR_n (ms)", gridcolor=_GRID,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(title="RR_n+1 (ms)", gridcolor=_GRID),
        margin=dict(l=50, r=20, t=10, b=40),
        height=300,
    )
    return fig
