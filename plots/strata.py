"""Height strata (Bottom / Middle / Top) + Top−Bottom differential vs. temperature."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.temporal import TemporalRateFitter
from plots.helpers import add_season_shading


def build_building_timeseries_figure(
    res: dict[str, Any],
    dates_dt: pd.DatetimeIndex,
    temp_c: np.ndarray,
    ts_display_mode: str,
    show_temp_overlay: bool,
    show_season_bands: bool,
    baseline_epochs: int,
) -> go.Figure:
    fitter: TemporalRateFitter = res["fitter"]
    n_ref = min(baseline_epochs, len(dates_dt))
    deseasonalize = ts_display_mode == "De-seasonalized (Permanent Trend Only)"
    show_fit = ts_display_mode == "Raw + Fitted Model Overlay"
    thermal_on = "thermal_inertia" in res["enabled_ml"]

    def _prep(ts: np.ndarray | None) -> np.ndarray | None:
        if ts is None:
            return None
        out = fitter.deseasonalize(ts) if deseasonalize else ts
        return out - out[:n_ref].mean()

    bot_ts = _prep(res["bot_ts"])
    mid_ts = _prep(res["mid_ts"])
    top_ts = _prep(res["top_ts"])
    tb_ts = top_ts - bot_ts

    left_title = (
        f"1. Height Strata & {res['gbdt_name']} Median Fit (R² = {res['gbdt_r2']:.2f})"
        if thermal_on
        else "1. Height Strata"
    )
    fig = make_subplots(
        rows=1, cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.08,
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        subplot_titles=(
            left_title,
            f"2. Top − Bottom Differential & Temperature (r = {res['tb_temp_r']:+.2f})",
        ),
    )

    if show_season_bands:
        add_season_shading(fig, dates_dt, cols=[1, 2])

    for series, name, color in [
        (bot_ts, f"Bottom ({res['n_bot']} pts)", "#2b6cb0"),
        (mid_ts, f"Middle ({res['n_mid']} pts)", "#718096"),
        (top_ts, f"Top ({res['n_top']} pts)", "#c53030"),
    ]:
        if series is not None:
            fig.add_trace(
                go.Scatter(x=dates_dt, y=series, mode="lines", name=name, line=dict(color=color, width=2)),
                row=1, col=1,
            )
    if show_fit and thermal_on:
        gbdt_med = res["gbdt_pred_med"] - res["gbdt_pred_med"][:n_ref].mean()
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=gbdt_med, mode="lines",
                name=f"{res['gbdt_name']} Lagged-Temp Fit",
                line=dict(color="#1a202c", width=1.8, dash="dot"),
            ),
            row=1, col=1,
        )
    fig.update_yaxes(title_text="LOS Displacement (mm)", row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)
    fig.add_trace(
        go.Scatter(x=dates_dt, y=tb_ts, mode="lines", name="Top − Bottom (mm)", line=dict(color="#6b46c1", width=2.2)),
        row=1, col=2, secondary_y=False,
    )
    if show_fit:
        tb_decomp = fitter.decompose(tb_ts)
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=tb_decomp["total_fit"], mode="lines",
                name="Harmonic + Trend Fit", line=dict(color="#1a202c", width=1.8, dash="dash"),
            ),
            row=1, col=2, secondary_y=False,
        )
    if show_temp_overlay:
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=temp_c, mode="lines", name="Temperature (°C)",
                line=dict(color="rgba(221, 107, 32, 0.55)", width=1.4, dash="dot"),
            ),
            row=1, col=2, secondary_y=True,
        )
        fig.update_yaxes(title_text="Temp (°C)", showgrid=False, row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Differential (mm)", row=1, col=2, secondary_y=False)

    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"Building-Scale Strata Evolution — {ts_display_mode}", font=dict(size=13)),
        height=350,
        margin=dict(l=50, r=45, t=65, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0.0),
        hovermode="x unified",
    )
    return fig
