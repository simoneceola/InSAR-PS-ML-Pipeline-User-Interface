"""All PS time series of one building (gray) + overall median (bold red)."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plots.helpers import add_season_shading, pack_multilines


def build_all_points_median_figure(
    res: dict[str, Any],
    dates_dt: pd.DatetimeIndex,
    signal_choice: str = "Raw Displacement (Zero-Referenced)",
    show_iqr_band: bool = True,
    show_season_bands: bool = True,
    highlighted_pt: int | None = None,
) -> go.Figure:
    if signal_choice == "Thermal-Cleaned (Lagged Ridge)":
        series_2d = res["permanent_zeroed"]
        v_summary = res["v_clean_median"]
    elif signal_choice == "STL Trend (De-seasonalized)":
        series_2d = res["stl_trends_zeroed"]
        v_summary = res["v_clean_median"]
    else:
        series_2d = res["disp_zeroed"]
        v_summary = res["v_median"]

    n_pts = series_2d.shape[0]
    med_ts = np.nanmedian(series_2d, axis=0)
    q25_ts = np.nanpercentile(series_2d, 25, axis=0)
    q75_ts = np.nanpercentile(series_2d, 75, axis=0)

    fig = go.Figure()

    if show_season_bands:
        add_season_shading(fig, dates_dt)

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(113, 128, 150, 0.6)")

    x_gray, y_gray = pack_multilines(dates_dt, series_2d)
    fig.add_trace(
        go.Scatter(
            x=x_gray, y=y_gray, mode="lines",
            name=f"Individual PS Scatterers ({n_pts} pts)",
            line=dict(color="rgba(128, 128, 128, 0.28)", width=1.1),
            hoverinfo="skip",
        )
    )

    if show_iqr_band and n_pts >= 3:
        dates_list = dates_dt.strftime("%Y-%m-%d").tolist()
        fig.add_trace(
            go.Scatter(
                x=dates_list + dates_list[::-1],
                y=np.concatenate([q75_ts, q25_ts[::-1]]),
                fill="toself",
                fillcolor="rgba(229, 62, 62, 0.13)",
                line=dict(color="rgba(229, 62, 62, 0.0)"),
                name="25th–75th Percentile (IQR)",
                hoverinfo="skip",
            )
        )

    if highlighted_pt is not None and 0 <= highlighted_pt < n_pts:
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=series_2d[highlighted_pt], mode="lines",
                name=f"Selected Point #{highlighted_pt} (H: {res['h_rel'][highlighted_pt]:.1f}m)",
                line=dict(color="#d69e2e", width=2.0, dash="dash"),
                hovertemplate=(
                    f"<b>Point #{highlighted_pt}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>Disp: %{y:+.2f} mm<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates_dt, y=med_ts, mode="lines",
            name=f"Overall Building Median (v = {v_summary:+.2f} mm/yr)",
            line=dict(color="#e53e3e", width=3.2),
            hovertemplate=(
                "<b>Overall Building Median</b><br>"
                "Date: %{x|%Y-%m-%d}<br>Median Disp: %{y:+.2f} mm<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"Building #{res['building']} — All {n_pts} PS Time Series (Gray) & Overall Median (Red) [{signal_choice}]",
            font=dict(size=14),
        ),
        xaxis_title="Date",
        yaxis_title="LOS Displacement (mm)",
        height=380,
        margin=dict(l=50, r=25, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        hovermode="x unified",
    )
    return fig
