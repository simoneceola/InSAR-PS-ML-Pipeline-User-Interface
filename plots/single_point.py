"""Single scatterer diagnostic: trajectories (left) and detrended motion vs. temperature (right)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from plots.helpers import add_season_shading


def build_single_point_figure(
    pt_idx: int,
    dates_dt: pd.DatetimeIndex,
    temp_c: np.ndarray,
    stl_trend_zeroed: np.ndarray,
    permanent_zeroed: np.ndarray,
    ae_recon_zeroed: np.ndarray,
    decomp: dict[str, np.ndarray],
    temp_link: dict[str, float],
    show_season_bands: bool,
    show_thermal_clean: bool = True,
    show_ae_recon: bool = True,
) -> go.Figure:
    parts = ["Raw", "STL Trend"]
    if show_ae_recon:
        parts.append("Autoencoder Recon")
    if show_thermal_clean:
        parts.append("Thermal-Cleaned")
    left_title = f"Point #{pt_idx}: " + ", ".join(parts[:-1]) + (" & " if len(parts) > 1 else "") + parts[-1] + " Trajectory"

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.66, 0.34],
        horizontal_spacing=0.08,
        subplot_titles=(
            left_title,
            f"Detrended Motion vs. Temperature (r = {temp_link['r']:+.2f}, {temp_link['slope_mm_per_c']:+.2f} mm/°C)",
        ),
    )

    if show_season_bands:
        add_season_shading(fig, dates_dt, cols=[1])

    fig.add_trace(
        go.Scatter(
            x=dates_dt, y=decomp["raw"], mode="markers+lines", name="Raw PS measurement",
            marker=dict(size=4.5, color="rgba(49, 130, 206, 0.45)"),
            line=dict(color="rgba(49, 130, 206, 0.25)", width=1),
        ),
        row=1, col=1,
    )
    if show_thermal_clean:
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=permanent_zeroed, mode="lines", name="Thermal-Cleaned Series (Lagged Ridge)",
                line=dict(color="#2f855a", width=2.2),
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=dates_dt, y=stl_trend_zeroed, mode="lines", name="STL Trend (RF input)",
            line=dict(color="#805ad5", width=2.0, dash="dash"),
        ),
        row=1, col=1,
    )
    if show_ae_recon:
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=ae_recon_zeroed, mode="lines", name="Autoencoder Reconstruction",
                line=dict(color="#dd6b20", width=1.6, dash="dot"),
            ),
            row=1, col=1,
        )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="LOS Displacement (mm)", row=1, col=1)

    detrended = decomp["raw"] - decomp["trend_only"]
    fig.add_trace(
        go.Scatter(
            x=temp_c, y=detrended, mode="markers", name="Epoch vs. Temp",
            marker=dict(size=7, color=dates_dt.month, colorscale="Twilight", cmin=1, cmax=12, showscale=False),
            hovertemplate="Temp: %{x:.1f} °C<br>Detrended Disp: %{y:+.2f} mm<extra></extra>",
        ),
        row=1, col=2,
    )
    if np.std(temp_c) > 1e-6:
        slope, intercept, *_ = stats.linregress(temp_c, detrended)
        t_line = np.array([temp_c.min(), temp_c.max()])
        fig.add_trace(
            go.Scatter(
                x=t_line, y=intercept + slope * t_line, mode="lines", name="Thermal sensitivity",
                line=dict(color="#dd6b20", width=2, dash="dash"), hoverinfo="skip",
            ),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Detrended Displacement (mm)", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0.0),
    )
    return fig
