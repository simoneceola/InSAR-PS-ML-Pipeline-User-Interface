"""Time series split by a chosen partition (DTW / HDBSCAN / ST-GNN / RF / AE-DBSCAN)."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_config import (
    CLUSTER_PALETTE, LINEARITY_COLORS, REGIME_COLORS,
    SUB_AE, SUB_HDBSCAN, SUB_STGNN,
)
from plots.helpers import pack_multilines


def _partition_labels(res: dict[str, Any], cluster_source: str) -> tuple[np.ndarray, list[str]]:
    """Return (per-point label strings, ordered unique label names) for the chosen partition."""
    if cluster_source == "RF Kinematic Regime":
        cat_labels = res["regime_labels"]
        ordered = [c for c in ["Stable", "Sinking", "Uplifting", "Unclassified"] if np.any(cat_labels == c)]
    elif cluster_source == "RF Trajectory Linearity":
        cat_labels = res["linearity_labels"]
        ordered = [c for c in ["Linear", "Non-linear"] if np.any(cat_labels == c)]
    elif cluster_source in (SUB_HDBSCAN, "HDBSCAN Density Cluster"):
        raw = res["hdbscan_labels"]
        keys = sorted(np.unique(raw).tolist())
        cat_labels = np.array(["Noise (-1)" if c == -1 else f"HDBSCAN Cluster {c}" for c in raw], dtype=object)
        ordered = ["Noise (-1)" if c == -1 else f"HDBSCAN Cluster {c}" for c in keys]
    elif cluster_source in (SUB_STGNN, "ST-GNN 3D Structural Zone"):
        raw = res["stgnn_labels"]
        keys = sorted(np.unique(raw).tolist())
        cat_labels = np.array([f"3D Zone {c}" for c in raw], dtype=object)
        ordered = [f"3D Zone {c}" for c in keys]
    elif cluster_source == SUB_AE:
        raw = res["dbscan_labels"]
        keys = sorted(np.unique(raw).tolist())
        cat_labels = np.array(["Noise (-1)" if c == -1 else f"AE-DBSCAN {c}" for c in raw], dtype=object)
        ordered = ["Noise (-1)" if c == -1 else f"AE-DBSCAN {c}" for c in keys]
    else:  # DTW (default)
        raw = res["dtw_labels"]
        keys = sorted(np.unique(raw).tolist())
        cat_labels = np.array([f"DTW Cluster {c}" for c in raw], dtype=object)
        ordered = [f"DTW Cluster {c}" for c in keys]
    return cat_labels, ordered


def build_clustered_timeseries_figure(
    res: dict[str, Any],
    dates_dt: pd.DatetimeIndex,
    temp_c: np.ndarray,
    cluster_source: str,
    signal_choice: str = "Raw Displacement (Zero-Referenced)",
    show_temp_overlay: bool = True,
) -> go.Figure:
    if signal_choice == "Thermal-Cleaned (Lagged Ridge)":
        series_2d = res["permanent_zeroed"]
        rates_arr = res["v_thermal_clean"]
    elif signal_choice == "STL Trend (De-seasonalized)":
        series_2d = res["stl_trends_zeroed"]
        rates_arr = res["v_thermal_clean"]
    else:
        series_2d = res["disp_zeroed"]
        rates_arr = res["rates"]

    heights_arr = res["h_rel"]
    member_word = "pts"
    height_label = "H̄"

    cat_labels, ordered_names = _partition_labels(res, cluster_source)
    if not ordered_names:
        ordered_names = ["All"]
        cat_labels = np.full(series_2d.shape[0], "All", dtype=object)

    n_clusters = len(ordered_names)
    n_cols = min(3, max(1, n_clusters))
    n_grid_rows = int(np.ceil(n_clusters / n_cols))
    total_rows = 1 + n_grid_rows

    specs: list[list[dict | None]] = [[{"colspan": n_cols, "secondary_y": True}] + [None] * (n_cols - 1)]
    for _ in range(n_grid_rows):
        specs.append([{} for _ in range(n_cols)])

    subplot_titles = [f"All Cluster Medians Comparison ({cluster_source})"]
    for cname in ordered_names:
        mask = cat_labels == cname
        cnt = int(mask.sum())
        mean_h = float(np.nanmean(heights_arr[mask])) if cnt > 0 else 0.0
        med_v = float(np.nanmedian(rates_arr[mask])) if cnt > 0 else 0.0
        subplot_titles.append(f"{cname} (n={cnt} {member_word}, {height_label}={mean_h:.1f}m, v={med_v:+.2f} mm/yr)")
    while len(subplot_titles) < 1 + n_grid_rows * n_cols:
        subplot_titles.append("")

    row_heights = [0.42] + [0.58 / n_grid_rows] * n_grid_rows
    fig = make_subplots(
        rows=total_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.14 if n_grid_rows == 1 else 0.11,
        horizontal_spacing=0.06,
        row_heights=row_heights,
    )

    color_lookup: dict[str, str] = {}
    for idx, cname in enumerate(ordered_names):
        if cname in REGIME_COLORS:
            color_lookup[cname] = REGIME_COLORS[cname]
        elif cname in LINEARITY_COLORS:
            color_lookup[cname] = LINEARITY_COLORS[cname]
        elif "Noise" in cname:
            color_lookup[cname] = "#a0aec0"
        else:
            color_lookup[cname] = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(113, 128, 150, 0.5)", row=1, col=1)
    for cname in ordered_names:
        mask = cat_labels == cname
        if not np.any(mask):
            continue
        c_med = np.nanmedian(series_2d[mask], axis=0)
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=c_med, mode="lines",
                name=f"{cname} ({mask.sum()})",
                line=dict(color=color_lookup[cname], width=2.6),
                hovertemplate=f"<b>{cname}</b><br>Date: %{{x|%Y-%m-%d}}<br>Median: %{{y:+.2f}} mm<extra></extra>",
            ),
            row=1, col=1, secondary_y=False,
        )

    if show_temp_overlay:
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=temp_c, mode="lines", name="Temperature (°C)",
                line=dict(color="rgba(221, 107, 32, 0.45)", width=1.4, dash="dot"),
            ),
            row=1, col=1, secondary_y=True,
        )
        fig.update_yaxes(title_text="Temp (°C)", showgrid=False, row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="LOS Disp (mm)", row=1, col=1, secondary_y=False)

    y_min_global = float(np.nanpercentile(series_2d, 1))
    y_max_global = float(np.nanpercentile(series_2d, 99))
    y_pad = max(1.0, 0.1 * (y_max_global - y_min_global))

    for idx, cname in enumerate(ordered_names):
        r = 2 + (idx // n_cols)
        c = 1 + (idx % n_cols)
        mask = cat_labels == cname
        if not np.any(mask):
            continue

        sub_series = series_2d[mask]
        c_med = np.nanmedian(sub_series, axis=0)
        x_mem, y_mem = pack_multilines(dates_dt, sub_series)

        fig.add_hline(y=0, line_dash="dot", line_color="rgba(113, 128, 150, 0.45)", row=r, col=c)
        fig.add_trace(
            go.Scatter(
                x=x_mem, y=y_mem, mode="lines",
                line=dict(color="rgba(128, 128, 128, 0.30)", width=1.0),
                showlegend=False, hoverinfo="skip",
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=dates_dt, y=c_med, mode="lines",
                line=dict(color=color_lookup[cname], width=2.8),
                showlegend=False,
                hovertemplate=f"<b>{cname} Median</b><br>Date: %{{x|%Y-%m-%d}}<br>Disp: %{{y:+.2f}} mm<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.update_yaxes(range=[y_min_global - y_pad, y_max_global + y_pad], row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text="Disp (mm)", row=r, col=c)

    fig.update_layout(
        template="plotly_white",
        height=310 + 235 * n_grid_rows,
        margin=dict(l=50, r=45, t=65, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
        hovermode="x unified",
    )
    return fig
