"""Raw vs. thermal-cleaned velocity-vs-height profile (top) + latent trajectory space (bottom)."""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_config import CLUSTER_PALETTE


def build_ml_latent_and_profile_figure(
    res: dict[str, Any], sub_cluster_method: str, highlighted_pt: int | None = None
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.18,
        subplot_titles=(
            f"Raw vs. Thermal-Adjusted Velocity Profile (Clean dh/dt: {res['dh_clean_theilsen']:+.3f} mm/yr/m)",
            f"Latent Trajectory Space — {sub_cluster_method}",
        ),
    )

    h_rel = res["h_rel"]
    rates = res["rates"]
    v_clean = res["v_thermal_clean"]

    fig.add_trace(
        go.Scatter(
            x=h_rel, y=rates, mode="markers", name="Raw LOS Rate",
            marker=dict(size=6, color="rgba(113, 128, 150, 0.55)", symbol="circle"),
            hovertemplate="Point #%{pointNumber}<br>Height: %{x:.1f} m | Raw: %{y:+.2f} mm/yr<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=h_rel, y=v_clean, mode="markers", name="Thermal-Cleaned Rate (Lagged Ridge)",
            marker=dict(size=7, color="#2b6cb0", symbol="diamond", opacity=0.85),
            hovertemplate="Point #%{pointNumber}<br>Height: %{x:.1f} m | Clean: %{y:+.2f} mm/yr<extra></extra>",
        ),
        row=1, col=1,
    )

    if len(h_rel) > 3 and np.ptp(h_rel) > 0:
        x_line = np.array([h_rel.min(), h_rel.max()])
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=res["dh_clean_intercept"] + res["dh_clean_theilsen"] * x_line,
                mode="lines", name="Clean Theil-Sen slope",
                line=dict(color="#1a202c", width=1.8, dash="dash"),
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.update_xaxes(title_text="Height above base (m)", row=1, col=1)
    fig.update_yaxes(title_text="LOS Rate (mm/yr)", row=1, col=1)

    ae_on = "autoencoder" in res["enabled_ml"]
    if ae_on and ("Autoencoder" in sub_cluster_method or "ST-GNN" in sub_cluster_method):
        emb = res["ae_latent_2d"]
        x_lbl, y_lbl = "AE Latent 1", "AE Latent 2"
    else:
        emb = res["embedding_2d"]
        x_lbl = "t-SNE 1" if "t-SNE" in sub_cluster_method else "PC 1"
        y_lbl = "t-SNE 2" if "t-SNE" in sub_cluster_method else "PC 2"

    if "DTW" in sub_cluster_method:
        lbls, prefix = res["dtw_labels"], "DTW"
    elif "ST-GNN" in sub_cluster_method:
        lbls, prefix = res["stgnn_labels"], "Zone"
    elif "HDBSCAN" in sub_cluster_method:
        lbls, prefix = res["hdbscan_labels"], "HDB"
    else:
        lbls, prefix = res["dbscan_labels"], "DB"

    unique_lbls = sorted(np.unique(lbls).tolist())
    for idx, lbl in enumerate(unique_lbls):
        mask = lbls == lbl
        name = "Noise (-1)" if lbl == -1 else f"{prefix} {lbl}"
        color = "#a0aec0" if lbl == -1 else CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=emb[mask, 0], y=emb[mask, 1], mode="markers", name=name,
                marker=dict(size=8, color=color, opacity=0.85, line=dict(width=0.5, color="white")),
                customdata=np.column_stack([np.where(mask)[0], h_rel[mask], v_clean[mask]]),
                hovertemplate=(
                    "Point #%{customdata[0]} (" + name + ")<br>"
                    "Height: %{customdata[1]:.1f} m | Clean Rate: %{customdata[2]:+.2f} mm/yr<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    if highlighted_pt is not None and 0 <= highlighted_pt < len(h_rel):
        fig.add_trace(
            go.Scatter(
                x=[h_rel[highlighted_pt]], y=[v_clean[highlighted_pt]],
                mode="markers", name=f"Selected #{highlighted_pt}",
                marker=dict(size=12, color="gold", symbol="star", line=dict(width=1.5, color="black")),
                hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[emb[highlighted_pt, 0]], y=[emb[highlighted_pt, 1]],
                mode="markers", showlegend=False,
                marker=dict(size=12, color="gold", symbol="star", line=dict(width=1.5, color="black")),
                hoverinfo="skip",
            ),
            row=2, col=1,
        )

    fig.update_xaxes(title_text=x_lbl, row=2, col=1)
    fig.update_yaxes(title_text=y_lbl, row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0, font=dict(size=10)),
    )
    return fig
