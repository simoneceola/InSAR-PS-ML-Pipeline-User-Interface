"""3D point cloud of a building with switchable colouring (ML layers, velocity, strata…)."""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from app_config import CLUSTER_PALETTE, LINEARITY_COLORS, REGIME_COLORS


def build_3d_building_figure(
    res: dict[str, Any],
    color_mode: str,
    show_stgnn_edges: bool,
    highlighted_pt: int | None = None,
) -> go.Figure:
    e = res["e_local"]
    n = res["n_local"]
    h = res["h_rel"]
    lat = res["lat"]
    lon = res["lon"]

    fig = go.Figure()

    if len(e) >= 2 and np.ptp(e) > 0 and np.ptp(n) > 0:
        pad = 1.5
        eg, ng = np.meshgrid(
            [float(e.min()) - pad, float(e.max()) + pad],
            [float(n.min()) - pad, float(n.max()) + pad],
        )
        fig.add_trace(
            go.Surface(
                x=eg, y=ng, z=np.zeros_like(eg),
                colorscale=[[0, "rgba(160, 174, 192, 0.25)"], [1, "rgba(160, 174, 192, 0.25)"]],
                showscale=False,
                name="Ground base (z = 0 m)",
                hoverinfo="skip",
            )
        )

    if show_stgnn_edges and res["edge_pairs"]:
        ex, ey, ez = [], [], []
        for i, j in res["edge_pairs"]:
            ex.extend([e[i], e[j], None])
            ey.extend([n[i], n[j], None])
            ez.extend([h[i], h[j], None])
        fig.add_trace(
            go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                name="3D k-NN Spatial Graph",
                line=dict(color="rgba(113, 128, 150, 0.30)", width=1.5),
                hoverinfo="skip",
            )
        )

    def _add_categorical_traces(labels_arr: np.ndarray, ordered_cats: list[str], color_map: dict[str, str]):
        for cat in ordered_cats:
            mask = labels_arr == cat
            if not np.any(mask):
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=e[mask], y=n[mask], z=h[mask],
                    mode="markers",
                    name=f"{cat} ({mask.sum()})",
                    marker=dict(
                        size=5.5,
                        color=color_map.get(cat, "#718096"),
                        line=dict(width=0.5, color="white"),
                    ),
                    customdata=np.column_stack([
                        np.where(mask)[0],
                        res["rates"][mask],
                        res["v_thermal_clean"][mask],
                        res["regime_labels"][mask],
                        res["linearity_labels"][mask],
                    ]),
                    hovertemplate=(
                        "<b>Point #%{customdata[0]}</b> (" + cat + ")<br>"
                        "E: %{x:.2f} m | N: %{y:.2f} m | H: %{z:.2f} m<br>"
                        "Raw LOS: %{customdata[1]:+.2f} | Clean LOS: %{customdata[2]:+.2f} mm/yr<br>"
                        "RF: %{customdata[3]} / %{customdata[4]}<extra></extra>"
                    ),
                )
            )

    if color_mode == "RF Kinematic Regime":
        _add_categorical_traces(
            res["regime_labels"],
            ["Stable", "Sinking", "Uplifting", "Unclassified"],
            REGIME_COLORS,
        )
    elif color_mode == "RF Trajectory Linearity":
        _add_categorical_traces(res["linearity_labels"], ["Linear", "Non-linear"], LINEARITY_COLORS)
    elif color_mode == "DTW Time-Series Cluster (Phase-Lag Aware)":
        unique_lbls = sorted(np.unique(res["dtw_labels"]).tolist())
        cat_names = [f"DTW Cluster {c}" for c in unique_lbls]
        lbl_str = np.array([f"DTW Cluster {c}" for c in res["dtw_labels"]], dtype=object)
        cmap = {name: CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)] for idx, name in enumerate(cat_names)}
        _add_categorical_traces(lbl_str, cat_names, cmap)
    elif color_mode == "HDBSCAN Density Cluster":
        unique_lbls = sorted(np.unique(res["hdbscan_labels"]).tolist())
        cat_names = ["Noise (-1)" if c == -1 else f"Cluster {c}" for c in unique_lbls]
        lbl_str = np.array(["Noise (-1)" if c == -1 else f"Cluster {c}" for c in res["hdbscan_labels"]], dtype=object)
        cmap = {
            name: ("#a0aec0" if c == -1 else CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)])
            for idx, (c, name) in enumerate(zip(unique_lbls, cat_names))
        }
        _add_categorical_traces(lbl_str, cat_names, cmap)
    elif color_mode == "ST-GNN 3D Structural Zone":
        unique_lbls = sorted(np.unique(res["stgnn_labels"]).tolist())
        cat_names = [f"3D Zone {c}" for c in unique_lbls]
        lbl_str = np.array([f"3D Zone {c}" for c in res["stgnn_labels"]], dtype=object)
        cmap = {name: CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)] for idx, name in enumerate(cat_names)}
        _add_categorical_traces(lbl_str, cat_names, cmap)
    elif color_mode == "Autoencoder + Isolation Forest Anomalies":
        comb = np.where(
            res["is_ae_anomaly"] & res["is_if_anomaly"],
            "Both (AE + IF Outlier)",
            np.where(
                res["is_ae_anomaly"],
                "Autoencoder Outlier",
                np.where(res["is_if_anomaly"], "Isolation Forest Outlier", "Normal Inlier"),
            ),
        )
        _add_categorical_traces(
            comb,
            ["Normal Inlier", "Autoencoder Outlier", "Isolation Forest Outlier", "Both (AE + IF Outlier)"],
            {
                "Normal Inlier": "#4a5568",
                "Autoencoder Outlier": "#dd6b20",
                "Isolation Forest Outlier": "#805ad5",
                "Both (AE + IF Outlier)": "#e53e3e",
            },
        )
    elif color_mode == "Height Stratum":
        _add_categorical_traces(
            res["stratum_labels"],
            ["Bottom", "Middle", "Top"],
            {"Bottom": "#2b6cb0", "Middle": "#718096", "Top": "#c53030"},
        )
    else:
        if color_mode == "Thermal-Adjusted Velocity (Lagged Ridge)":
            values = res["v_thermal_clean"]
            cscale = "RdBu_r"
            vmax = max(float(np.percentile(np.abs(values), 95)), 0.1)
            cmin, cmax = -vmax, vmax
            cbar_title = "Clean Rate<br>(mm/yr)"
        elif color_mode == "Thermal Lag (Acquisitions)":
            values = res["effective_lag_epochs"]
            cscale = "Plasma"
            cmin, cmax = 0.0, max(float(np.max(values)), 0.5)
            cbar_title = "Thermal Lag<br>(epochs)"
        elif color_mode == "Autoencoder Reconstruction MSE":
            values = res["ae_recon_mse"]
            cscale = "Inferno"
            cmin, cmax = 0.0, max(float(np.percentile(values, 95)), 0.05)
            cbar_title = "AE MSE"
        elif color_mode == "Thermal Amplitude (mm)" and res["seasonal_amp"] is not None:
            values = res["seasonal_amp"]
            cscale = "Viridis"
            cmin, cmax = 0.0, max(float(np.percentile(values, 95)), 0.1)
            cbar_title = "Seasonal Amp<br>(mm)"
        else:
            values = res["rates"]
            cscale = "RdBu_r"
            vmax = max(float(np.percentile(np.abs(values), 95)), 0.1)
            cmin, cmax = -vmax, vmax
            cbar_title = "LOS Rate<br>(mm/yr)"

        fig.add_trace(
            go.Scatter3d(
                x=e, y=n, z=h,
                mode="markers",
                name=f"PS Scatterers ({len(h)})",
                marker=dict(
                    size=5.5,
                    color=values,
                    colorscale=cscale,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title=cbar_title, thickness=14, x=1.02),
                    line=dict(width=0.5, color="white"),
                ),
                customdata=np.column_stack([
                    np.arange(len(h)),
                    res["rates"],
                    res["v_thermal_clean"],
                    res["effective_lag_epochs"],
                    lat,
                    lon,
                ]),
                hovertemplate=(
                    "<b>Point #%{customdata[0]}</b><br>"
                    "E: %{x:.2f} m | N: %{y:.2f} m | H: %{z:.2f} m<br>"
                    "Lat: %{customdata[4]:.5f} | Lon: %{customdata[5]:.5f}<br>"
                    "Raw LOS: %{customdata[1]:+.2f} | Clean LOS: %{customdata[2]:+.2f} mm/yr<br>"
                    "Thermal Lag: %{customdata[3]:.1f} epochs<extra></extra>"
                ),
            )
        )

    if highlighted_pt is not None and 0 <= highlighted_pt < len(h):
        fig.add_trace(
            go.Scatter3d(
                x=[e[highlighted_pt]], y=[n[highlighted_pt]], z=[h[highlighted_pt]],
                mode="markers+text",
                text=[f"Point #{highlighted_pt}"],
                textposition="top center",
                name=f"Selected Point #{highlighted_pt}",
                marker=dict(size=10, color="gold", symbol="diamond", line=dict(width=2, color="black")),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"3D Building Map & ML Layers — Building #{res['building']} ({color_mode})",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Height above base (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=-1.55, z=1.05)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
        margin=dict(l=0, r=0, t=55, b=0),
        height=520,
    )
    return fig
