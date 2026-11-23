"""2D portfolio map: footprints, haloed markers and PS scatterers."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

def _build_footprint_feature_collection(
    bld_ids: list[int],
    building_datasets: dict[int, dict],
) -> dict[str, Any]:
    """GeoJSON FeatureCollection of building polygons (convex-hull fallback)."""
    features = []
    for bid in bld_ids:
        bdata = building_datasets.get(bid)
        if not bdata:
            continue
        geom = bdata.get("polygon_geojson")
        if geom is None:
            lons = np.asarray(bdata["lon"], dtype=float)
            lats = np.asarray(bdata["lat"], dtype=float)
            if len(lons) >= 3:
                try:
                    pts = np.column_stack([lons, lats])
                    hull = ConvexHull(pts)
                    ring = pts[hull.vertices].tolist()
                    ring.append(ring[0])
                    geom = {"type": "Polygon", "coordinates": [ring]}
                except Exception:
                    geom = None
        if geom is not None:
            features.append({"type": "Feature", "properties": {"building": int(bid)}, "geometry": geom})
    return {"type": "FeatureCollection", "features": features}


def _compute_map_center_and_zoom(
    lats: np.ndarray, lons: np.ndarray, padding_factor: float = 1.40
) -> tuple[float, float, float]:
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    center_lat = 0.5 * (lat_min + lat_max)
    center_lon = 0.5 * (lon_min + lon_max)

    lat_span = max((lat_max - lat_min) * padding_factor, 0.0014)
    lon_span = max((lon_max - lon_min) * padding_factor * np.cos(np.radians(center_lat)), 0.0014)

    zoom_lat = np.log2(180.0 / lat_span)
    zoom_lon = np.log2(360.0 / lon_span)
    zoom = float(np.clip(min(zoom_lat, zoom_lon), 9.0, 17.5))
    return center_lat, center_lon, zoom


def build_portfolio_map(
    df: pd.DataFrame,
    selected_id: int,
    building_datasets: dict[int, dict],
    basemap_choice: str = "Clean Light (Carto Positron)",
    color_by: str = "Vertical Gradient dh/dt (mm/yr/m)",
    camera_focus: str = "Fit All Filtered Buildings",
    show_footprints_and_ps: bool = True,
) -> go.Figure:
    use_maplibre = hasattr(go, "Scattermap")
    TraceCls = go.Scattermap if use_maplibre else go.Scattermapbox

    halo_color = "#1a202c"

    style_map = {
        "Light Theme": "carto-positron",
        "OpenStreetMap": "open-street-map",
    }
    map_style = style_map.get(basemap_choice, "carto-positron")

    hover_text = [
        f"<b>Building #{r.building}</b><br>"
        f"PS Points: <b>{r.n_pts}</b><br>"
        f"Height wrt Ground: {r.h_min_m:.1f} – {r.h_max_m:.1f} m (span {r.h_range_m:.1f} m)<br>"
        for r in df.itertuples()
    ]

    fig = go.Figure()

    # 1. Optional individual PS scatterer overlay
    if show_footprints_and_ps:
        all_ps_lats, all_ps_lons = [], []
        for bid in df["building"]:
            bdata = building_datasets.get(bid)
            if bdata is not None:
                all_ps_lats.append(np.asarray(bdata["lat"], dtype=float))
                all_ps_lons.append(np.asarray(bdata["lon"], dtype=float))
        if all_ps_lats:
            ps_lat_cat = np.concatenate(all_ps_lats)
            ps_lon_cat = np.concatenate(all_ps_lons)
            fig.add_trace(
                TraceCls(
                    lat=ps_lat_cat,
                    lon=ps_lon_cat,
                    mode="markers",
                    marker=dict(
                        size=4.5,
                        color="rgba(43, 108, 176, 0.55)",
                    ),
                    hoverinfo="skip",
                    name=f"PS Scatterers ({len(ps_lat_cat)})",
                )
            )

    # 2. Halo ring so every building marker pops on any basemap
    fig.add_trace(
        TraceCls(
            lat=df["lat_center"],
            lon=df["lon_center"],
            mode="markers",
            marker=dict(size=17, color=halo_color),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 3. Highlight outer ring for the active building (drawn BEFORE inner colored dot)
    sel_row = df[df["building"] == selected_id]
    if not sel_row.empty:
        fig.add_trace(
            TraceCls(
                lat=sel_row["lat_center"],
                lon=sel_row["lon_center"],
                mode="markers+text",
                marker=dict(size=23, color="#f6e05e"),
                text=[f"★ Building #{selected_id}"],
                textposition="top center",
                textfont=dict(size=12, color="#1a202c"),
                hoverinfo="skip",
                name="Selected Building",
            )
        )

    # 4. Inner coloured building centroid markers
    if color_by == "Median LOS Velocity (mm/yr)":
        vals = df["v_median"]
        vmax = max(float(np.nanpercentile(np.abs(vals), 95)), 0.5)
        cmin, cmax = -vmax, vmax
        cscale = "RdBu_r"
        cbar_title = "LOS Rate<br>(mm/yr)"
    elif color_by == "PS Point Count":
        vals = df["n_pts"]
        cmin, cmax = float(vals.min()), max(float(vals.max()), float(vals.min()) + 1.0)
        cscale = "Plasma"
        cbar_title = "PS Pts"
    elif color_by == "Max Height wrt Ground (m)":
        vals = df["h_max_m"]
        cmin, cmax = float(vals.min()), max(float(vals.max()), float(vals.min()) + 1.0)
        cscale = "Viridis"
        cbar_title = "Max H<br>(m)"
    else:
        col_name = "dh_dt" if "dh_dt" in df.columns else "v_median"
        vals = df[col_name]
        vmax = max(float(np.nanpercentile(np.abs(vals), 95)), 0.1)
        cmin, cmax = -vmax, vmax
        cscale = "RdBu_r"
        cbar_title = "dh/dt<br>(mm/yr/m)"

    fig.add_trace(
        TraceCls(
            lat=df["lat_center"],
            lon=df["lon_center"],
            mode="markers",
            marker=dict(
                size=12,
                color=vals,
                colorscale=cscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title=cbar_title, thickness=12, x=1.01),
            ),
            text=hover_text,
            hoverinfo="text",
            name="Buildings with PS",
        )
    )

    # 5. Layers: GeoJSON footprints
    layers = []

    if show_footprints_and_ps:
        all_fc = _build_footprint_feature_collection(df["building"].tolist(), building_datasets)
        if all_fc["features"]:
            layers.extend([
                dict(
                    sourcetype="geojson",
                    source=all_fc,
                    type="fill",
                    color="rgba(49, 130, 206, 0.26)",
                    below="traces",
                ),
                dict(
                    sourcetype="geojson",
                    source=all_fc,
                    type="line",
                    color="#1a365d",
                    line=dict(width=2.0),
                    below="traces",
                ),
            ])
        sel_fc = _build_footprint_feature_collection([selected_id], building_datasets)
        if sel_fc["features"]:
            layers.extend([
                dict(
                    sourcetype="geojson",
                    source=sel_fc,
                    type="fill",
                    color="rgba(246, 224, 94, 0.55)",
                    below="traces",
                ),
                dict(
                    sourcetype="geojson",
                    source=sel_fc,
                    type="line",
                    color="#e53e3e",
                    line=dict(width=3.0),
                    below="traces",
                ),
            ])

    # 6. Viewport
    if camera_focus == "Zoom to Selected Building" and selected_id in building_datasets:
        b_active = building_datasets[selected_id]
        center_lat, center_lon, zoom = _compute_map_center_and_zoom(
            np.asarray(b_active["lat"], dtype=float),
            np.asarray(b_active["lon"], dtype=float),
            padding_factor=2.5,
        )
    else:
        center_lat, center_lon, zoom = _compute_map_center_and_zoom(
            df["lat_center"].to_numpy(dtype=float),
            df["lon_center"].to_numpy(dtype=float),
            padding_factor=1.40,
        )

    map_cfg = dict(
        style=map_style,
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom,
        layers=layers,
    )
    layout_kwargs = {"map": map_cfg} if use_maplibre else {"mapbox": map_cfg}

    fig.update_layout(
        **layout_kwargs,
        margin=dict(l=0, r=0, t=0, b=0),
        height=440,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.85)",
            font=dict(size=10, color="#1a202c"),
        ),
    )
    return fig