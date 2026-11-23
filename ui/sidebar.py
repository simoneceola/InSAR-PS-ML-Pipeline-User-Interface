"""
Sidebar controls.

Phase 1 (before data load): CSV / GeoPackage pre-filters.
Phase 2 (after data load):  portfolio filters + ML + visual controls.

Controls that belong to a disabled ML module or a hidden page section are not shown;
their value falls back to the original default.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from app_config import (
    COLOR_MODE_REQUIRES, DEFAULT_CSV_PATH, DEFAULT_GPKG_PATH,
    SUB_CLUSTER_REQUIRES, SUB_HDBSCAN, available_options,
)


""" Sidebar controls for the InSAR Inspector app.
        - Filters for the CSV and GeoPackage data (before loading)
        - Filters for the portfolio (after loading)
        - Controls for the ML modules (after loading)
        - Controls for the visualizations (after loading)
"""
@dataclass(frozen=True)
class DataFilters:
    csv_path: str
    gpkg_path: str
    filter_height_wrt_dem: bool
    min_h_dem: float
    max_h_dem: float
    filter_coherence: bool
    min_coh: float
    max_coh: float
    min_pts_per_bld: int


@dataclass(frozen=True)
class Settings:
    # portfolio filters
    min_h_filter: float = 0.0
    only_sig_dh: bool = False
    # ML controls
    rf_conf_threshold: float = 0.70
    sub_cluster_method: str = SUB_HDBSCAN
    hdbscan_min_cluster: int = 4
    stgnn_k: int = 6
    stgnn_zones: int = 3
    thermal_max_lag: int = 3
    # visual controls
    top_bottom_frac: float = 0.30
    color_mode: str = "LOS Velocity (mm/yr)"
    show_stgnn_edges: bool = False
    ts_display_mode: str = "Raw + Fitted Model Overlay"
    show_temp_overlay: bool = True
    show_season_bands: bool = True


def render_data_filters() -> DataFilters:
    """Draw the pre-load CSV / GeoPackage filters in the sidebar and return their values.
    """
    with st.sidebar:
        st.header("0. CSV & GeoPackage Pre-Filters")
        csv_path_input = st.text_input("InSAR CSV path", value=str(DEFAULT_CSV_PATH))
        gpkg_path_input = st.text_input("Building polygons (.gpkg)", value=str(DEFAULT_GPKG_PATH))

        filter_height_wrt_dem = st.checkbox("Filter Height WRT DEM", value=True)
        c_h1, c_h2 = st.columns(2)
        min_h_dem = c_h1.number_input("Min H above DEM (m)", value=0.01, step=1.0)
        max_h_dem = c_h2.number_input("Max H above DEM (m)", value=300.0, step=10.0)

        filter_coherence = st.checkbox("Filter Coherence & Quality", value=True)
        c_c1, c_c2 = st.columns(2)
        min_coh = c_c1.number_input("Min coherence", value=0.80, min_value=0.0, max_value=1.0, step=0.05)
        max_coh = c_c2.number_input("Max coherence", value=1.00, min_value=0.0, max_value=1.0, step=0.05)

        min_pts_per_bld = st.slider("Min PS points per building", 3, 50, 12, 1)

    return DataFilters(
        csv_path=csv_path_input,
        gpkg_path=gpkg_path_input,
        filter_height_wrt_dem=filter_height_wrt_dem,
        min_h_dem=float(min_h_dem),
        max_h_dem=float(max_h_dem),
        filter_coherence=filter_coherence,
        min_coh=float(min_coh),
        max_coh=float(max_coh),
        min_pts_per_bld=int(min_pts_per_bld),
    )


def render_analysis_controls(
    intra_df: pd.DataFrame,
    enabled_ml: frozenset[str],
    enabled_sections: set[str],
) -> Settings:
    """Draw the post-load portfolio filters, ML controls, and visual controls in the sidebar.
    """
    d = Settings()  # defaults
    has = lambda m: m in enabled_ml
    show = lambda s: s in enabled_sections
    values: dict = {}

    with st.sidebar:
        if not enabled_sections:
            return Settings(**values)

        # ---------------- 1. ML controls
        st.divider()
        st.header("1. Machine Learning Controls")
        if has("random_forest"):
            values["rf_conf_threshold"] = st.slider("Random Forest confidence threshold", 0.40, 0.95, 0.70, 0.05)

        sub_options = available_options(SUB_CLUSTER_REQUIRES, enabled_ml)
        values["sub_cluster_method"] = st.selectbox("Sub-Building Clustering Algorithm", options=sub_options, index=0)

        if has("hdbscan"):
            values["hdbscan_min_cluster"] = st.slider("HDBSCAN min cluster size", 2, 12, 4, 1)
        if has("stgnn"):
            values["stgnn_k"] = st.slider("ST-GNN 3D graph neighbors (k)", 3, 12, 6, 1)
            values["stgnn_zones"] = st.slider("ST-GNN contiguous 3D zones", 2, 6, 3, 1)
        if has("thermal_inertia"):
            values["thermal_max_lag"] = st.slider("Lagged thermal inertia max lag (epochs)", 1, 6, 3, 1)
        if has("isolation_forest") or has("autoencoder"):
            st.caption("Outliers: automatic (`contamination='auto'` & Median + 3.5×MAD RMSE).")

        # ---------------- 2. visual controls
        st.divider()
        st.header("2. 3D & Temporal Visuals")
        values["top_bottom_frac"] = st.slider("Top / Bottom quantile fraction", 0.15, 0.45, 0.30, 0.05)

        if show("scatterer_3d"):
            values["color_mode"] = st.selectbox(
                "3D Point Coloring Mode",
                options=available_options(COLOR_MODE_REQUIRES, enabled_ml),
                index=0,
            )
            if has("stgnn"):
                values["show_stgnn_edges"] = st.checkbox("Show 3D k-NN spatial graph edges", value=False)
        if show("height_strata"):
            values["ts_display_mode"] = st.selectbox(
                "Time-Series Signal Mode",
                options=[
                    "Raw + Fitted Model Overlay",
                    "Raw Data Only",
                    "De-seasonalized (Permanent Trend Only)",
                ],
                index=0,
            )
        if show("clustered_timeseries") or show("height_strata"):
            values["show_temp_overlay"] = st.checkbox("Overlay Temperature (°C) on Top − Bottom", value=True)
        if show("all_timeseries") or show("height_strata") or show("single_point"):
            values["show_season_bands"] = st.checkbox("Shade Summer (Jun–Aug) & Winter (Dec–Feb)", value=True)

    return Settings(**values)