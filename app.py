"""
Interactive Streamlit + Plotly Inspector for Single-Orbit Intra-Building PS-InSAR Behaviour.

Run with:   streamlit run app.py

This file only wires things together. To change WHAT is shown, edit page_layout.py.

    page_layout.py   → switch page sections / ML models on or off, reorder sections
    app_config.py    → paths, colours, defaults
    data_io/         → CSV + GeoPackage loading, synthetic data, temperature
    analysis/        → physics (rates, strata, vertical gradient), pipeline, diagnosis text
    ml/              → one file per ML model
    plots/           → pure Plotly figure builders
    sections/        → one file per page section (calls plots/, reads the context)
    ui/              → sidebar controls and the scatterer selector
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Single-Orbit InSAR Full ML Inspector",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

from analysis.pipeline import analyse_single_building  # noqa: E402
from analysis.temporal import estimate_annual_period  # noqa: E402
from app_config import APP_DIR, ALL_ML_MODULES, AnalysisConfig  # noqa: E402
from app_context import AppContext  # noqa: E402
from data_io.loader import load_csv_and_gpkg_workspace  # noqa: E402
from ml.random_forest import train_random_forest_models  # noqa: E402
from page_layout import ML_MODULES, PORTFOLIO  # noqa: E402
from sections import get_active_sections  # noqa: E402
from sections.portfolio import render_portfolio  # noqa: E402
from ui.sidebar import render_analysis_controls, render_data_filters  # noqa: E402


def main() -> None:
    st.title("Building and PS InSAR Time Series Inspector")

    active_sections = get_active_sections()
    enabled_sections = {key for key, _ in active_sections}
    enabled_ml = frozenset(m for m in ALL_ML_MODULES if ML_MODULES.get(m, False))

    # ---------------- 1. data pre-filters + load
    filters = render_data_filters()
    try:
        (
            building_datasets, intra_df, dates_dt, temp_c, source_label, temp_is_measured,
        ) = load_csv_and_gpkg_workspace(
            csv_path_str=filters.csv_path,
            gpkg_path_str=filters.gpkg_path,
            filter_height_wrt_dem=filters.filter_height_wrt_dem,
            min_h_dem=filters.min_h_dem,
            max_h_dem=filters.max_h_dem,
            filter_coherence=filters.filter_coherence,
            min_coh=filters.min_coh,
            max_coh=filters.max_coh,
            min_pts_per_building=filters.min_pts_per_bld,
        )
    except Exception as exc:
        st.error(f"Error loading dataset: {exc}")
        return

    period = estimate_annual_period(dates_dt)

    if "Synthetic" in source_label:
        st.info(
            f"Running with **{source_label}** (could not find `{Path(filters.csv_path).name}` and "
            f"`{Path(filters.gpkg_path).name}` in `{APP_DIR}`). Update the sidebar paths if they live elsewhere.",
            icon="ℹ️",
        )
    else:
        st.caption(
            f"This is using: {source_label} | {len(intra_df)} buildings (numbered 1 to {len(intra_df)}) | "
            f"{'Measured `tavg`' if temp_is_measured else 'Climatological T'}."
        )

    # ---------------- 2. sidebar (portfolio + ML + visual controls) & filtering
    settings = render_analysis_controls(intra_df, enabled_ml, enabled_sections)

    mask = intra_df["h_range_m"] >= settings.min_h_filter
    if settings.only_sig_dh and "dh_ols_p" in intra_df.columns:
        mask &= intra_df["dh_ols_p"] < 0.05
    filtered_df = intra_df.loc[mask].reset_index(drop=True)
    if filtered_df.empty:
        st.warning("No buildings match the current sidebar filters. Relax a filter to continue.")
        return

    # ---------------- 3. top row: map + table + building selector
    selected_id = render_portfolio(
        filtered_df=filtered_df,
        intra_df=intra_df,
        building_datasets=building_datasets,
        show_map=PORTFOLIO.get("map", True),
        show_table=PORTFOLIO.get("table", True),
    )

    if not active_sections:
        return  # nothing below the portfolio row → skip the expensive analysis

    # ---------------- 4. per-building analysis (only enabled ML modules run)
    rf_regime = rf_linearity = None
    if "random_forest" in enabled_ml:
        rf_regime, rf_linearity = train_random_forest_models(len(dates_dt), period)

    cfg = AnalysisConfig(
        top_bottom_frac=settings.top_bottom_frac,
        rf_conf_threshold=settings.rf_conf_threshold,
        sub_cluster_method=settings.sub_cluster_method,
        hdbscan_min_cluster_size=settings.hdbscan_min_cluster,
        stgnn_k_neighbors=settings.stgnn_k,
        stgnn_n_zones=settings.stgnn_zones,
        thermal_max_lag=settings.thermal_max_lag,
        enabled_ml=enabled_ml,
    )
    try:
        res = analyse_single_building(
            selected_id, building_datasets[selected_id], dates_dt, temp_c, cfg, period, rf_regime, rf_linearity
        )
    except Exception as exc:
        st.error(f"Could not analyse Building #{selected_id}: {exc}")
        return

    ctx = AppContext(
        building_datasets=building_datasets,
        intra_df=intra_df,
        filtered_df=filtered_df,
        dates_dt=dates_dt,
        temp_c=temp_c,
        temp_is_measured=temp_is_measured,
        period=period,
        cfg=cfg,
        settings=settings,
        selected_id=selected_id,
        res=res,
    )

    # ---------------- 5. page sections, in the order set in page_layout.py
    for _key, section in active_sections:
        if section.divider:
            st.divider()
        section.render(ctx)


if __name__ == "__main__":
    main()
