"""Section: every scatterer (gray) + overall building median (red)."""

import streamlit as st

from app_config import SIGNAL_OPTIONS_ALL
from app_context import AppContext
from plots.all_points import build_all_points_median_figure


def render(ctx: AppContext) -> None:
    st.subheader(f"All Scatterer Time Series (Gray) & Overall Building Median (Red) — Building #{ctx.selected_id}")
    selected_pt = ctx.selected_point()

    thermal_on = "thermal_inertia" in ctx.res["enabled_ml"]
    signal_options = SIGNAL_OPTIONS_ALL if thermal_on else [s for s in SIGNAL_OPTIONS_ALL if "Thermal" not in s]

    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        signal = st.radio("Trajectory Signal:", options=signal_options, horizontal=True, key="all_pts_signal_radio")
    with c2:
        show_iqr = st.checkbox("Show 25th–75th percentile spread", value=True, key="all_pts_iqr_chk")

    st.plotly_chart(
        build_all_points_median_figure(
            res=ctx.res,
            dates_dt=ctx.dates_dt,
            signal_choice=signal,
            show_iqr_band=show_iqr,
            show_season_bands=ctx.settings.show_season_bands,
            highlighted_pt=selected_pt,
        ),
        use_container_width=True,
    )
