"""Section: time series grouped by the chosen clustering / classifier partition."""

import streamlit as st

from app_config import CLUSTER_VIEW_REQUIRES, SIGNAL_OPTIONS_ALL, available_options
from app_context import AppContext
from plots.clustered import build_clustered_timeseries_figure


def render(ctx: AppContext) -> None:
    st.subheader("Clustered Time Series Behaviour")
    en = ctx.res["enabled_ml"]

    view_options = available_options(CLUSTER_VIEW_REQUIRES, en)
    if not view_options:
        st.info("No clustering / classification module is enabled (see `page_layout.py → ML_MODULES`).")
        return

    default = ctx.settings.sub_cluster_method
    default_idx = view_options.index(default) if default in view_options else 0

    signal_options = SIGNAL_OPTIONS_ALL if "thermal_inertia" in en else [s for s in SIGNAL_OPTIONS_ALL if "Thermal" not in s]

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        cluster_source = st.selectbox("Select Clustering Partition to Visualize:", options=view_options, index=default_idx)
    with c2:
        signal = st.selectbox(
            "Trajectory Signal inside Clusters:", options=signal_options, index=0, key="cluster_signal_sel"
        )

    st.plotly_chart(
        build_clustered_timeseries_figure(
            res=ctx.res,
            dates_dt=ctx.dates_dt,
            temp_c=ctx.temp_c,
            cluster_source=cluster_source,
            signal_choice=signal,
            show_temp_overlay=ctx.settings.show_temp_overlay,
        ),
        use_container_width=True,
    )
