"""Section: 3D building point cloud + latent space / velocity profile."""

import streamlit as st

from app_context import AppContext
from plots.building_3d import build_3d_building_figure
from plots.latent_profile import build_ml_latent_and_profile_figure


def render(ctx: AppContext) -> None:
    st.subheader("3D Building Map & Sub-Building ML Clustering")
    selected_pt = ctx.selected_point()

    col_3d, col_prof = st.columns([0.56, 0.44], gap="medium")
    with col_3d:
        st.plotly_chart(
            build_3d_building_figure(
                ctx.res,
                color_mode=ctx.settings.color_mode,
                show_stgnn_edges=ctx.settings.show_stgnn_edges,
                highlighted_pt=selected_pt,
            ),
            use_container_width=True,
        )
    with col_prof:
        st.plotly_chart(
            build_ml_latent_and_profile_figure(
                ctx.res,
                sub_cluster_method=ctx.settings.sub_cluster_method,
                highlighted_pt=selected_pt,
            ),
            use_container_width=True,
        )
