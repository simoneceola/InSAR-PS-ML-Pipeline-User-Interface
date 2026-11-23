"""Section: Bottom / Middle / Top strata evolution and Top−Bottom vs. temperature."""

import streamlit as st

from app_context import AppContext
from plots.strata import build_building_timeseries_figure


def render(ctx: AppContext) -> None:
    st.subheader("Height Strata Evolution")
    st.plotly_chart(
        build_building_timeseries_figure(
            ctx.res,
            dates_dt=ctx.dates_dt,
            temp_c=ctx.temp_c,
            ts_display_mode=ctx.settings.ts_display_mode,
            show_temp_overlay=ctx.settings.show_temp_overlay,
            show_season_bands=ctx.settings.show_season_bands,
            baseline_epochs=ctx.cfg.baseline_epochs,
        ),
        use_container_width=True,
    )
