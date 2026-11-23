"""Section: single-scatterer ML diagnostic."""

import streamlit as st

from analysis.diagnosis import explain_single_point
from analysis.temporal import compute_temperature_link
from app_context import AppContext
from plots.single_point import build_single_point_figure


def render(ctx: AppContext) -> None:
    st.subheader("Single-Point ML Diagnostic")
    res = ctx.res
    pt = ctx.selected_point()
    n_ref = min(ctx.cfg.baseline_epochs, len(ctx.dates_dt))

    pt_raw_zeroed = res["disp_zeroed"][pt]
    pt_stl_zeroed = res["stl_trends_zeroed"][pt]
    pt_perm_zeroed = res["permanent_zeroed"][pt]
    pt_ae_zeroed = res["recon_denorm"][pt] - res["recon_denorm"][pt, :n_ref].mean()
    pt_decomp = res["fitter"].decompose(pt_raw_zeroed)
    pt_temp_link = compute_temperature_link(pt_raw_zeroed, ctx.temp_c, res["fitter"])

    explanation = explain_single_point(pt, res, pt_decomp)
    if res["is_if_anomaly"][pt] or res["is_ae_anomaly"][pt]:
        st.warning(explanation, icon="🚨")
    else:
        st.info(explanation, icon="📍")

    st.plotly_chart(
        build_single_point_figure(
            pt_idx=pt,
            dates_dt=ctx.dates_dt,
            temp_c=ctx.temp_c,
            stl_trend_zeroed=pt_stl_zeroed,
            permanent_zeroed=pt_perm_zeroed,
            ae_recon_zeroed=pt_ae_zeroed,
            decomp=pt_decomp,
            temp_link=pt_temp_link,
            show_season_bands=ctx.settings.show_season_bands,
            show_thermal_clean="thermal_inertia" in res["enabled_ml"],
            show_ae_recon="autoencoder" in res["enabled_ml"],
        ),
        use_container_width=True,
    )
