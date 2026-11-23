"""Section: KPI card strip. Cards that depend on a disabled ML module are dropped."""

import streamlit as st

from app_context import AppContext


def render(ctx: AppContext) -> None:
    res = ctx.res
    en = res["enabled_ml"]
    rc, lc = res["regime_counts"], res["lin_counts"]

    cards: list[tuple[str, str, str]] = []  # (label, value, delta)

    if "random_forest" in en:
        cards.append((
            "RF Dominant Regime",
            max(rc, key=rc.get),
            f"S:{rc['Stable']} | Sink:{rc['Sinking']} | Up:{rc['Uplifting']}",
        ))
        cards.append((
            "RF Non-Linear Points",
            f"{lc['Non-linear']} / {res['n_pts']}",
            f"{100.0 * lc['Non-linear'] / max(res['n_pts'], 1):.0f}% non-linear",
        ))
    if "isolation_forest" in en or "autoencoder" in en:
        cards.append((
            "Auto Outliers (IF / AE)",
            f"{int(res['is_if_anomaly'].sum())} / {int(res['is_ae_anomaly'].sum())}",
            "IsoForest / Autoencoder",
        ))

    cards.append((
        "Clean LOS Velocity",
        f"{res['v_clean_median']:+.2f} mm/yr",
        f"Raw median: {res['v_median']:+.2f} mm/yr",
    ))
    total_diff_yr = res["dh_theilsen"] * res["h_range_m"]
    cards.append((
        "Vertical LOS Gradient",
        f"{res['dh_theilsen']:+.3f} mm/yr/m",
        f"{total_diff_yr:+.1f} mm/yr span (p={res['dh_ols_p']:.3f})",
    ))

    if "thermal_inertia" in en:
        cards.append((
            "Thermal Lag / Sens",
            f"{res['effective_lag_epochs'].mean():.1f} epochs",
            f"{res['total_thermal_sens'].mean():+.2f} mm/°C ({res['gbdt_name']} R²={res['gbdt_r2']:.2f})",
        ))

    for col, (label, value, delta) in zip(st.columns(len(cards)), cards):
        col.metric(label, value, delta=delta, delta_color="off")
