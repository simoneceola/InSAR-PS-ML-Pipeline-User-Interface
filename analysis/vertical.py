"""Physical (non-ML) building statistics: vertical LOS gradient and height strata."""

from typing import Any

import numpy as np
from scipy import stats
import statsmodels.api as sm

from analysis.temporal import TemporalRateFitter, compute_temperature_link


def vertical_gradient_stats(
    rates: np.ndarray,
    v_thermal_clean: np.ndarray,
    h_rel: np.ndarray,
    seasonal_amp: np.ndarray | None,
) -> dict[str, float]:
    """Theil-Sen / Spearman / HC3-OLS statistics of LOS rate versus height."""
    n_pts = len(h_rel)
    if n_pts >= 3 and np.ptp(h_rel) > 0:
        ts_slope, ts_intercept, ts_lo, ts_hi = stats.theilslopes(rates, h_rel)
        ts_clean_slope, ts_clean_intercept, _, _ = stats.theilslopes(v_thermal_clean, h_rel)
        rho, sp_p = stats.spearmanr(h_rel, rates)
        ols_h = sm.OLS(rates, sm.add_constant(h_rel)).fit(cov_type="HC3")
        if not np.all(np.isfinite(ols_h.pvalues)):
            ols_h = sm.OLS(rates, sm.add_constant(h_rel)).fit(cov_type="HC1")
        dh_ols_p = float(ols_h.pvalues[1])
        therm_grad = (
            float(stats.theilslopes(seasonal_amp, h_rel)[0])
            if seasonal_amp is not None
            else np.nan
        )
    else:
        ts_slope, ts_intercept, ts_lo, ts_hi = 0.0, float(np.median(rates)), 0.0, 0.0
        ts_clean_slope, ts_clean_intercept = 0.0, float(np.median(v_thermal_clean))
        rho, sp_p, dh_ols_p, therm_grad = 0.0, 1.0, 1.0, 0.0

    return dict(
        dh_theilsen=float(ts_slope),
        dh_intercept=float(ts_intercept),
        dh_clean_theilsen=float(ts_clean_slope),
        dh_clean_intercept=float(ts_clean_intercept),
        dh_ci_lo=float(ts_lo),
        dh_ci_hi=float(ts_hi),
        dh_ols_p=dh_ols_p,
        dh_spearman_rho=float(rho),
        dh_spearman_p=float(sp_p),
        therm_grad=therm_grad,
    )


def height_strata(
    h_rel: np.ndarray,
    disp: np.ndarray,
    n_ref: int,
    top_bottom_frac: float,
    temp_c: np.ndarray,
    fitter: TemporalRateFitter,
) -> dict[str, Any]:
    """Split scatterers into Bottom / Middle / Top and build the strata time series."""
    n_pts = len(h_rel)
    q_low, q_high = np.quantile(h_rel, [top_bottom_frac, 1.0 - top_bottom_frac])
    bot_mask = h_rel <= q_low
    top_mask = h_rel >= q_high
    mid_mask = ~(bot_mask | top_mask)

    stratum_labels = np.full(n_pts, "Middle", dtype=object)
    stratum_labels[bot_mask] = "Bottom"
    stratum_labels[top_mask] = "Top"

    def _zero_ref(arr_2d: np.ndarray) -> np.ndarray:
        mean_ts = arr_2d.mean(axis=0)
        return mean_ts - mean_ts[:n_ref].mean()

    bot_ts = _zero_ref(disp[bot_mask])
    top_ts = _zero_ref(disp[top_mask])
    mid_ts = _zero_ref(disp[mid_mask]) if np.any(mid_mask) else None
    tb_ts = top_ts - bot_ts

    tb_link = compute_temperature_link(tb_ts, temp_c, fitter)
    return dict(
        stratum_labels=stratum_labels,
        bot_ts=bot_ts,
        mid_ts=mid_ts,
        top_ts=top_ts,
        tb_ts=tb_ts,
        n_bot=int(bot_mask.sum()),
        n_mid=int(mid_mask.sum()),
        n_top=int(top_mask.sum()),
        h_span_strata=float(np.mean(h_rel[top_mask]) - np.mean(h_rel[bot_mask])),
        tb_temp_r=tb_link["r"],
        tb_temp_p=tb_link["p_val"],
        tb_temp_slope=tb_link["slope_mm_per_c"],
    )
