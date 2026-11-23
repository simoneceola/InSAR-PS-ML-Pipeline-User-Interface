"""Fast vectorised per-building summary metrics for the portfolio table & map."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from analysis.temporal import TemporalRateFitter, compute_temperature_link


def summarize_building_fast(
    bld_id: int,
    bdata: dict[str, Any],
    dates_dt: pd.DatetimeIndex,
    temp_c: np.ndarray,
    fitter: TemporalRateFitter,
) -> dict[str, Any]:
    disp = np.asarray(bdata["displacement"], dtype=float)
    height = np.asarray(bdata["h_above_dem"], dtype=float)
    lon = np.asarray(bdata["lon"], dtype=float)
    lat = np.asarray(bdata["lat"], dtype=float)

    valid = (height > 0.0) & np.isfinite(height) & np.isfinite(lon) & np.isfinite(lat)
    disp, height, lon, lat = disp[valid], height[valid], lon[valid], lat[valid]
    if np.isnan(disp).any():
        disp = pd.DataFrame(disp).interpolate(axis=1, limit_direction="both").fillna(0.0).to_numpy(dtype=float)

    n_pts = len(height)
    h_min_m = float(np.min(height))
    h_max_m = float(np.max(height))
    h_rel = height - float(np.min(height))
    h_rel = np.maximum(h_rel, 0.0)
    h_range = float(np.ptp(h_rel))
    rates, seasonal_amp = fitter.fit(disp)

    if n_pts >= 3 and h_range > 0:
        ts_slope = float(stats.theilslopes(rates, h_rel)[0])
        ols_h = sm.OLS(rates, sm.add_constant(h_rel)).fit(cov_type="HC1")
        dh_ols_p = float(ols_h.pvalues[1]) if np.isfinite(ols_h.pvalues[1]) else 1.0
        therm_grad = float(stats.theilslopes(seasonal_amp, h_rel)[0]) if seasonal_amp is not None else np.nan
    else:
        ts_slope, dh_ols_p, therm_grad = 0.0, 1.0, 0.0

    q_low, q_high = np.quantile(h_rel, [0.30, 0.70])
    bot_mask, top_mask = h_rel <= q_low, h_rel >= q_high
    n_ref = min(5, disp.shape[1])
    bot_ts = disp[bot_mask].mean(axis=0) - disp[bot_mask, :n_ref].mean()
    top_ts = disp[top_mask].mean(axis=0) - disp[top_mask, :n_ref].mean()
    tb_temp_r = compute_temperature_link(top_ts - bot_ts, temp_c, fitter)["r"]

    row = dict(
        building=bld_id,
        n_pts=n_pts,
        h_min_m=h_min_m,
        h_max_m=h_max_m,
        h_range_m=h_range,
        lat_center=float(np.mean(lat)),
        lon_center=float(np.mean(lon)),
        v_median=float(np.median(rates)),
        v_iqr=float(np.subtract(*np.percentile(rates, [75, 25]))),
        dh_theilsen=ts_slope,
        dh_ols_p=dh_ols_p,
        therm_grad=therm_grad,
        tb_temp_r=tb_temp_r,
    )
    if "max_height_m" in bdata:
        row["max_height_m"] = float(bdata["max_height_m"])
    if "construction_year" in bdata and bdata["construction_year"] is not None:
        row["construction_year"] = bdata["construction_year"]
    return row
