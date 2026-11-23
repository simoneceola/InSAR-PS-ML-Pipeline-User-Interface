"""Synthetic demo workspace, used when the CSV / GeoPackage / export files are not found."""

import numpy as np
import pandas as pd

from analysis.temporal import TemporalRateFitter
from app_config import RD_TO_WGS84
from data_io.summary import summarize_building_fast
from data_io.temperature import load_temperature_series


def generate_synthetic_workspace() -> tuple[dict[int, dict], pd.DataFrame, pd.DatetimeIndex, np.ndarray, bool]:
    rng = np.random.default_rng(42)
    dates_dt = pd.date_range("2022-01-06", "2025-12-25", freq="12D")
    t_yr = (dates_dt - dates_dt[0]).total_seconds().to_numpy() / (365.25 * 86400.0)

    temp_c, temp_is_measured = load_temperature_series(dates_dt)
    if not temp_is_measured:
        temp_c = temp_c + rng.normal(0.0, 1.4, len(dates_dt))

    base_e, base_n = 92_500.0, 437_000.0
    scenarios = [
        (14.0, 8.0, 18.0, -1.8, -0.18, 0.015),
        (16.0, 16.0, 32.0, -0.5, -0.01, 0.028),
        (22.0, 1.2, 21.0, -2.5, -0.24, 0.018),
        (28.0, 18.0, 11.0, -0.3, 0.005, 0.010),
        (20.0, 12.0, 15.0, -3.4, -0.12, 0.016),
        (24.0, 14.0, 2.2, -1.5, 0.03, 0.009),
        (1.5, 19.0, 17.0, -2.1, -0.19, 0.020),
        (26.0, 15.0, 24.0, -1.1, -0.09, 0.022),
    ]

    datasets: dict[int, dict] = {}
    temp_anom = temp_c - temp_c.mean()
    temp_lag2 = np.concatenate([[temp_anom[0], temp_anom[0]], temp_anom[:-2]])
    fitter = TemporalRateFitter(dates_dt, trend_order=1, seasonal=True)
    summary_rows = []

    for idx, (len_e, len_n, max_h, v0, dv_dh, alpha_mm_m_c) in enumerate(scenarios):
        bld_num = idx + 1
        ce = base_e + (idx % 4) * 180.0
        cn = base_n + (idx // 4) * 160.0

        n_pts = 44
        pe = rng.uniform(-len_e / 2, len_e / 2, n_pts)
        pn = rng.uniform(-len_n / 2, len_n / 2, n_pts)
        ph = rng.uniform(0.5, max_h, n_pts)

        point_vel = v0 + dv_dh * ph + rng.normal(0.0, 0.25, n_pts)
        lagged_mask = pe > 0
        thermal_disp = np.where(
            lagged_mask[:, None],
            (alpha_mm_m_c * ph)[:, None] * temp_lag2[None, :],
            (alpha_mm_m_c * ph)[:, None] * temp_anom[None, :],
        )
        noise = rng.normal(0.0, 1.0, size=(n_pts, len(t_yr)))
        disp = point_vel[:, None] * t_yr[None, :] + thermal_disp + noise

        disp[0] += np.where(t_yr > 2.0, -6.5 * (t_yr - 2.0), 0.0)
        disp[1] += np.where(t_yr > 2.5, -5.0, 0.0)

        lon, lat = RD_TO_WGS84.transform(ce + pe, cn + pn)
        he, hn = max(len_e / 2.0, 3.0), max(len_n / 2.0, 3.0)
        corners_e = [ce - he, ce + he, ce + he, ce - he, ce - he]
        corners_n = [cn - hn, cn - hn, cn + hn, cn + hn, cn - hn]
        c_lon, c_lat = RD_TO_WGS84.transform(corners_e, corners_n)
        poly_geojson = {
            "type": "Polygon",
            "coordinates": [[[float(x), float(y)] for x, y in zip(c_lon, c_lat)]],
        }

        bdata = dict(
            displacement=disp,
            h_above_dem=ph,
            lon=np.asarray(lon),
            lat=np.asarray(lat),
            pnt_enscoh=rng.uniform(0.82, 0.99, n_pts),
            polygon_geojson=poly_geojson,
        )
        datasets[bld_num] = bdata
        summary_rows.append(summarize_building_fast(bld_num, bdata, dates_dt, temp_c, fitter))

    df = pd.DataFrame(summary_rows)
    return datasets, df, dates_dt, temp_c, temp_is_measured
