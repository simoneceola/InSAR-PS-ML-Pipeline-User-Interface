"""Time-axis helpers: annual period, STL trend, parametric rate fitter, temperature link."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL


def estimate_annual_period(dates_dt: pd.DatetimeIndex) -> int:
    if len(dates_dt) < 3:
        return 12
    dt_days = float(np.median(np.diff(dates_dt).astype("timedelta64[D]").astype(float)))
    dt_days = max(dt_days, 1.0)
    return max(4, int(round(365.25 / dt_days)))


def extract_stl_trend(ts: np.ndarray, period: int) -> np.ndarray:
    arr = np.asarray(ts, dtype=np.float64)
    if np.isnan(arr).any():
        arr = pd.Series(arr).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float64)
    if len(arr) >= 2 * period:
        return STL(arr, period=period).fit().trend
    win = max(3, min(len(arr) // 3, period))
    return pd.Series(arr).rolling(win, center=True, min_periods=1).mean().to_numpy()


class TemporalRateFitter:
    """Linear (or polynomial) trend + annual harmonic least-squares model."""

    def __init__(self, dates_dt: pd.DatetimeIndex, trend_order: int = 1, seasonal: bool = True):
        self.seasonal = seasonal
        self.t_years = (dates_dt - dates_dt[0]).total_seconds().to_numpy() / (365.25 * 86400.0)
        self.t_centered = self.t_years - self.t_years.mean()
        self.trend_order = trend_order

        columns = [np.ones_like(self.t_centered)]
        columns.extend(self.t_centered ** p for p in range(1, trend_order + 1))
        if seasonal:
            self.sin_term = np.sin(2.0 * np.pi * self.t_years)
            self.cos_term = np.cos(2.0 * np.pi * self.t_years)
            columns.extend([self.sin_term, self.cos_term])

        self.design_matrix = np.column_stack(columns)

    def fit(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        y_2d = np.atleast_2d(np.asarray(y, dtype=float))
        coef, *_ = np.linalg.lstsq(self.design_matrix, y_2d.T, rcond=None)
        rates = coef[1]
        seasonal_amp = np.hypot(coef[-2], coef[-1]) if self.seasonal else None
        return rates, seasonal_amp

    def decompose(self, y_1d: np.ndarray) -> dict[str, np.ndarray]:
        y = np.asarray(y_1d, dtype=float)
        coef, *_ = np.linalg.lstsq(self.design_matrix, y, rcond=None)

        trend_cols = 1 + self.trend_order
        trend_only = self.design_matrix[:, :trend_cols] @ coef[:trend_cols]
        seasonal = (
            self.design_matrix[:, trend_cols:] @ coef[trend_cols:]
            if self.seasonal
            else np.zeros_like(y)
        )
        total_fit = trend_only + seasonal
        residual = y - total_fit
        return dict(
            raw=y,
            total_fit=total_fit,
            trend_only=trend_only,
            seasonal=seasonal,
            residual=residual,
            rate=float(coef[1]),
            seasonal_amp=float(np.hypot(coef[-2], coef[-1])) if self.seasonal else 0.0,
        )

    def deseasonalize(self, y: np.ndarray) -> np.ndarray:
        if not self.seasonal:
            return np.asarray(y, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        is_1d = y_arr.ndim == 1
        y_2d = np.atleast_2d(y_arr)
        coef, *_ = np.linalg.lstsq(self.design_matrix, y_2d.T, rcond=None)
        seasonal_comp = np.outer(coef[-2], self.sin_term) + np.outer(coef[-1], self.cos_term)
        cleaned = y_2d - seasonal_comp
        return cleaned[0] if is_1d else cleaned


def compute_temperature_link(
    series_mm: np.ndarray, temp_c: np.ndarray, fitter: TemporalRateFitter
) -> dict[str, float]:
    decomp = fitter.decompose(series_mm)
    detrended = decomp["raw"] - decomp["trend_only"]
    valid = np.isfinite(detrended) & np.isfinite(temp_c)
    if valid.sum() < 4 or np.std(detrended[valid]) < 1e-9 or np.std(temp_c[valid]) < 1e-9:
        return dict(r=0.0, p_val=1.0, slope_mm_per_c=0.0)

    r_val, p_val = stats.pearsonr(temp_c[valid], detrended[valid])
    slope, *_ = stats.linregress(temp_c[valid], detrended[valid])
    return dict(r=float(r_val), p_val=float(p_val), slope_mm_per_c=float(slope))
