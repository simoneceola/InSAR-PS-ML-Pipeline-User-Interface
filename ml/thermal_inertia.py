"""Lagged Ridge + gradient-boosted regression: thermal inertia and thermal-cleaned settlement."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def fit_lagged_thermal_model(
    disp: np.ndarray,
    dates_dt: pd.DatetimeIndex,
    temp_c: np.ndarray,
    max_lag: int = 3,
) -> dict[str, Any]:
    t_yr = (dates_dt - dates_dt[0]).total_seconds().to_numpy() / (365.25 * 86400.0)
    t_cent = t_yr - t_yr.mean()

    temp_anom = temp_c - np.mean(temp_c)
    lag_cols = [temp_anom]
    for lag in range(1, max_lag + 1):
        shifted = np.concatenate([np.repeat(temp_anom[0], lag), temp_anom[:-lag]])
        lag_cols.append(shifted)
    T_lags = np.column_stack(lag_cols)

    X_design = np.column_stack([t_cent, T_lags])
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 25))
    ridge.fit(X_design, disp.T)

    v_thermal_clean = ridge.coef_[:, 0]
    lag_weights = ridge.coef_[:, 1:]
    total_thermal_sens = lag_weights.sum(axis=1)

    lag_indices = np.arange(max_lag + 1, dtype=float)
    effective_lag_epochs = (np.abs(lag_weights) @ lag_indices) / (np.abs(lag_weights).sum(axis=1) + 1e-6)

    thermal_series = (T_lags @ lag_weights.T).T
    permanent_series = disp - thermal_series

    med_disp = np.median(disp, axis=0)
    if HAS_XGB:
        gbdt = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.06, random_state=42)
        gbdt_name = "XGBoost"
    elif HAS_LGBM:
        gbdt = LGBMRegressor(n_estimators=120, max_depth=3, learning_rate=0.06, random_state=42, verbose=-1)
        gbdt_name = "LightGBM"
    else:
        gbdt = HistGradientBoostingRegressor(max_iter=120, max_depth=3, learning_rate=0.06, random_state=42)
        gbdt_name = "HistGBDT"

    gbdt.fit(X_design, med_disp)
    gbdt_pred_med = gbdt.predict(X_design)
    gbdt_r2 = float(r2_score(med_disp, gbdt_pred_med))

    return dict(
        v_thermal_clean=v_thermal_clean,
        lag_weights=lag_weights,
        total_thermal_sens=total_thermal_sens,
        effective_lag_epochs=effective_lag_epochs,
        thermal_series=thermal_series,
        permanent_series=permanent_series,
        gbdt_pred_med=gbdt_pred_med,
        gbdt_r2=gbdt_r2,
        gbdt_name=gbdt_name,
    )


def skipped_thermal_model(disp: np.ndarray, rates: np.ndarray) -> dict[str, Any]:
    """Neutral output when thermal_inertia is off: 'clean' velocity = raw velocity."""
    n_pts = disp.shape[0]
    return dict(
        v_thermal_clean=np.asarray(rates, dtype=float).copy(),
        lag_weights=np.zeros((n_pts, 1)),
        total_thermal_sens=np.zeros(n_pts),
        effective_lag_epochs=np.zeros(n_pts),
        thermal_series=np.zeros_like(disp),
        permanent_series=disp.copy(),
        gbdt_pred_med=np.median(disp, axis=0),
        gbdt_r2=0.0,
        gbdt_name="n/a",
    )
