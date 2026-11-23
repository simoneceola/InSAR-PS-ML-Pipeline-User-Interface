"""Isolation Forest outlier detection (contamination='auto')."""

import numpy as np
from sklearn.ensemble import IsolationForest


def detect_isolation_forest_outliers(disp_zeroed: np.ndarray) -> np.ndarray:
    n_pts = disp_zeroed.shape[0]
    if n_pts >= 5:
        iso = IsolationForest(contamination="auto", random_state=42)
        return iso.fit_predict(disp_zeroed) == -1
    return np.zeros(n_pts, dtype=bool)


def skipped_isolation_forest(n_pts: int) -> np.ndarray:
    return np.zeros(n_pts, dtype=bool)
