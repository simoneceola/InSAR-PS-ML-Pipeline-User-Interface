"""Random Forest #1 (kinematic regime) and #2 (linear vs non-linear trajectory)."""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import streamlit as st

from analysis.temporal import extract_stl_trend
from analysis.trajectory_features import extract_linearity_features, extract_regime_features
from app_config import LINEARITY_NAMES, REGIME_NAMES


def _train_classifier_with_splits(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=42,
        stratify=y_remaining,
    )

    classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    classifier.fit(X_train, y_train)

    validation_predictions = classifier.predict(X_validation)
    validation_metrics = {
        "validation_accuracy": float(accuracy_score(y_validation, validation_predictions)),
        "validation_macro_f1": float(
            f1_score(y_validation, validation_predictions, average="macro")
        ),
    }

    classifier.fit(
        np.concatenate((X_train, X_validation)),
        np.concatenate((y_train, y_validation)),
    )
    test_predictions = classifier.predict(X_test)
    test_metrics = {
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_macro_f1": float(f1_score(y_test, test_predictions, average="macro")),
    }

    return classifier, validation_metrics | test_metrics


@st.cache_resource(show_spinner="Training Random Forest trajectory classifiers...")
def _train_random_forest_models_cached(
    n_steps: int,
    period: int,
) -> tuple[
    RandomForestClassifier,
    RandomForestClassifier,
    dict[str, dict[str, float]],
]:
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, n_steps)
    n_per_class = 450
    stable_limit_mm = 2.0
    max_change_mm = 25.0
    min_nonlin_dev_mm = 2.0

    def _make_series(curve: np.ndarray) -> np.ndarray:
        """Create a synthetic time series with the given curve, offset, seasonal component, and noise.
        """
        offset = rng.uniform(-8.0, 8.0)
        seasonal = rng.uniform(0.0, 4.0) * np.sin(
            2.0 * np.pi * np.arange(n_steps) / period + rng.uniform(0.0, 2.0 * np.pi)
        )
        noise = rng.normal(0.0, rng.uniform(0.5, 2.5), size=n_steps)
        return extract_stl_trend(offset + curve + seasonal + noise, period=period)

    def _random_shape(u: np.ndarray) -> np.ndarray:
        """Generate a random shape function.
        """
        kind = rng.integers(0, 4)
        if kind == 0:
            return u ** rng.uniform(0.3, 3.5)
        if kind == 1:
            r = rng.uniform(2.5, 5.5)
            return (np.exp(r * u) - 1.0) / (np.exp(r) - 1.0)
        if kind == 2:
            r = rng.uniform(2.5, 5.5)
            return (1.0 - np.exp(-r * u)) / (1.0 - np.exp(-r))
        c, k = rng.uniform(0.2, 0.8), rng.uniform(8.0, 25.0)
        sig = lambda v: 1.0 / (1.0 + np.exp(-k * (v - c)))
        return (sig(u) - sig(0.0)) / (sig(1.0) - sig(0.0))

    def _big() -> float:
        return float(rng.uniform(stable_limit_mm * 1.1, max_change_mm))

    synth_regime = np.array(
        [_make_series(rng.uniform(-stable_limit_mm, stable_limit_mm) * _random_shape(t)) for _ in range(n_per_class)]
        + [_make_series(-_big() * _random_shape(t)) for _ in range(n_per_class)]
        + [_make_series(_big() * _random_shape(t)) for _ in range(n_per_class)]
    )
    y_regime = np.repeat([0, 1, 2], n_per_class)
    X_regime = np.array([extract_regime_features(ts) for ts in synth_regime])
    rf_regime, regime_metrics = _train_classifier_with_splits(X_regime, y_regime)

    def _nonlinear_curve() -> np.ndarray:
        """Generate a random non-linear curve with a given max change and min deviation from linearity.
        """
        kind = rng.integers(0, 5)
        if kind == 0:
            n_segs = rng.integers(3, 6)
            bps = np.sort(rng.uniform(0.12, 0.88, size=n_segs - 1))
            while np.any(np.diff(np.concatenate([[0.0], bps, [1.0]])) < 0.10):
                bps = np.sort(rng.uniform(0.12, 0.88, size=n_segs - 1))
            knots = np.concatenate([[0.0], bps, [1.0]])
            seg_changes = rng.uniform(-15.0, 15.0, size=n_segs)
            flat_mask = rng.random(n_segs) < 0.30
            seg_changes[flat_mask] = rng.uniform(-0.8, 0.8, size=flat_mask.sum())
            y_knots = np.concatenate([[0.0], np.cumsum(seg_changes)])
            curve = np.interp(t, knots, y_knots)
        elif kind == 1:
            cycles = rng.uniform(1.0, 2.2)
            amp = rng.uniform(2.5, 10.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            drift = rng.uniform(-12.0, 12.0) * t
            curve = drift + amp * (np.sin(2.0 * np.pi * cycles * t + phase) - np.sin(phase))
        elif kind == 2:
            b = rng.uniform(0.20, 0.80)
            c1 = rng.choice([-1.0, 1.0]) * rng.uniform(3.0, max_change_mm)
            c2 = rng.choice([-1.0, 1.0]) * rng.uniform(3.0, max_change_mm)
            leg1 = c1 * np.clip(t / b, 0.0, 1.0)
            leg2 = c1 + c2 * np.clip((t - b) / (1.0 - b), 0.0, 1.0)
            curve = np.where(t <= b, leg1, leg2)
        elif kind == 3:
            c1 = rng.choice([-1.0, 1.0]) * rng.uniform(4.0, max_change_mm)
            if rng.random() < 0.5:
                p = rng.uniform(2.2, 4.0) if rng.random() < 0.5 else rng.uniform(0.25, 0.45)
                curve = c1 * (t ** p)
            else:
                r = rng.uniform(3.0, 6.0)
                if rng.random() < 0.5:
                    curve = c1 * (np.exp(r * t) - 1.0) / (np.exp(r) - 1.0)
                else:
                    curve = c1 * (1.0 - np.exp(-r * t)) / (1.0 - np.exp(-r))
        else:
            c1 = rng.choice([-1.0, 1.0]) * rng.uniform(3.5, max_change_mm)
            b1, k1 = rng.uniform(0.2, 0.8), rng.uniform(10.0, 25.0)
            sig1 = 1.0 / (1.0 + np.exp(-k1 * (t - b1)))
            curve = c1 * (sig1 - sig1[0]) / (sig1[-1] - sig1[0])

        lin_fit = np.polyval(np.polyfit(t, curve, 1), t)
        dev = float(np.max(np.abs(curve - lin_fit)))
        if dev < min_nonlin_dev_mm:
            target_dev = rng.uniform(min_nonlin_dev_mm, 9.0)
            curve = lin_fit + (curve - lin_fit) * (target_dev / (dev + 1e-6))
        return curve

    synth_lin = np.array(
        [_make_series(rng.uniform(-max_change_mm, max_change_mm) * t) for _ in range(n_per_class)]
        + [_make_series(_nonlinear_curve()) for _ in range(n_per_class)]
    )
    y_lin = np.repeat([0, 1], n_per_class)
    X_lin = np.array([extract_linearity_features(ts) for ts in synth_lin])
    rf_linearity, linearity_metrics = _train_classifier_with_splits(X_lin, y_lin)

    metrics = {
        "regime": regime_metrics,
        "linearity": linearity_metrics,
    }
    return rf_regime, rf_linearity, metrics


def train_random_forest_models(
    n_steps: int,
    period: int,
) -> tuple[RandomForestClassifier, RandomForestClassifier]:
    """Train and return the regime and linearity Random Forest classifiers.
    """
    rf_regime, rf_linearity, _ = _train_random_forest_models_cached(n_steps, period)
    return rf_regime, rf_linearity


def get_random_forest_split_metrics(
    n_steps: int,
    period: int,
) -> dict[str, dict[str, float]]:
    """Return cached validation and test metrics for both classifiers.
    """
    _, _, metrics = _train_random_forest_models_cached(n_steps, period)
    return {name: values.copy() for name, values in metrics.items()}


def classify_trajectories(
    stl_trends: np.ndarray,
    rf_regime: RandomForestClassifier,
    rf_linearity: RandomForestClassifier,
    conf_threshold: float,
) -> dict[str, Any]:
    """Classify each trajectory into a kinematic regime and linearity class.
    """
    n_pts = len(stl_trends)

    X_reg = np.array([extract_regime_features(tr) for tr in stl_trends])
    reg_proba = rf_regime.predict_proba(X_reg)
    reg_raw_idx = rf_regime.classes_[reg_proba.argmax(axis=1)]
    reg_max_prob = reg_proba.max(axis=1)
    reg_idx = np.where(reg_max_prob < conf_threshold, 3, reg_raw_idx)
    regime_labels = np.array([REGIME_NAMES[int(i)] for i in reg_idx], dtype=object)

    X_lin = np.array([extract_linearity_features(tr) for tr in stl_trends])
    lin_proba = rf_linearity.predict_proba(X_lin)
    lin_idx = rf_linearity.classes_[lin_proba.argmax(axis=1)]
    lin_prob_nonlin = lin_proba[:, 1] if lin_proba.shape[1] > 1 else np.zeros(n_pts)
    linearity_labels = np.array([LINEARITY_NAMES[int(i)] for i in lin_idx], dtype=object)

    return dict(
        regime_labels=regime_labels,
        regime_max_prob=reg_max_prob,
        linearity_labels=linearity_labels,
        lin_prob_nonlin=lin_prob_nonlin,
    )


def skipped_classification(n_pts: int) -> dict[str, Any]:
    """Neutral output used when the random_forest module is switched off.
    """
    return dict(
        regime_labels=np.full(n_pts, "Unclassified", dtype=object),
        regime_max_prob=np.zeros(n_pts),
        linearity_labels=np.full(n_pts, "Linear", dtype=object),
        lin_prob_nonlin=np.zeros(n_pts),
    )
