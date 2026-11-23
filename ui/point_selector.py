"""Dropdown to pick one PS scatterer; shared by the 3D, all-series and single-point sections."""

from typing import Any

import numpy as np
import streamlit as st


def _format_point(idx: int, res: dict[str, Any]) -> str:
    en = res["enabled_ml"]
    is_if, is_ae = bool(res["is_if_anomaly"][idx]), bool(res["is_ae_anomaly"][idx])
    flag = "[IF+AE ANOMALY] " if (is_if and is_ae) else ("[ANOMALY] " if (is_if or is_ae) else "")

    bits = []
    if "random_forest" in en:
        bits.append(f"{res['regime_labels'][idx]} / {res['linearity_labels'][idx]}")
    if "stgnn" in en:
        bits.append(f"ST-GNN Zone {res['stgnn_labels'][idx]}")
    if "dtw" in en:
        bits.append(f"DTW {res['dtw_labels'][idx]}")

    middle = ("— " + " | ".join(bits) + " ") if bits else "— "
    return (
        f"Point #{idx} {flag}{middle}"
        f"(H: {res['h_rel'][idx]:.1f}m, Clean: {res['v_thermal_clean'][idx]:+.2f} mm/yr)"
    )


def render_point_selector(res: dict[str, Any]) -> int:
    n_pts = res["n_pts"]
    both_anom = np.where(res["is_if_anomaly"] & res["is_ae_anomaly"])[0]
    any_anom = np.where(res["is_if_anomaly"] | res["is_ae_anomaly"])[0]
    default_pt = (
        int(both_anom[0])
        if len(both_anom) > 0
        else (int(any_anom[0]) if len(any_anom) > 0 else int(np.argmax(res["h_rel"])))
    )
    return int(
        st.selectbox(
            "Highlight & Inspect Individual Scatterer (PS Point):",
            options=list(range(n_pts)),
            index=default_pt,
            format_func=lambda idx: _format_point(idx, res),
        )
    )
