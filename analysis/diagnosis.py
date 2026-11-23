"""Plain-English diagnosis text for a building and for a single scatterer.

Every ML-related sentence is only produced if the corresponding module is enabled.
"""

from typing import Any

import numpy as np


def _join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " & " + names[-1]


def generate_building_diagnosis(res: dict[str, Any], temp_is_measured: bool) -> dict[str, Any]:
    en = res["enabled_ml"]
    h_span = res["h_range_m"]
    dh_rate = res["dh_theilsen"]
    dh_p = res["dh_ols_p"]
    v_med = res["v_median"]
    v_clean = res["v_clean_median"]
    therm_g = res["therm_grad"]
    temp_r = res["tb_temp_r"]
    span_yr = res["span_years"]
    n_pts = res["n_pts"]

    total_diff_rate = dh_rate * h_span
    cum_diff_mm = total_diff_rate * span_yr

    sig_vertical = (dh_p < 0.05) and (abs(total_diff_rate) >= 0.8)
    strong_thermal = (abs(temp_r) >= 0.60) and (abs(therm_g) >= 0.05)
    whole_bld_moving = abs(v_clean) >= 1.5

    if sig_vertical:
        severity = "warning"
        headline = (
            f"Vertical Differential LOS Movement Detected "
            f"({total_diff_rate:+.1f} mm/yr top-to-bottom divergence)"
        )
    elif strong_thermal and not whole_bld_moving:
        severity = "info"
        headline = (
            f"Reversible Seasonal Thermal Breathing "
            f"(Temperature r = {temp_r:+.2f}; thermal-cleaned drift = {v_clean:+.2f} mm/yr)"
        )
    elif whole_bld_moving:
        severity = "info"
        direction = "subsiding (away from satellite)" if v_clean < 0 else "heaving (toward satellite)"
        headline = (
            f"Uniform Whole-Building LOS Motion: Building is {direction} at {v_clean:+.2f} mm/yr (thermal-adjusted)"
        )
    else:
        severity = "success"
        headline = "Structurally Stable: Minimal permanent LOS drift or vertical divergence"

    bullets: list[str] = []

    def add(title: str, body: str) -> None:
        bullets.append(f"**{len(bullets) + 1}. {title}:** {body}")

    if "random_forest" in en:
        rc, lc = res["regime_counts"], res["lin_counts"]
        nonlin_pct = 100.0 * lc["Non-linear"] / max(n_pts, 1)
        add(
            "Random Forest Trajectory Classification",
            f"Across {n_pts} scatterers, "
            f"**{rc['Stable']}** are `Stable`, **{rc['Sinking']}** are `Sinking`, **{rc['Uplifting']}** are `Uplifting`, "
            f"and **{rc['Unclassified']}** are `Unclassified`. **{lc['Non-linear']} ({nonlin_pct:.0f}%)** show **Non-linear** trajectories "
            f"({'indicating episodic acceleration/slowdown or step changes' if nonlin_pct >= 30 else 'indicating predominantly steady-rate creep'}).",
        )

    if "thermal_inertia" in en:
        mean_lag = float(np.mean(res["effective_lag_epochs"]))
        mean_sens = float(np.mean(res["total_thermal_sens"]))
        temp_src = "measured daily `tavg`" if temp_is_measured else "seasonal temperature"
        add(
            f"Lagged Ridge & {res['gbdt_name']} Thermal Inertia",
            f"Fitting $[t, T_t, T_{{t-1}}, T_{{t-2}}, T_{{t-3}}]$ "
            f"explains **{res['gbdt_r2']*100:.0f}%** of median variance, revealing an average thermal sensitivity of "
            f"**{mean_sens:+.2f} mm/°C** with a thermal lag of **{mean_lag:.1f} acquisitions** (~{mean_lag*11:.0f} days) from {temp_src}. "
            f"Removing this reversible thermal inertia shifts median velocity from **{v_med:+.2f}** to **{v_clean:+.2f} mm/yr**.",
        )

    names: list[str] = []
    parts: list[str] = []
    if "dtw" in en:
        n_dtw = len(np.unique(res["dtw_labels"]))
        names.append("DTW")
        parts.append(f"DTW aligned phase-shifted seasonal waves into **{n_dtw} material/lag cluster(s)**")
    if "hdbscan" in en:
        n_hdb = len([c for c in np.unique(res["hdbscan_labels"]) if c != -1])
        n_hdb_noise = int((res["hdbscan_labels"] == -1).sum())
        names.append("HDBSCAN")
        parts.append(f"density-adaptive HDBSCAN found **{n_hdb} cluster(s)** (+{n_hdb_noise} noise)")
    if "stgnn" in en:
        n_stgnn = len(np.unique(res["stgnn_labels"]))
        names.append("3D ST-GNN")
        parts.append(
            f"2-hop 3D Spatial-Temporal Graph Convolution partitioned the $(E, N, H)$ point cloud into "
            f"**{n_stgnn} contiguous spatial-temporal zone(s)**"
        )
    if parts:
        if len(parts) > 1:
            parts[-1] = "and " + parts[-1]
        add(f"Structural Sub-Zoning ({_join_names(names)})", "; ".join(parts) + ".")

    if "isolation_forest" in en or "autoencoder" in en:
        n_if = int(res["is_if_anomaly"].sum())
        n_ae = int(res["is_ae_anomaly"].sum())
        n_both = int((res["is_if_anomaly"] & res["is_ae_anomaly"]).sum())
        if "isolation_forest" in en and "autoencoder" in en:
            add(
                f"Automatic Dual Anomaly Detection (Isolation Forest & {res['ae_backend']})",
                f"Using parameter-free thresholds (`contamination='auto'` and $\\text{{Median}} + 3.5\\times\\text{{MAD}}$ RMSE), "
                f"Isolation Forest detected **{n_if}** global outliers and the Autoencoder detected **{n_ae}** non-linear shape anomalies "
                f"(**{n_both} point(s)** flagged by both).",
            )
        elif "isolation_forest" in en:
            add(
                "Isolation Forest Anomaly Detection",
                f"With `contamination='auto'`, Isolation Forest detected **{n_if}** global outliers.",
            )
        else:
            add(
                f"Autoencoder Anomaly Detection ({res['ae_backend']})",
                f"Using the parameter-free $\\text{{Median}} + 3.5\\times\\text{{MAD}}$ RMSE threshold, "
                f"the Autoencoder detected **{n_ae}** non-linear shape anomalies.",
            )

    if sig_vertical:
        sense = "faster away/downward" if total_diff_rate < 0 else "faster toward/upward"
        add(
            f"Vertical LOS Divergence ({dh_rate:+.3f} mm/yr/m, p = {dh_p:.3f})",
            f"Over **{h_span:.1f} m** of height, "
            f"upper scatterers move **{abs(total_diff_rate):.1f} mm/yr** {sense} than the base (**{cum_diff_mm:+.1f} mm** cumulative).",
        )

    bullets.append(
        "**Single-Orbit Viewing Note:** Velocities are 1D satellite Line-of-Sight (LOS) projections from one orbit pass. "
        "2D horizontal/rotational tilt decomposition is excluded until both Ascending and Descending tracks are combined."
    )

    return dict(severity=severity, headline=headline, bullets=bullets)


def explain_single_point(pt_idx: int, res: dict[str, Any], decomp: dict[str, np.ndarray]) -> str:
    en = res["enabled_ml"]
    h_val = float(res["h_rel"][pt_idx])
    stratum = str(res["stratum_labels"][pt_idx])
    is_if = bool(res["is_if_anomaly"][pt_idx])
    is_ae = bool(res["is_ae_anomaly"][pt_idx])

    parts = [f"**Point #{pt_idx}** ({h_val:.1f} m above base, {stratum}):"]

    if "random_forest" in en:
        regime = str(res["regime_labels"][pt_idx])
        reg_prob = float(res["regime_max_prob"][pt_idx])
        linearity = str(res["linearity_labels"][pt_idx])
        nonlin_prob = float(res["lin_prob_nonlin"][pt_idx])
        lin_conf = nonlin_prob if linearity == "Non-linear" else (1.0 - nonlin_prob)
        parts.append(
            f"RF classifies its STL trend as **{regime}** ({reg_prob*100:.0f}%) and **{linearity}** ({lin_conf*100:.0f}%)."
        )

    if "thermal_inertia" in en:
        v_clean = float(res["v_thermal_clean"][pt_idx])
        t_sens = float(res["total_thermal_sens"][pt_idx])
        t_lag = float(res["effective_lag_epochs"][pt_idx])
        parts.append(
            f"Raw LOS rate is **{decomp['rate']:+.2f} mm/yr**, while Lagged Ridge thermal-adjusted velocity is **{v_clean:+.2f} mm/yr** "
            f"(thermal sensitivity **{t_sens:+.2f} mm/°C**, lag **{t_lag:.1f} epochs**)."
        )
    else:
        parts.append(f"Raw LOS rate is **{decomp['rate']:+.2f} mm/yr**.")

    assigned = []
    if "stgnn" in en:
        assigned.append(f"**ST-GNN Zone #{int(res['stgnn_labels'][pt_idx])}**")
    if "dtw" in en:
        assigned.append(
            f"**DTW Cluster #{int(res['dtw_labels'][pt_idx])}** (wave shift {int(res['phase_lags'][pt_idx]):+d} epochs)"
        )
    if "hdbscan" in en:
        hdb_c = int(res["hdbscan_labels"][pt_idx])
        assigned.append(f"**HDBSCAN {'Noise (-1)' if hdb_c == -1 else f'Cluster #{hdb_c}'}**")
    if assigned:
        parts.append("Assigned to " + ", ".join(assigned) + ".")

    if is_if and is_ae:
        parts.append("🚨 **Dual ML Anomaly:** Flagged by BOTH Isolation Forest and the 1D-CNN Autoencoder as a strong trajectory outlier.")
    elif is_ae:
        parts.append("⚠️ **Autoencoder Anomaly:** Exceeds automatic $3.5\\times\\text{MAD}$ reconstruction threshold—exhibits a non-linear shape departure from building norms.")
    elif is_if:
        parts.append("⚠️ **Isolation Forest Anomaly:** Global outlier relative to neighboring scatterers on this building.")

    return " ".join(parts)
