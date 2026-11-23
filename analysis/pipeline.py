"""
Single-building analysis pipeline.

Runs the physical analysis plus every ML module listed in `cfg.enabled_ml`.
Disabled modules are skipped and replaced by neutral outputs so downstream
code (plots, KPIs) never breaks.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from analysis.temporal import TemporalRateFitter, extract_stl_trend
from analysis.vertical import height_strata, vertical_gradient_stats
from app_config import LINEARITY_NAMES, REGIME_NAMES, WGS84_TO_RD, AnalysisConfig
from ml.autoencoder import run_autoencoder_anomaly, skipped_autoencoder
from ml.dtw_clustering import compute_dtw_clusters, skipped_dtw
from ml.embedding import compute_embedding_and_dbscan
from ml.hdbscan_clustering import compute_hdbscan_clusters, skipped_hdbscan
from ml.isolation_forest import detect_isolation_forest_outliers, skipped_isolation_forest
from ml.random_forest import classify_trajectories, skipped_classification
from ml.stgnn import run_stgnn_clustering, skipped_stgnn
from ml.thermal_inertia import fit_lagged_thermal_model, skipped_thermal_model


def _clean_inputs(bld_data: dict[str, Any]):
    disp = np.asarray(bld_data["displacement"], dtype=float)
    height = np.asarray(bld_data["h_above_dem"], dtype=float)
    lon = np.asarray(bld_data["lon"], dtype=float)
    lat = np.asarray(bld_data["lat"], dtype=float)

    coh = None
    for coh_key in ("pnt_enscoh", "coherence", "enscoh"):
        if coh_key in bld_data and bld_data[coh_key] is not None:
            coh = np.asarray(bld_data[coh_key], dtype=float)
            break

    valid_pt = (height > 0.0) & np.isfinite(height) & np.isfinite(lon) & np.isfinite(lat)
    disp, height, lon, lat = disp[valid_pt], height[valid_pt], lon[valid_pt], lat[valid_pt]
    if coh is not None:
        coh = coh[valid_pt]

    if np.isnan(disp).any():
        disp = (
            pd.DataFrame(disp)
            .interpolate(axis=1, limit_direction="both")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    return disp, height, lon, lat, coh


def analyse_single_building(
    bld_id: int,
    bld_data: dict[str, Any],
    dates_dt: pd.DatetimeIndex,
    temp_c: np.ndarray,
    cfg: AnalysisConfig,
    period: int,
    rf_regime: RandomForestClassifier | None = None,
    rf_linearity: RandomForestClassifier | None = None,
) -> dict[str, Any]:
    disp, height, lon, lat, coh = _clean_inputs(bld_data)

    n_pts = len(height)
    x_rd, y_rd = WGS84_TO_RD.transform(lon, lat)
    e_local = np.asarray(x_rd, dtype=float) - np.mean(x_rd)
    n_local = np.asarray(y_rd, dtype=float) - np.mean(y_rd)
    # The 3D view and height analyses use a building-relative height:
    # the lowest retained PS is the building base (0 m).
    h_rel = height - float(np.min(height))
    h_rel = np.maximum(h_rel, 0.0)
    h_range = float(np.ptp(h_rel))

    # ---- physical baseline: linear rate + annual harmonic
    fitter = TemporalRateFitter(dates_dt, trend_order=cfg.trend_order, seasonal=cfg.seasonal_nuisance)
    rates, seasonal_amp = fitter.fit(disp)
    stl_trends = np.array([extract_stl_trend(d, period=period) for d in disp])

    # ---- zero-reference & standardise trajectories
    n_ref = min(cfg.baseline_epochs, disp.shape[1])
    disp_zeroed = disp - disp[:, :n_ref].mean(axis=1, keepdims=True)
    stl_trends_zeroed = stl_trends - stl_trends[:, :n_ref].mean(axis=1, keepdims=True)
    scaler = StandardScaler()
    disp_scaled = scaler.fit_transform(disp_zeroed)

    # ---- 1 & 2. Random Forest classifiers
    if cfg.uses("random_forest") and rf_regime is not None and rf_linearity is not None:
        rf_out = classify_trajectories(stl_trends, rf_regime, rf_linearity, cfg.rf_conf_threshold)
    else:
        rf_out = skipped_classification(n_pts)

    # ---- 3. DTW
    dtw_labels, phase_lags = (
        compute_dtw_clusters(disp_zeroed, radius=4) if cfg.uses("dtw") else skipped_dtw(n_pts)
    )

    # ---- 4. HDBSCAN
    hdbscan_labels = (
        compute_hdbscan_clusters(disp_scaled, min_cluster_size=cfg.hdbscan_min_cluster_size)
        if cfg.uses("hdbscan")
        else skipped_hdbscan(n_pts)
    )

    # ---- 5. Lagged Ridge + GBDT thermal inertia
    if cfg.uses("thermal_inertia"):
        thermal = fit_lagged_thermal_model(disp_zeroed, dates_dt, temp_c, max_lag=cfg.thermal_max_lag)
    else:
        thermal = skipped_thermal_model(disp_zeroed, rates)
    v_thermal_clean = thermal["v_thermal_clean"]
    permanent_series = thermal["permanent_series"]
    permanent_zeroed = permanent_series - permanent_series[:, :n_ref].mean(axis=1, keepdims=True)

    # ---- 6. Autoencoder
    ae_out = (
        run_autoencoder_anomaly(disp_scaled, coherence=coh)
        if cfg.uses("autoencoder")
        else skipped_autoencoder(disp_scaled)
    )
    recon_denorm = scaler.inverse_transform(ae_out["recon_scaled"])

    # ---- 7. Isolation Forest
    is_if_anomaly = (
        detect_isolation_forest_outliers(disp_zeroed)
        if cfg.uses("isolation_forest")
        else skipped_isolation_forest(n_pts)
    )

    # ---- 8. ST-GNN
    if cfg.uses("stgnn"):
        gnn_node_feats = np.column_stack([
            ae_out["ae_latent_2d"],
            v_thermal_clean,
            thermal["total_thermal_sens"],
            thermal["effective_lag_epochs"],
        ])
        stgnn_out = run_stgnn_clustering(
            e_local=e_local, n_local=n_local, h_rel=h_rel, node_features=gnn_node_feats,
            k_neighbors=cfg.stgnn_k_neighbors, n_zones=cfg.stgnn_n_zones,
        )
    else:
        stgnn_out = skipped_stgnn(n_pts)

    # ---- 2D embedding + DBSCAN for the latent scatter
    embedding_2d, dbscan_labels = compute_embedding_and_dbscan(
        disp_scaled, ae_out["ae_latent_2d"], cfg.sub_cluster_method
    )

    # ---- physical vertical profile & height strata
    vertical = vertical_gradient_stats(rates, v_thermal_clean, h_rel, seasonal_amp)
    strata = height_strata(h_rel, disp, n_ref, cfg.top_bottom_frac, temp_c, fitter)

    regime_labels = rf_out["regime_labels"]
    linearity_labels = rf_out["linearity_labels"]
    regime_counts = {name: int((regime_labels == name).sum()) for name in REGIME_NAMES.values()}
    lin_counts = {name: int((linearity_labels == name).sum()) for name in LINEARITY_NAMES.values()}

    return dict(
        building=bld_id,
        enabled_ml=frozenset(cfg.enabled_ml),
        n_pts=n_pts,
        h_min_m=float(np.min(height)),
        h_max_m=float(np.max(height)),
        h_range_m=h_range,
        lat_center=float(lat.mean()),
        lon_center=float(lon.mean()),
        lat=lat,
        lon=lon,
        disp=disp,
        disp_zeroed=disp_zeroed,
        stl_trends=stl_trends,
        stl_trends_zeroed=stl_trends_zeroed,
        permanent_zeroed=permanent_zeroed,
        recon_denorm=recon_denorm,
        e_local=e_local,
        n_local=n_local,
        h_rel=h_rel,
        rates=rates,
        v_thermal_clean=v_thermal_clean,
        seasonal_amp=seasonal_amp,
        regime_labels=regime_labels,
        regime_max_prob=rf_out["regime_max_prob"],
        regime_counts=regime_counts,
        linearity_labels=linearity_labels,
        lin_prob_nonlin=rf_out["lin_prob_nonlin"],
        lin_counts=lin_counts,
        dtw_labels=dtw_labels,
        phase_lags=phase_lags,
        hdbscan_labels=hdbscan_labels,
        dbscan_labels=dbscan_labels,
        stgnn_labels=stgnn_out["stgnn_labels"],
        spatial_discordance=stgnn_out["spatial_discordance"],
        edge_pairs=stgnn_out["edge_pairs"],
        is_if_anomaly=is_if_anomaly,
        is_ae_anomaly=ae_out["is_ae_anomaly"],
        ae_recon_mse=ae_out["recon_mse"],
        ae_latent_2d=ae_out["ae_latent_2d"],
        ae_backend=ae_out["backend"],
        embedding_2d=embedding_2d,
        total_thermal_sens=thermal["total_thermal_sens"],
        effective_lag_epochs=thermal["effective_lag_epochs"],
        gbdt_pred_med=thermal["gbdt_pred_med"],
        gbdt_r2=thermal["gbdt_r2"],
        gbdt_name=thermal["gbdt_name"],
        v_median=float(np.median(rates)),
        v_clean_median=float(np.median(v_thermal_clean)),
        v_iqr=float(np.subtract(*np.percentile(rates, [75, 25]))),
        span_years=float(fitter.t_years[-1] - fitter.t_years[0]),
        fitter=fitter,
        **vertical,
        **strata,
    )
