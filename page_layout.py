"""
PAGE LAYOUT — the single file you edit to decide what shows up on the web page.

Three switches:

1. PORTFOLIO   – the top row (2D map + building table).
2. SECTIONS    – everything below it. Set the flag to False to hide a section,
                 or move a line up/down to change the order on the page.
3. ML_MODULES  – which models actually RUN. A disabled model is skipped
                 (saves compute) and every chart/KPI/sidebar control that depends
                 on it disappears automatically.

If every entry in SECTIONS is False, the (expensive) per-building analysis is
skipped entirely and you only get the portfolio map/table.

Note: "stgnn" builds its graph from the autoencoder latent space and the thermal
model outputs. It still runs without them, but its zones are then less informative.
"""

# ------------------------------------------------------------------ 1. top row
PORTFOLIO = {
    "map": True,     # 2D portfolio map
    "table": False,   # clickable buildings table
}

# ------------------------------------------------------------------ 2. page sections (in display order)
SECTIONS = [
    ("diagnosis",            True),   # automated diagnosis banner + explanation + ML guide
    ("kpis",                 True),   # KPI card strip
    ("scatterer_3d",         True),   # 3D point cloud + latent/velocity profile
    ("all_timeseries",       True),   # all scatterers (gray) + building median (red)
    ("clustered_timeseries", True),   # time series split by chosen clustering
    ("height_strata",        True),   # bottom / middle / top strata + top−bottom vs temperature
    ("single_point",         True),   # single scatterer ML diagnostic
]

# ------------------------------------------------------------------ 3. ML modules
ML_MODULES = {
    "random_forest":    True,   # kinematic regime + linearity classifiers
    "dtw":              False,   # DTW phase-lag clustering
    "hdbscan":          False,   # density-adaptive clustering
    "thermal_inertia":  False,   # lagged Ridge + GBDT thermal model
    "autoencoder":      False,   # 1D-CNN / bottleneck autoencoder anomalies
    "isolation_forest": False,   # Isolation Forest anomalies
    "stgnn":            False,   # spatial-temporal graph zones
}
