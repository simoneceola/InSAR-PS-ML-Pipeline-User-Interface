"""
Global constants: paths, coordinate transformers, colours, label→ML-module
requirements and the AnalysisConfig dataclass.

Nothing in here touches Streamlit, so it is safe to import from anywhere.
"""

from dataclasses import dataclass
from pathlib import Path

from pyproj import Transformer

# ------------------------------------------------------------------ paths
APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = ""C:\Users\samsung\OneDrive - Politecnico di Milano\Desktop\data_06_oct.csv""
DEFAULT_GPKG_PATH = APP_DIR / "rotterdam.gpkg"
HEIGHT_INFO_PATH = APP_DIR / "data_io" / "height_info.geojson"
EXPORT_DIR = APP_DIR / "insar_export"
TEMP_CSV_PATH = APP_DIR / "filtered_date_tavg.csv"

# ------------------------------------------------------------------ coordinate transforms
WGS84_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
RD_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# ------------------------------------------------------------------ labels & colours
REGIME_NAMES = {0: "Stable", 1: "Sinking", 2: "Uplifting", 3: "Unclassified"}
REGIME_COLORS = {
    "Stable": "#2f855a",
    "Sinking": "#c53030",
    "Uplifting": "#2b6cb0",
    "Unclassified": "#a0aec0",
}

LINEARITY_NAMES = {0: "Linear", 1: "Non-linear"}
LINEARITY_COLORS = {"Linear": "#319795", "Non-linear": "#dd6b20"}

CLUSTER_PALETTE = [
    "#3182ce", "#e53e3e", "#38a169", "#d69e2e", "#805ad5",
    "#dd6b20", "#319795", "#d53f8c", "#2b6cb0", "#718096",
]

# ------------------------------------------------------------------ ML module registry
# These keys are what you switch on/off in page_layout.py → ML_MODULES.
ALL_ML_MODULES = (
    "random_forest",     # RF #1 kinematic regime + RF #2 linearity
    "dtw",               # DTW phase-lag clustering
    "hdbscan",           # density-adaptive clustering
    "thermal_inertia",   # lagged Ridge + gradient-boosted thermal model
    "autoencoder",       # 1D-CNN / bottleneck autoencoder + MAD anomalies
    "isolation_forest",  # Isolation Forest anomalies
    "stgnn",             # spatial-temporal graph clustering
)

# Sub-building clustering choices (sidebar) -----------------------------------
SUB_HDBSCAN = "HDBSCAN (Density-Adaptive)"
SUB_DTW = "DTW Clustering (Phase-Lag Aware)"
SUB_STGNN = "ST-GNN 3D Graph-Regularized"
SUB_AE = "Autoencoder Latent + DBSCAN"
SUB_TSNE = "t-SNE + DBSCAN"

# option label -> ML modules it needs (empty tuple = always available;
# several modules = available if ANY of them is enabled)
SUB_CLUSTER_REQUIRES: dict[str, tuple[str, ...]] = {
    SUB_HDBSCAN: ("hdbscan",),
    SUB_DTW: ("dtw",),
    SUB_STGNN: ("stgnn",),
    SUB_AE: ("autoencoder",),
    SUB_TSNE: (),
}

COLOR_MODE_REQUIRES: dict[str, tuple[str, ...]] = {
    "LOS Velocity (mm/yr)": (),
    "Thermal-Adjusted Velocity (Lagged Ridge)": ("thermal_inertia",),
    "RF Kinematic Regime": ("random_forest",),
    "RF Trajectory Linearity": ("random_forest",),
    "DTW Time-Series Cluster (Phase-Lag Aware)": ("dtw",),
    "HDBSCAN Density Cluster": ("hdbscan",),
    "ST-GNN 3D Structural Zone": ("stgnn",),
    "Autoencoder + Isolation Forest Anomalies": ("autoencoder", "isolation_forest"),
    "Thermal Lag (Acquisitions)": ("thermal_inertia",),
    "Autoencoder Reconstruction MSE": ("autoencoder",),
    "Thermal Amplitude (mm)": (),
    "Height Stratum": (),
}

CLUSTER_VIEW_REQUIRES: dict[str, tuple[str, ...]] = {
    SUB_DTW: ("dtw",),
    SUB_HDBSCAN: ("hdbscan",),
    SUB_STGNN: ("stgnn",),
    "RF Kinematic Regime": ("random_forest",),
    "RF Trajectory Linearity": ("random_forest",),
    SUB_AE: ("autoencoder",),
}

SIGNAL_OPTIONS_ALL = [
    "Raw Displacement (Zero-Referenced)",
    "Thermal-Cleaned (Lagged Ridge)",
    "STL Trend (De-seasonalized)",
]


def available_options(requires: dict[str, tuple[str, ...]], enabled_ml: frozenset[str]) -> list[str]:
    """Keep only the dropdown options whose ML module(s) are switched on."""
    return [
        label for label, reqs in requires.items()
        if not reqs or any(m in enabled_ml for m in reqs)
    ]


# ------------------------------------------------------------------ analysis configuration
@dataclass(frozen=True)
class AnalysisConfig:
    min_points: int = 12
    min_height_range_m: float = 3.0
    trend_order: int = 1
    seasonal_nuisance: bool = True
    top_bottom_frac: float = 0.30
    baseline_epochs: int = 5
    rf_conf_threshold: float = 0.70
    sub_cluster_method: str = SUB_HDBSCAN
    hdbscan_min_cluster_size: int = 4
    stgnn_k_neighbors: int = 6
    stgnn_n_zones: int = 3
    thermal_max_lag: int = 3
    enabled_ml: frozenset = frozenset(ALL_ML_MODULES)

    def uses(self, module: str) -> bool:
        return module in self.enabled_ml
