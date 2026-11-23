"""HDBSCAN density-adaptive clustering on PCA-compressed trajectories."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
    HAS_SKLEARN_HDBSCAN = True
except ImportError:
    HAS_SKLEARN_HDBSCAN = False


def compute_hdbscan_clusters(disp_scaled: np.ndarray, min_cluster_size: int = 4) -> np.ndarray:
    n_pts = disp_scaled.shape[0]
    if n_pts < max(5, min_cluster_size + 1):
        return np.zeros(n_pts, dtype=int)

    n_comp = min(5, n_pts - 1, disp_scaled.shape[1])
    pca_feats = PCA(n_components=n_comp, random_state=42).fit_transform(disp_scaled)
    mcs = min(min_cluster_size, max(2, n_pts // 4))

    if HAS_SKLEARN_HDBSCAN:
        return SklearnHDBSCAN(min_cluster_size=mcs, min_samples=max(2, mcs - 1)).fit_predict(pca_feats)
    return DBSCAN(eps=2.0, min_samples=mcs).fit_predict(StandardScaler().fit_transform(pca_feats))


def skipped_hdbscan(n_pts: int) -> np.ndarray:
    return np.zeros(n_pts, dtype=int)
