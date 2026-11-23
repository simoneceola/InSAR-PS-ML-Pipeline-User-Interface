"""Dynamic Time Warping clustering (phase-lag aware)."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

try:
    from tslearn.metrics import cdist_dtw as tslearn_cdist_dtw
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False


def _sakoe_chiba_dtw_pair(s1: np.ndarray, s2: np.ndarray, radius: int) -> float:
    n = len(s1)
    cost = np.full((n + 1, n + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(n, i + radius)
        diffs = (s1[i - 1] - s2[j_start - 1 : j_end]) ** 2
        for offset, d in enumerate(diffs):
            j = j_start + offset
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(np.sqrt(cost[n, n] / n))


def compute_dtw_clusters(disp_zeroed: np.ndarray, radius: int = 4) -> tuple[np.ndarray, np.ndarray]:
    n_pts, n_epochs = disp_zeroed.shape
    means = disp_zeroed.mean(axis=1, keepdims=True)
    stds = disp_zeroed.std(axis=1, keepdims=True) + 1e-6
    z_series = (disp_zeroed - means) / stds

    med_wave = np.median(z_series, axis=0)
    lags = np.zeros(n_pts, dtype=int)
    max_shift = min(4, n_epochs // 6)
    for i in range(n_pts):
        best_lag, best_corr = 0, -np.inf
        for shift in range(-max_shift, max_shift + 1):
            if shift < 0:
                c = float(np.dot(z_series[i, :shift], med_wave[-shift:]))
            elif shift > 0:
                c = float(np.dot(z_series[i, shift:], med_wave[:-shift]))
            else:
                c = float(np.dot(z_series[i], med_wave))
            if c > best_corr:
                best_corr, best_lag = c, shift
        lags[i] = best_lag

    if n_pts < 4:
        return np.zeros(n_pts, dtype=int), lags

    if HAS_TSLEARN:
        dtw_mat = tslearn_cdist_dtw(
            z_series[:, :, None], global_constraint="sakoe_chiba", sakoe_chiba_radius=radius
        )
    else:
        dtw_mat = np.zeros((n_pts, n_pts), dtype=np.float64)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                d = _sakoe_chiba_dtw_pair(z_series[i], z_series[j], radius=radius)
                dtw_mat[i, j] = dtw_mat[j, i] = d

    condensed = squareform(dtw_mat, checks=False)
    Z = linkage(condensed, method="average")
    max_k = min(5, n_pts - 1)
    best_k, best_sil = 2, -1.0
    for k in range(2, max_k + 1):
        lbls = fcluster(Z, t=k, criterion="maxclust") - 1
        if len(np.unique(lbls)) > 1:
            sil = float(silhouette_score(dtw_mat, lbls, metric="precomputed"))
            if sil > best_sil:
                best_sil, best_k = sil, k

    final_lbls = fcluster(Z, t=best_k, criterion="maxclust") - 1
    return final_lbls, lags


def skipped_dtw(n_pts: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(n_pts, dtype=int), np.zeros(n_pts, dtype=int)
