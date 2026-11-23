"""Spatial-Temporal GNN-style graph propagation and contiguous 3D zone clustering."""

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def skipped_stgnn(n_pts: int) -> dict[str, Any]:
    return dict(
        stgnn_labels=np.zeros(n_pts, dtype=int),
        spatial_discordance=np.zeros(n_pts),
        edge_pairs=[],
    )


def run_stgnn_clustering(
    e_local: np.ndarray,
    n_local: np.ndarray,
    h_rel: np.ndarray,
    node_features: np.ndarray,
    k_neighbors: int = 6,
    n_zones: int = 3,
) -> dict[str, Any]:
    n_pts = len(h_rel)
    if n_pts < 4:
        return skipped_stgnn(n_pts)

    coords_3d = np.column_stack([e_local, n_local, h_rel])
    k = min(k_neighbors, n_pts - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords_3d)
    dists, indices = nbrs.kneighbors(coords_3d)

    sigma = float(np.median(dists[:, 1:])) + 1e-6
    W = np.zeros((n_pts, n_pts), dtype=np.float64)
    edge_pairs: list[tuple[int, int]] = []
    for i in range(n_pts):
        for rank in range(1, k + 1):
            j = int(indices[i, rank])
            w_ij = float(np.exp(-(dists[i, rank] ** 2) / (2.0 * sigma ** 2)))
            W[i, j] = max(W[i, j], w_ij)
            W[j, i] = max(W[j, i], w_ij)
            if i < j:
                edge_pairs.append((i, j))

    A_tilde = W + np.eye(n_pts)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
    A_hat = deg_inv_sqrt @ A_tilde @ deg_inv_sqrt

    X_norm = StandardScaler().fit_transform(node_features)
    H1 = np.tanh(A_hat @ X_norm)
    H2 = np.tanh(A_hat @ H1)

    W_row = W / (W.sum(axis=1, keepdims=True) + 1e-9)
    neighbor_mean = W_row @ X_norm
    spatial_discordance = np.linalg.norm(X_norm - neighbor_mean, axis=1)

    target_zones = min(n_zones, max(2, n_pts // 3))
    conn = (W > 0).astype(int)
    agg = AgglomerativeClustering(n_clusters=target_zones, connectivity=conn, linkage="ward")
    stgnn_labels = agg.fit_predict(H2)

    return dict(
        stgnn_labels=stgnn_labels,
        spatial_discordance=spatial_discordance,
        edge_pairs=edge_pairs,
    )
