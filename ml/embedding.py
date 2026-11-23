"""2D latent embedding (PCA / t-SNE / AE latent) + DBSCAN used by the latent-space scatter."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def compute_embedding_and_dbscan(
    disp_scaled: np.ndarray,
    ae_latent_2d: np.ndarray,
    sub_cluster_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_pts = disp_scaled.shape[0]
    if n_pts < 4:
        return np.zeros((n_pts, 2)), np.zeros(n_pts, dtype=int)

    if "t-SNE" in sub_cluster_method and n_pts >= 6:
        perp = min(20, max(2, (n_pts - 1) // 3))
        embedding_2d = TSNE(
            n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto"
        ).fit_transform(disp_scaled)
    elif "Autoencoder" in sub_cluster_method:
        embedding_2d = ae_latent_2d
    else:
        embedding_2d = PCA(n_components=2, random_state=42).fit_transform(disp_scaled)

    emb_std = StandardScaler().fit_transform(embedding_2d)
    dbscan_labels = DBSCAN(eps=2.0, min_samples=min(4, max(2, n_pts // 4))).fit_predict(emb_std)
    return embedding_2d, dbscan_labels
