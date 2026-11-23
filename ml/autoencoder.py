"""1D-CNN / bottleneck autoencoder with coherence weighting and automatic Median + 3.5×MAD anomalies."""

from typing import Any

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _trivial_output(disp_scaled: np.ndarray, backend: str) -> dict[str, Any]:
    n_pts = disp_scaled.shape[0]
    return dict(
        recon_scaled=disp_scaled.copy(),
        recon_mse=np.zeros(n_pts),
        is_ae_anomaly=np.zeros(n_pts, dtype=bool),
        ae_latent_2d=np.zeros((n_pts, 2)),
        backend=backend,
    )


def skipped_autoencoder(disp_scaled: np.ndarray) -> dict[str, Any]:
    return _trivial_output(disp_scaled, "Disabled")


def run_autoencoder_anomaly(
    disp_scaled: np.ndarray,
    coherence: np.ndarray | None,
) -> dict[str, Any]:
    n_pts, n_epochs = disp_scaled.shape
    weights = (
        np.clip(coherence, 0.1, 1.0)
        if (coherence is not None and len(coherence) == n_pts and np.isfinite(coherence).all())
        else np.ones(n_pts, dtype=float)
    )
    weights = weights / weights.mean()

    if n_pts < 5 or n_epochs < 8:
        return _trivial_output(disp_scaled, "Fallback (Insufficient points)")

    if HAS_TORCH:
        torch.manual_seed(42)

        class Conv1DAutoencoder(nn.Module):
            def __init__(self, seq_len: int):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(8, 4, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.dec = nn.Sequential(
                    nn.ConvTranspose1d(4, 8, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.Conv1d(8, 1, kernel_size=5, padding=2),
                )
                self.seq_len = seq_len

            def forward(self, x: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
                z = self.enc(x)
                out = self.dec(z)
                if out.shape[-1] < self.seq_len:
                    out = nn.functional.pad(out, (0, self.seq_len - out.shape[-1]))
                elif out.shape[-1] > self.seq_len:
                    out = out[..., : self.seq_len]
                return out, z.flatten(start_dim=1)

        x_tensor = torch.tensor(disp_scaled[:, None, :], dtype=torch.float32)
        w_tensor = torch.tensor(weights[:, None, None], dtype=torch.float32)

        model = Conv1DAutoencoder(n_epochs)
        opt = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=1e-4)
        model.train()
        for _ in range(55):
            opt.zero_grad()
            recon, _ = model(x_tensor)
            loss = (w_tensor * (recon - x_tensor) ** 2).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            recon_t, latent_t = model(x_tensor)
            recon_scaled = recon_t[:, 0, :].numpy()
            latent_raw = latent_t.numpy()

        backend = "PyTorch 1D-CNN Autoencoder"
    else:
        bottleneck_dim = min(6, max(2, n_pts // 3))
        mlp = MLPRegressor(
            hidden_layer_sizes=(24, bottleneck_dim, 24),
            activation="tanh",
            max_iter=250,
            random_state=42,
        )
        reps = np.clip(np.round(weights * 2).astype(int), 1, 4)
        X_train = np.repeat(disp_scaled, reps, axis=0)
        mlp.fit(X_train, X_train)
        recon_scaled = mlp.predict(disp_scaled)

        h1 = np.tanh(disp_scaled @ mlp.coefs_[0] + mlp.intercepts_[0])
        latent_raw = np.tanh(h1 @ mlp.coefs_[1] + mlp.intercepts_[1])
        backend = "Bottleneck Neural Autoencoder"

    recon_mse = np.mean((disp_scaled - recon_scaled) ** 2, axis=1)

    recon_rmse = np.sqrt(np.maximum(recon_mse, 0.0))
    med_rmse = float(np.median(recon_rmse))
    mad_rmse = float(stats.median_abs_deviation(recon_rmse, scale="normal"))
    if mad_rmse < 1e-6:
        mad_rmse = max(float(np.std(recon_rmse)), 1e-4)
    thresh_rmse = med_rmse + 3.5 * mad_rmse
    is_ae_anomaly = recon_rmse > thresh_rmse

    n_lat_comp = min(2, latent_raw.shape[1], n_pts - 1)
    ae_latent_2d = (
        PCA(n_components=n_lat_comp, random_state=42).fit_transform(latent_raw)
        if n_lat_comp == 2
        else np.column_stack([latent_raw[:, 0], np.zeros(n_pts)])
    )

    return dict(
        recon_scaled=recon_scaled,
        recon_mse=recon_mse,
        is_ae_anomaly=is_ae_anomaly,
        ae_latent_2d=ae_latent_2d,
        backend=backend,
    )
