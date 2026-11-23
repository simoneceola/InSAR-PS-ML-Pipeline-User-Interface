"""Hand-crafted trajectory features that feed the two Random Forest classifiers."""

import numpy as np
from scipy import stats

BP_GRID = np.linspace(0.08, 0.92, 43)


def multi_segment_fit(x: np.ndarray, ts: np.ndarray, max_segments: int = 5, min_gap: float = 0.08):
    X0 = np.column_stack([np.ones_like(x), x])
    Q, _ = np.linalg.qr(X0)
    r = ts - Q @ (Q.T @ ts)
    rmses = [float(np.sqrt(np.mean(r ** 2)))]

    H_all = np.maximum(0.0, x[:, None] - BP_GRID[None, :])
    chosen_bps: list[float] = []

    for _ in range(max_segments - 1):
        valid_mask = np.ones(len(BP_GRID), dtype=bool)
        for bp in chosen_bps:
            valid_mask &= np.abs(BP_GRID - bp) >= min_gap
        if not np.any(valid_mask):
            rmses.append(rmses[-1])
            continue

        cand_indices = np.where(valid_mask)[0]
        H_cand = H_all[:, cand_indices]
        H_perp = H_cand - Q @ (Q.T @ H_cand)
        norms = np.linalg.norm(H_perp, axis=0)
        usable = norms > 1e-8
        if not np.any(usable):
            rmses.append(rmses[-1])
            continue

        H_perp = H_perp[:, usable]
        norms = norms[usable]
        cand_indices = cand_indices[usable]

        gains = (r @ H_perp) ** 2 / (norms ** 2 + 1e-16)
        best_local = int(np.argmax(gains))
        chosen_bps.append(float(BP_GRID[cand_indices[best_local]]))

        q_new = H_perp[:, best_local] / norms[best_local]
        Q = np.column_stack([Q, q_new])
        r = r - (r @ q_new) * q_new
        rmses.append(float(np.sqrt(np.mean(r ** 2))))

    bps = np.sort(chosen_bps)
    if len(bps) > 0:
        A = np.column_stack([np.ones_like(x), x] + [np.maximum(0.0, x - b) for b in bps])
        coef = np.linalg.lstsq(A, ts, rcond=None)[0]
        slopes = np.cumsum(coef[1:])
    else:
        slopes = np.array([np.polyfit(x, ts, 1)[0]])

    return rmses, bps, slopes


def extract_regime_features(ts: np.ndarray) -> list[float]:
    arr = np.asarray(ts, dtype=np.float64)
    n = len(arr)
    k = max(3, n // 10)
    arr = arr - arr[:k].mean()
    x = np.linspace(0.0, 1.0, n)

    slope, intercept = np.polyfit(x, arr, 1)
    resid = arr - (slope * x + intercept)
    rmse_lin = float(np.sqrt(np.mean(resid ** 2)))
    r2_lin = float(1.0 - np.sum(resid ** 2) / (np.sum((arr - arr.mean()) ** 2) + 1e-12))
    edge_change = float(arr[-k:].mean() - arr[:k].mean())
    tau = float(np.nan_to_num(stats.kendalltau(x, arr)[0]))
    frac_down = float(np.mean(np.diff(arr) < 0))

    return [
        float(slope),
        edge_change,
        float(arr.max()),
        float(arr.min()),
        rmse_lin,
        r2_lin,
        abs(float(slope)) / (rmse_lin + 1e-6),
        tau,
        frac_down,
        float(arr[-1] - arr.min()),
        float(arr.max() - arr[-1]),
        float(np.argmin(arr) / max(n - 1, 1)),
        float(np.argmax(arr) / max(n - 1, 1)),
    ]


def extract_linearity_features(ts: np.ndarray) -> list[float]:
    arr = np.asarray(ts, dtype=np.float64)
    n = len(arr)
    k = max(3, n // 10)
    arr = arr - arr[:k].mean()
    x = np.linspace(0.0, 1.0, n)

    p1 = np.polyfit(x, arr, 1)
    resid_lin = arr - np.polyval(p1, x)
    rmse_lin = float(np.sqrt(np.mean(resid_lin ** 2)))
    r2_lin = float(1.0 - np.sum(resid_lin ** 2) / (np.sum((arr - arr.mean()) ** 2) + 1e-12))

    rmses, _, seg_slopes = multi_segment_fit(x, arr, max_segments=5)
    denom = rmse_lin + 1e-6

    eff_segs = 1
    for idx in range(1, len(rmses)):
        if rmses[idx - 1] > 0.35 and rmses[idx] < 0.80 * rmses[idx - 1]:
            eff_segs = idx + 1
        else:
            break

    sig_slopes = seg_slopes[np.abs(seg_slopes) > 1.0]
    sign_changes = int(np.sum(np.diff(np.sign(sig_slopes)) != 0)) if len(sig_slopes) > 1 else 0

    vel = np.diff(arr) * (n - 1)
    accel = np.diff(arr, n=2) * ((n - 1) ** 2)
    zero_crossings = int(np.sum(np.diff(np.signbit(resid_lin))))
    autocorr = (
        float(np.corrcoef(resid_lin[:-1], resid_lin[1:])[0, 1])
        if np.std(resid_lin) > 1e-9
        else 0.0
    )

    return [
        abs(float(p1[0])),
        rmse_lin,
        float(np.max(np.abs(resid_lin))),
        r2_lin,
        rmses[1] / denom,
        rmses[2] / denom,
        rmses[3] / denom,
        rmses[4] / denom,
        (rmses[1] - rmses[-1]) / denom,
        float(eff_segs),
        float(np.std(seg_slopes)),
        float(np.ptp(seg_slopes)),
        float(sign_changes),
        float(np.percentile(vel, 90) - np.percentile(vel, 10)),
        float(np.sqrt(np.mean(accel ** 2))),
        float(zero_crossings),
        float(np.nan_to_num(autocorr)),
    ]
