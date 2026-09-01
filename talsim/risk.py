"""Covariance estimators and diagnostics.

Implemented in plain NumPy so the package has no estimation dependency.
These are for the risk-model comparisons and ex-ante diagnostics; realized
tracking error in the simulator is measured directly from active returns.
"""

from __future__ import annotations

import numpy as np


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of a (T, N) return matrix."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("returns must be (T, N) with T >= 2")
    return np.cov(x, rowvar=False, ddof=1)


def ewma_cov(returns: np.ndarray, halflife: float = 20.0) -> np.ndarray:
    """Exponentially weighted covariance with the given halflife (in rows)."""
    x = np.asarray(returns, dtype=float)
    t = x.shape[0]
    lam = 0.5 ** (1.0 / halflife)
    weights = lam ** np.arange(t - 1, -1, -1)
    weights /= weights.sum()
    mean = weights @ x
    demeaned = x - mean
    return (demeaned * weights[:, None]).T @ demeaned


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a scaled identity target (2004 estimator)."""
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    mu = np.trace(s) / n
    d2 = np.linalg.norm(s - mu * np.eye(n), "fro") ** 2
    b2_sum = 0.0
    for i in range(t):
        xi = x[i][:, None]
        b2_sum += np.linalg.norm(xi @ xi.T - s, "fro") ** 2
    b2 = min(b2_sum / t**2, d2)
    shrink = b2 / d2 if d2 > 0 else 1.0
    return shrink * mu * np.eye(n) + (1 - shrink) * s


def oas_cov(returns: np.ndarray) -> np.ndarray:
    """Oracle Approximating Shrinkage (Chen et al., 2010) toward scaled identity."""
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    mu = np.trace(s) / n
    tr_s2 = np.sum(s * s)
    num = (1 - 2.0 / n) * tr_s2 + (np.trace(s)) ** 2
    den = (t + 1 - 2.0 / n) * (tr_s2 - (np.trace(s)) ** 2 / n)
    shrink = 1.0 if den == 0 else min(num / den, 1.0)
    return shrink * mu * np.eye(n) + (1 - shrink) * s


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues to make a symmetric matrix PSD."""
    sym = (matrix + matrix.T) / 2
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, epsilon, None)
    return vecs @ np.diag(vals) @ vecs.T


def ex_ante_tracking_error(
    active_weights: np.ndarray, cov: np.ndarray, periods_per_year: int
) -> float:
    """Annualized ex-ante tracking error for active weights under `cov`."""
    w = np.asarray(active_weights, dtype=float)
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0) * periods_per_year))
