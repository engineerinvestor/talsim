"""Synthetic market: factor returns plus a persistent cross-sectional signal.

Returns are generated per step as market + sector + idiosyncratic components.
The cross-sectional signal follows an AR(1) so that rankings persist across
rebalances, which is what gives a long-short book stable active positions
(and therefore stable wash-sale interactions) rather than pure noise.

This is a research market: no dividend-price interaction, no delistings,
no corporate actions, and stationary parameters. Every result downstream is
conditional on this process; that is a feature for controlled experiments
and a limitation for empirical claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ScenarioConfig


@dataclass
class MarketPath:
    returns: np.ndarray  # (n_steps, n_assets) price returns per step
    market_returns: np.ndarray  # (n_steps,) market factor return per step
    signals: np.ndarray  # (n_steps, n_assets) cross-sectional signal at step start


def generate_path(cfg: ScenarioConfig, seed: int) -> MarketPath:
    rng = np.random.default_rng(seed)
    n, k = cfg.n_steps, cfg.n_assets
    dt = 1.0 / cfg.steps_per_year

    sectors = np.arange(k) % cfg.n_sectors
    betas = rng.uniform(0.8, 1.2, size=k)

    market = cfg.market_drift * dt + cfg.market_vol * np.sqrt(dt) * rng.standard_normal(n)
    sector_shocks = cfg.sector_vol * np.sqrt(dt) * rng.standard_normal((n, cfg.n_sectors))
    idio = cfg.idio_vol * np.sqrt(dt) * rng.standard_normal((n, k))

    # Persistent cross-sectional signal, unit-variance stationary AR(1).
    rho = cfg.signal_autocorr
    signals = np.empty((n, k))
    state = rng.standard_normal(k)
    for t in range(n):
        signals[t] = state
        state = rho * state + np.sqrt(1 - rho**2) * rng.standard_normal(k)

    returns = betas[None, :] * market[:, None] + sector_shocks[:, sectors] + idio
    return MarketPath(returns=returns, market_returns=market, signals=signals)
