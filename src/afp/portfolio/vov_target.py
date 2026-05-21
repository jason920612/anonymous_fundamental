"""Volatility-of-volatility (vov) targeting wrapper (Phase 50).

Theory: GARCH-style heteroscedasticity. Volatility itself is volatile
— σ_t is not constant. Periods of high `vov(t) = std(σ_t)` indicate
*regime instability* — the system is in a transient state where
distributional assumptions break.

This is distinct from:
- dd_target (Phase 44): reactive to realized drawdown.
- regime_filter (Phase 39): reactive to cross-sectional dispersion.
- vol_target (Phase 32): reactive to realized vol level.

vov targeting fires EARLIER than dd_target because it picks up the
"vol is shifting" signal *before* the drawdown manifests.

Causal computation:
  σ_t = std of strategy net returns over a short rolling window
  vov_t = std of σ_t over a longer rolling window
  scale_t = clip(1 - α · max(0, vov_z), floor, 1)
  where vov_z is the rolling z-score of vov(t).

Pre-declared parameters:
- σ window = 20 trading days
- vov window = 60 trading days
- kT = 2.0 (Boltzmann factor: 1σ → exp(-0.5)≈0.61, 2σ → exp(-1)≈0.37)
- floor = 0.30
- blend = 0.5 (EMA smoothing)

The kT=2 choice mirrors Phase 39's regime filter, treating vov-z
analogously to dispersion-z. No tuning on val/test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VovTargetConfig:
    enabled: bool = False
    sigma_window: int = 20
    vov_window: int = 60
    boltzmann_kt: float = 2.0       # Boltzmann factor (Phase 39 consistent)
    scale_floor: float = 0.30
    scale_ceiling: float = 1.0
    blend: float = 0.5


def _rolling_vov(returns: pd.Series, sigma_window: int, vov_window: int) -> pd.Series:
    """Causal vol-of-vol: rolling std of rolling std of returns."""
    sigma = returns.rolling(sigma_window, min_periods=5).std(ddof=1)
    vov = sigma.rolling(vov_window, min_periods=10).std(ddof=1)
    return vov


def compute_vov_scale(daily_returns: pd.Series, cfg: VovTargetConfig,
                     prev_scale: float | None = None) -> float:
    """Causal exposure scale based on the strategy's current vol-of-vol z-score."""
    n_needed = cfg.sigma_window + cfg.vov_window + 5
    if not cfg.enabled or len(daily_returns) < n_needed:
        return 1.0
    vov = _rolling_vov(daily_returns, cfg.sigma_window, cfg.vov_window).dropna()
    if len(vov) < 20:
        return 1.0
    # z-score the current vov against its own rolling history.
    mu = float(vov.iloc[:-1].mean())
    sd = float(vov.iloc[:-1].std(ddof=1))
    if sd < 1e-12:
        return 1.0
    z = (float(vov.iloc[-1]) - mu) / sd
    # Boltzmann damping on POSITIVE z only (vov is "hot" → cold-down).
    target = float(np.exp(-max(0.0, z) / cfg.boltzmann_kt))
    target = float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
    if prev_scale is not None and cfg.blend > 0:
        target = cfg.blend * prev_scale + (1 - cfg.blend) * target
    return float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
