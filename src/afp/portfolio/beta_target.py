"""Beta-targeting exposure wrapper (Phase 46).

Reactive vs predictive trade-off:

- Phase 44 (drawdown-targeting) is reactive — it fires AFTER a
  drawdown has already accumulated.
- Phase 46 (beta-targeting) is predictive — when the strategy's
  rolling beta to the market benchmark exceeds 1, it implies
  concentrated market exposure that will *amplify* future market
  shocks before any drawdown shows up.

Theoretical justification (CAPM / Sharpe 1964):
  r_p(t) = α + β · r_m(t) + ε(t)
  Var(r_p) = β² Var(r_m) + Var(ε)

A β > 1 portfolio is structurally more vulnerable to market shocks
than the market itself. For a long-only strategy that cannot hedge,
the only lever is to scale down gross exposure when β is high.

The scale is:
  scale(t) = clip(1 / max(1, β_rolling), floor, 1.0)

so β ≤ 1 → no change; β = 1.5 → 67% exposure; β = 2.0 → 50%.

Discipline: lookback (60 days), floor (0.3), ceiling (1.0) are all
pre-declared from theory. We do not tune.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BetaTargetConfig:
    enabled: bool = False
    lookback_days: int = 60            # rolling window for beta estimation
    scale_floor: float = 0.30
    scale_ceiling: float = 1.0
    blend: float = 0.5                 # EMA smoothing of scale across days
    market_zero_var: float = 1e-9      # treat near-zero market var as undefined


def _rolling_beta(strategy_returns: pd.Series, market_returns: pd.Series,
                  lookback: int) -> float:
    """OLS beta on the trailing `lookback` overlapping days."""
    strat = strategy_returns.tail(lookback)
    aligned = market_returns.reindex(strat.index).dropna()
    if len(aligned) < max(20, lookback // 4):
        return 1.0
    s = strat.reindex(aligned.index)
    var_m = float(aligned.var(ddof=1))
    if var_m < 1e-12:
        return 1.0
    cov_sm = float(((s - s.mean()) * (aligned - aligned.mean())).sum() / (len(aligned) - 1))
    return cov_sm / var_m


def compute_beta_scale(strategy_returns: pd.Series, market_returns: pd.Series,
                       cfg: BetaTargetConfig, prev_scale: float | None = None) -> float:
    """Scale factor for the equity slice on day t based on rolling beta."""
    if not cfg.enabled or len(strategy_returns) < 20 or len(market_returns) < 20:
        return 1.0
    beta = _rolling_beta(strategy_returns, market_returns, cfg.lookback_days)
    if not np.isfinite(beta):
        return 1.0
    target = 1.0 / max(1.0, beta)
    target = float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
    if prev_scale is not None and cfg.blend > 0:
        target = cfg.blend * prev_scale + (1 - cfg.blend) * target
    return float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
