"""Phase 32: portfolio-level vol targeting.

Wraps any allocator: scales the final weights so trailing realized portfolio
vol matches a target. Lowers leverage in stress, raises it in calm markets.
Cash absorbs the difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolTargetConfig:
    enabled: bool = False
    target_annual_vol: float = 0.15      # 15% target
    lookback_days: int = 60
    min_scale: float = 0.20              # never under-leverage past 20%
    max_scale: float = 1.20              # cap modest gross overleveraging
    blend: float = 0.7                   # 0=no smoothing, 1=full smoothing of prior scale


def compute_portfolio_vol_scale(daily_returns: pd.Series, cfg: VolTargetConfig,
                                prev_scale: float | None = None) -> float:
    """Given a series of strategy daily returns up to t-1, return a scale for t."""
    if not cfg.enabled or len(daily_returns) < 20:
        return 1.0
    window = daily_returns.tail(cfg.lookback_days)
    realized_vol = float(window.std(ddof=1) * np.sqrt(252))
    if realized_vol <= 1e-6:
        new_scale = cfg.max_scale
    else:
        new_scale = cfg.target_annual_vol / realized_vol
    new_scale = float(np.clip(new_scale, cfg.min_scale, cfg.max_scale))
    if prev_scale is not None:
        new_scale = cfg.blend * prev_scale + (1 - cfg.blend) * new_scale
    return new_scale
