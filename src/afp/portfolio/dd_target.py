"""Drawdown-targeting exposure wrapper (Phase 44).

Reactive, causal: tracks the realized rolling drawdown of the
strategy's daily-return series and scales gross exposure toward
cash when |DD| exceeds a pre-declared threshold.

Parameters are pre-declared from control-theory / damped-oscillator
analogy (see `docs/phase44_drawdown_target.md`); not tuned on
val/test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DrawdownTargetConfig:
    enabled: bool = False
    lookback_days: int = 252           # rolling window for high-water mark
    dd_trigger: float = 0.10           # static trigger (used when adaptive_trigger=False)
    alpha: float = 1.0                 # rate of de-risking with excess DD
    scale_floor: float = 0.30          # minimum gross exposure
    scale_ceiling: float = 1.0         # ≤100% exposure
    blend: float = 0.5                 # EMA smoothing of scale across days
    # Phase 55: vol-aware adaptive trigger
    adaptive_trigger: bool = False
    vol_window_days: int = 60          # rolling window for vol estimate
    z_score: float = 2.0               # VaR-style 2σ event
    horizon_months: float = 1.0        # 1-month VaR horizon
    adaptive_trigger_min: float = 0.05 # floor on adaptive trigger
    adaptive_trigger_max: float = 0.20 # cap on adaptive trigger


def _realized_drawdown(daily_returns: pd.Series, lookback: int) -> float:
    """Drawdown of cumulative returns vs trailing high-water-mark over `lookback` days."""
    if len(daily_returns) < 2:
        return 0.0
    window = daily_returns.tail(lookback)
    eq = (1.0 + window).cumprod()
    hwm = eq.cummax()
    dd = (eq.iloc[-1] / hwm.iloc[-1]) - 1.0
    return float(min(0.0, dd))


def _adaptive_trigger(daily_returns: pd.Series, cfg: DrawdownTargetConfig) -> float:
    """Vol-aware DD trigger: z·σ·sqrt(252/12) for 1-month VaR (Phase 55)."""
    window = daily_returns.tail(cfg.vol_window_days)
    if len(window) < 10:
        return cfg.dd_trigger
    vol = float(window.std(ddof=1))
    if not np.isfinite(vol) or vol <= 0:
        return cfg.dd_trigger
    days_per_horizon = 252.0 * cfg.horizon_months / 12.0
    trigger = cfg.z_score * vol * np.sqrt(days_per_horizon)
    return float(np.clip(trigger, cfg.adaptive_trigger_min, cfg.adaptive_trigger_max))


def compute_dd_scale(daily_returns: pd.Series, cfg: DrawdownTargetConfig,
                     prev_scale: float | None = None) -> float:
    """Scale factor for the equity slice on day t, based on returns up to t-1."""
    if not cfg.enabled or len(daily_returns) < 20:
        return 1.0
    dd = _realized_drawdown(daily_returns, cfg.lookback_days)
    abs_dd = -dd  # positive number
    trigger = (_adaptive_trigger(daily_returns, cfg)
               if cfg.adaptive_trigger else cfg.dd_trigger)
    if abs_dd <= trigger:
        target = cfg.scale_ceiling
    else:
        excess = abs_dd - trigger
        target = max(cfg.scale_floor,
                     cfg.scale_ceiling - cfg.alpha * excess / trigger)
    target = float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
    if prev_scale is not None and cfg.blend > 0:
        target = cfg.blend * prev_scale + (1 - cfg.blend) * target
    return float(np.clip(target, cfg.scale_floor, cfg.scale_ceiling))
