"""Thermodynamic regime detector (Phase 39).

Treats the cross-sectional dispersion of daily returns as a "temperature"
of the market microstate:

    T(t) = std_i { r_i(t) }   over the active universe on day t

Statistical-mechanics intuition:

- In a hot regime, many independent dimensions of variation are active:
  stock-specific information drives prices apart. Selection-based long-
  only strategies have meaningful breadth and tend to extract alpha.

- In a cold regime, the spectrum collapses onto a single (market) mode:
  every stock moves together. Selection breadth → 0; long-only stock
  picks behave like a beta-1 index but with extra idiosyncratic risk.

The detector outputs an exposure-scale factor in [scale_floor, 1]:

    z(t)     = (T(t) - μ_window) / σ_window         (causal rolling stats)
    scale(t) = clip(exp(z(t)/temperature_kt), floor, 1)

The damping form `exp(z / kT)` is the Boltzmann factor — exposure drops
exponentially as the regime cools. There is exactly one knob (kT, the
"Boltzmann temperature") and we set it from theory to 2.0, meaning a
–2σ cold regime is damped to ≈37% exposure. This is **not** tuned on
val/test data.

All inputs (`returns_panel`) come from past prices that are visible
under any honest causal evaluation — no peek into test labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class RegimeFilterConfig:
    enabled: bool = False
    lookback_days: int = 252           # rolling window for z-score (≈1y)
    boltzmann_kt: float = 2.0          # 1 z below mean → exp(-0.5)≈0.61 scale
    scale_floor: float = 0.30          # never go under 30% gross
    scale_ceiling: float = 1.0         # never lever above 100%
    smoothing: float = 0.5             # EMA smoothing of scale across days


def cross_sectional_temperature(price_panel: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional std of log-returns across the universe.

    Parameters
    ----------
    price_panel : DataFrame indexed by date, columns = internal_company_id,
                  values = adjusted_close (already wide-form pivoted).

    Returns
    -------
    Series indexed by date with the cross-sectional dispersion T(t).
    """
    log_p = np.log(price_panel.astype(float).replace(0.0, np.nan))
    ret = log_p.diff()
    # std across columns (assets) for each day
    return ret.std(axis=1, ddof=1)


def compute_regime_scales(temperature_series: pd.Series,
                          cfg: RegimeFilterConfig) -> pd.Series:
    """Causal exposure-scale series — one scalar per day."""
    if not cfg.enabled:
        return pd.Series(1.0, index=temperature_series.index)

    T = temperature_series.astype(float)
    mu = T.rolling(window=cfg.lookback_days, min_periods=20).mean()
    sd = T.rolling(window=cfg.lookback_days, min_periods=20).std(ddof=1)
    z = (T - mu) / sd.replace(0.0, np.nan)

    # Boltzmann factor: exp(z / kT) clamped to [floor, ceiling]
    raw_scale = np.exp(z / cfg.boltzmann_kt)
    raw_scale = raw_scale.clip(lower=cfg.scale_floor, upper=cfg.scale_ceiling)
    raw_scale = raw_scale.fillna(1.0)

    if cfg.smoothing > 0:
        smoothed = raw_scale.ewm(alpha=1.0 - cfg.smoothing, adjust=False).mean()
    else:
        smoothed = raw_scale
    return smoothed.clip(lower=cfg.scale_floor, upper=cfg.scale_ceiling)


def regime_scale_at(temperature_series: pd.Series, cfg: RegimeFilterConfig,
                    asof: date) -> float:
    """Lookup the regime scale for a specific date (or the latest before)."""
    if not cfg.enabled or temperature_series.empty:
        return 1.0
    scales = compute_regime_scales(temperature_series, cfg)
    asof_ts = pd.Timestamp(asof)
    past = scales.loc[scales.index <= asof_ts]
    if past.empty:
        return 1.0
    return float(past.iloc[-1])
