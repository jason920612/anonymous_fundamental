"""Phase 12: point-in-time universe filter applied to event samples.

Implements RFC-01 §6.1 at the *sample* level (after sample_builder, before
target normalization). A sample is kept only if, at its `entry_date`:

  - the company's adjusted close ≥ min_adjusted_close_price_usd
  - the trailing 252-day mean dollar volume ≥ min_avg_daily_dollar_volume_usd
  - the company has at least `min_years_price_history * 252` prior daily bars

This is the cleanest place to apply liquidity / price / history filters because
samples already carry `entry_date` and `internal_company_id`.

Crucially the filter uses *only past data* (RFC-01 §6.3): the rolling dollar
volume series is shifted by one bar before lookup, and we use bars strictly
before `entry_date`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afp.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SampleUniverseConfig:
    enabled: bool = False
    min_adjusted_close_price_usd: float = 5.0
    min_avg_daily_dollar_volume_usd: float = 10_000_000.0
    min_years_price_history: float = 5.0
    avg_dollar_volume_lookback_days: int = 252


def _trailing_dv(prices_one_ticker: pd.DataFrame, lookback: int) -> pd.Series:
    s = prices_one_ticker["adjusted_close"].astype(float) * prices_one_ticker["volume"].astype(float)
    return s.rolling(window=lookback, min_periods=lookback // 4).mean().shift(1)


def apply_sample_universe_filter(
    samples: pd.DataFrame,
    prices: pd.DataFrame,
    config: SampleUniverseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(kept_samples, dropped_samples)` with `exclusion_reason` set on dropped."""
    if not config.enabled or samples.empty or prices is None or prices.empty:
        return samples.copy(), pd.DataFrame()

    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["internal_company_id", "date"])

    # Per-company precomputed series for O(1) lookup
    series_by_company: dict[str, pd.DataFrame] = {}
    for icid, grp in px.groupby("internal_company_id", sort=False):
        grp = grp.copy().reset_index(drop=True)
        grp["trailing_dv"] = _trailing_dv(grp, config.avg_dollar_volume_lookback_days)
        # Strictly-prior look: shift so the lookup at entry_date uses data at d < entry_date.
        grp["adj_close_prev"] = grp["adjusted_close"].astype(float).shift(1)
        series_by_company[icid] = grp.set_index("date")

    min_history_bars = int(config.min_years_price_history * 252)

    kept_idx, dropped_rows = [], []
    for row in samples.itertuples():
        icid = row.internal_company_id
        entry_ts = pd.Timestamp(row.entry_date)
        company_px = series_by_company.get(icid)
        if company_px is None or company_px.empty:
            dropped_rows.append({"sample_id": row.sample_id,
                                 "exclusion_reason": "filter_no_price_series"})
            continue

        past = company_px.loc[company_px.index < entry_ts]
        if len(past) < min_history_bars:
            dropped_rows.append({"sample_id": row.sample_id,
                                 "exclusion_reason": "filter_insufficient_history"})
            continue

        latest = past.iloc[-1]
        adj_close = float(latest["adj_close_prev"] if not np.isnan(latest["adj_close_prev"])
                          else latest["adjusted_close"])
        dv = float(latest["trailing_dv"]) if not np.isnan(latest["trailing_dv"]) else None

        if adj_close < config.min_adjusted_close_price_usd:
            dropped_rows.append({"sample_id": row.sample_id,
                                 "exclusion_reason": "filter_below_min_price"})
            continue
        if dv is None or dv < config.min_avg_daily_dollar_volume_usd:
            dropped_rows.append({"sample_id": row.sample_id,
                                 "exclusion_reason": "filter_below_min_adv"})
            continue

        kept_idx.append(row.Index)

    kept = samples.loc[kept_idx].reset_index(drop=True)
    dropped = pd.DataFrame(dropped_rows)
    log.info("sample_universe_filter_applied",
             extra={"kept": int(len(kept)), "dropped": int(len(dropped)),
                    "drop_rate": round(len(dropped) / max(1, len(samples)), 3)})
    return kept, dropped
