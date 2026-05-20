"""Point-in-time universe construction (RFC-01 §6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class UniverseConfig:
    min_adjusted_close_price_usd: float = 5.0
    min_avg_daily_dollar_volume_usd: float = 10_000_000.0
    min_financial_report_events: int = 12
    min_years_price_history: int = 5
    avg_dollar_volume_lookback_days: int = 252


def compute_dollar_volume(prices: pd.DataFrame) -> pd.DataFrame:
    """Adds `dollar_volume` column (adjusted_close * volume)."""
    df = prices.copy()
    df["dollar_volume"] = df["adjusted_close"].astype(float) * df["volume"].astype(float)
    return df


def trailing_dollar_volume(prices_one_ticker: pd.DataFrame, lookback: int) -> pd.Series:
    """Rolling mean of `dollar_volume` over the past `lookback` trading rows.

    The result is shifted by one row so the value at row `i` uses only past data
    (RFC-01 §6.2: "Must use only past data").
    """
    s = prices_one_ticker["dollar_volume"].astype(float)
    return s.rolling(window=lookback, min_periods=lookback // 2).mean().shift(1)


def point_in_time_universe(
    prices: pd.DataFrame,
    filings: pd.DataFrame,
    as_of: date,
    config: UniverseConfig,
) -> pd.DataFrame:
    """Return a DataFrame of `internal_company_id`s eligible on `as_of`.

    Eligibility = liquidity + price + history filters at `as_of`. Uses
    `ticker_at_date` to identify the security row; pair it with `companies`
    upstream to map back to `internal_company_id`. This function expects
    `prices` to already carry `internal_company_id` and `dollar_volume`.
    """
    cutoff = pd.Timestamp(as_of).date()
    px = prices[prices["date"] <= cutoff].copy()
    if px.empty:
        return pd.DataFrame(columns=["internal_company_id"])

    eligible = []
    for icid, grp in px.groupby("internal_company_id", sort=False):
        grp = grp.sort_values("date")
        if grp["date"].iloc[-1] != cutoff and grp["date"].iloc[-1] < cutoff - pd.Timedelta(days=10):
            continue  # stale series
        latest_close = float(grp["adjusted_close"].iloc[-1])
        if latest_close < config.min_adjusted_close_price_usd:
            continue
        if len(grp) < config.min_years_price_history * 252:
            continue
        adv = trailing_dollar_volume(grp, config.avg_dollar_volume_lookback_days)
        adv_latest = adv.iloc[-1]
        if not np.isfinite(adv_latest) or adv_latest < config.min_avg_daily_dollar_volume_usd:
            continue
        eligible.append(icid)

    if filings is not None and not filings.empty:
        ev_counts = (filings[filings["accepted_datetime"] <= str(cutoff)]
                     .groupby("internal_company_id").size())
        eligible = [c for c in eligible
                    if ev_counts.get(c, 0) >= config.min_financial_report_events]

    return pd.DataFrame({"internal_company_id": eligible, "as_of": cutoff})
