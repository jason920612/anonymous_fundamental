"""Ex-ante volatility scale for event-period target normalization (RFC-04 §4-7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from afp.data.trading_calendar import TradingCalendar


@dataclass
class ScaleConfig:
    method: str = "hybrid_daily_event"   # hybrid_daily_event | daily | event
    daily_vol_lookback_days: int = 252
    event_vol_lookback_events: int = 8
    min_event_history: int = 4
    daily_weight: float = 0.7
    event_weight: float = 0.3
    min_ex_ante_scale: float = 0.02
    max_ex_ante_scale: float = 1.00


# -----------------------------------------------------------------------------

def _company_daily_log_returns(price_history: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from a per-company sorted price frame."""
    p = price_history["adjusted_close"].astype(float).to_numpy()
    if len(p) < 2:
        return pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    log_ret = np.log(p[1:] / p[:-1])
    return pd.Series(log_ret, index=pd.to_datetime(price_history["date"].iloc[1:].to_numpy()))


def trailing_daily_vol(
    daily_returns: pd.Series,
    end_date_exclusive: date,
    lookback_days: int,
) -> float | None:
    """Std of daily log returns over the trailing window strictly before `end_date_exclusive`.

    RFC-04 §5.2 — must use only data with `d < entry_date`.
    """
    if daily_returns.empty:
        return None
    cutoff = pd.Timestamp(end_date_exclusive)
    past = daily_returns.loc[daily_returns.index < cutoff]
    if len(past) < lookback_days // 2:
        return None
    window = past.tail(lookback_days)
    vol = float(window.std(ddof=1))
    return vol if np.isfinite(vol) and vol > 0 else None


def trailing_event_vol(
    prior_event_returns: pd.Series,
    lookback_events: int,
    min_events: int,
) -> float | None:
    """Robust std of prior event-to-event log returns (MAD-based, RFC-04 §6)."""
    if len(prior_event_returns) < min_events:
        return None
    window = prior_event_returns.tail(lookback_events).to_numpy()
    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med)))
    if mad <= 0:
        std = float(np.std(window, ddof=1))
        return std if std > 0 else None
    return 1.4826 * mad


# -----------------------------------------------------------------------------

def compute_ex_ante_scale_for_samples(
    samples: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    config: ScaleConfig,
) -> pd.DataFrame:
    """Return a copy of `samples` with `daily_vol`, `event_vol`, `ex_ante_scale`,
    `standardized_movement`, `is_extreme_target_event` filled in.

    `target_normalized_signal` is *not* set here — that's `target_transform.apply`.
    """
    if samples.empty:
        return samples.copy()

    out = samples.copy()

    # Index daily-log-returns by company
    price_by_company: dict[str, pd.Series] = {}
    for icid, grp in prices.sort_values(["internal_company_id", "date"]).groupby(
            "internal_company_id", sort=False):
        price_by_company[icid] = _company_daily_log_returns(grp)

    # Prior event returns per company, sorted by entry_date
    out_sorted = out.sort_values(["internal_company_id", "entry_date"]).copy()
    prior_event_returns: dict[str, list[float]] = {}
    daily_vols, event_vols, scales = [], [], []

    for row in out_sorted.itertuples():
        icid = row.internal_company_id
        entry_date = pd.Timestamp(row.entry_date).date()
        holding_days = int(row.holding_days)

        daily_log_returns = price_by_company.get(icid, pd.Series(dtype=float))
        daily_vol = trailing_daily_vol(daily_log_returns, entry_date,
                                       config.daily_vol_lookback_days)
        scale_daily = daily_vol * np.sqrt(max(holding_days, 1)) if daily_vol is not None else None

        prior = pd.Series(prior_event_returns.get(icid, []), dtype=float)
        event_vol = trailing_event_vol(prior, config.event_vol_lookback_events,
                                       config.min_event_history)

        if config.method == "daily" or event_vol is None:
            scale = scale_daily
        elif config.method == "event" or scale_daily is None:
            scale = event_vol
        else:
            scale = config.daily_weight * scale_daily + config.event_weight * event_vol

        if scale is not None and np.isfinite(scale) and scale > 0:
            scale = float(np.clip(scale, config.min_ex_ante_scale, config.max_ex_ante_scale))
        else:
            scale = np.nan

        daily_vols.append(daily_vol)
        event_vols.append(event_vol)
        scales.append(scale)

        # Append after computing — uses only strictly-prior events.
        prior_event_returns.setdefault(icid, []).append(float(row.raw_log_return))

    out_sorted["daily_vol"] = daily_vols
    out_sorted["event_vol"] = event_vols
    out_sorted["ex_ante_scale"] = scales
    out_sorted["standardized_movement"] = out_sorted["raw_log_return"] / out_sorted["ex_ante_scale"]
    out_sorted["is_extreme_target_event"] = (
        out_sorted["raw_log_return"].abs() > 3.0 * out_sorted["ex_ante_scale"]
    ).fillna(False)

    return out_sorted.reset_index(drop=True)
