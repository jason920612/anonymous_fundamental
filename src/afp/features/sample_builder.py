"""Phase 3: event-driven supervised sample construction (RFC-02)."""

from __future__ import annotations

import hashlib
import math
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from afp.data.trading_calendar import TradingCalendar
from afp.utils.dates import to_eastern_date


# RFC-02 §15 — allowed exclusion reasons
EXCLUSION_REASONS = {
    "missing_next_event", "missing_entry_price", "missing_exit_price",
    "entry_after_exit", "insufficient_price_history",
    "insufficient_financial_history", "failed_liquidity_filter",
    "security_not_common_stock", "duplicate_event_same_date",
    "invalid_adjusted_close", "missing_ex_ante_scale",
    "target_outlier_filtered",
}


def _sample_id(internal_company_id: str, filing_event_id: str) -> str:
    return "SAMPLE_" + hashlib.sha1(f"{internal_company_id}|{filing_event_id}".encode()).hexdigest()[:16]


def _price_lookup(prices: pd.DataFrame) -> dict[tuple[str, date], float]:
    """`{(internal_company_id, date): adjusted_close}` lookup."""
    keys = list(zip(prices["internal_company_id"], prices["date"]))
    return dict(zip(keys, prices["adjusted_close"].astype(float).tolist()))


def _benchmark_lookup(benchmarks: pd.DataFrame) -> dict[tuple[str, date], float]:
    keys = list(zip(benchmarks["ticker_at_date"], benchmarks["date"]))
    return dict(zip(keys, benchmarks["adjusted_close"].astype(float).tolist()))


def _benchmark_return(lookup: dict[tuple[str, date], float], symbol: str,
                      entry: date, exit_: date) -> float | None:
    p0 = lookup.get((symbol, entry))
    p1 = lookup.get((symbol, exit_))
    if p0 is None or p1 is None or p0 <= 0 or p1 <= 0:
        return None
    return math.log(p1 / p0)


def _dedupe_same_date(filings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per RFC-02 §12: collapse multiple filings on the same Eastern date.
    Prefer 10-K, then latest accepted_datetime. Returns (kept, dropped)."""
    df = filings.copy()
    df["event_date"] = df["accepted_datetime"].map(to_eastern_date)
    df["_form_rank"] = (df["form_type"] == "10-K").astype(int)
    df = df.sort_values(["internal_company_id", "event_date", "_form_rank", "accepted_datetime"],
                        ascending=[True, True, False, False])
    is_kept = ~df.duplicated(subset=["internal_company_id", "event_date"], keep="first")
    kept = df[is_kept].drop(columns=["_form_rank"]).reset_index(drop=True)
    dropped = df[~is_kept].drop(columns=["_form_rank"]).reset_index(drop=True)
    return kept, dropped


def build_event_samples(
    filings: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    benchmarks: pd.DataFrame | None = None,
    train_end_date: str = "2015-12-31",
    validation_end_date: str = "2019-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct one supervised sample per (company, filing-with-next-filing).

    Returns:
      samples : valid rows per RFC-02 §6.
      excluded: rows that could not be turned into a sample, with `exclusion_reason`.
    """
    if filings.empty:
        return pd.DataFrame(), pd.DataFrame()

    kept, dropped = _dedupe_same_date(filings)
    price_lookup = _price_lookup(prices)
    bench_lookup = _benchmark_lookup(benchmarks) if benchmarks is not None else {}

    train_end = pd.Timestamp(train_end_date).date()
    val_end = pd.Timestamp(validation_end_date).date()

    samples: list[dict] = []
    excluded: list[dict] = []

    # Dropped duplicates first
    for row in dropped.itertuples(index=False):
        excluded.append({
            "internal_company_id": row.internal_company_id,
            "filing_event_id": row.filing_event_id,
            "exclusion_reason": "duplicate_event_same_date",
        })

    for icid, grp in kept.groupby("internal_company_id", sort=False):
        grp = grp.sort_values("event_date").reset_index(drop=True)
        for idx in range(len(grp) - 1):
            cur = grp.iloc[idx]
            nxt = grp.iloc[idx + 1]

            try:
                entry_date = calendar.next_trading_day(cur["event_date"])
                exit_date = calendar.previous_trading_day(nxt["event_date"])
            except IndexError:
                excluded.append({"internal_company_id": icid,
                                 "filing_event_id": cur["filing_event_id"],
                                 "exclusion_reason": "insufficient_price_history"})
                continue

            if entry_date >= exit_date:
                excluded.append({"internal_company_id": icid,
                                 "filing_event_id": cur["filing_event_id"],
                                 "exclusion_reason": "entry_after_exit"})
                continue

            entry_price = price_lookup.get((icid, entry_date))
            exit_price = price_lookup.get((icid, exit_date))
            if entry_price is None or entry_price <= 0:
                excluded.append({"internal_company_id": icid,
                                 "filing_event_id": cur["filing_event_id"],
                                 "exclusion_reason": "missing_entry_price"})
                continue
            if exit_price is None or exit_price <= 0:
                excluded.append({"internal_company_id": icid,
                                 "filing_event_id": cur["filing_event_id"],
                                 "exclusion_reason": "missing_exit_price"})
                continue

            raw_log = math.log(exit_price / entry_price)
            raw_simple = exit_price / entry_price - 1.0

            split = ("train" if entry_date <= train_end
                     else "validation" if entry_date <= val_end
                     else "test")

            samples.append({
                "sample_id": _sample_id(icid, cur["filing_event_id"]),
                "internal_company_id": icid,
                "cik": cur["cik"],
                "ticker_at_event": None,
                "filing_event_id": cur["filing_event_id"],
                "report_event_date": cur["event_date"],
                "accepted_datetime": cur["accepted_datetime"],
                "form_type": cur["form_type"],
                "period_end_date": cur["report_period_end_date"],
                "entry_date": entry_date,
                "entry_price": entry_price,
                "next_filing_event_id": nxt["filing_event_id"],
                "next_report_event_date": nxt["event_date"],
                "exit_date": exit_date,
                "exit_price": exit_price,
                "raw_log_return": raw_log,
                "raw_simple_return": raw_simple,
                "ex_ante_scale": np.nan,
                "target_normalized_signal": np.nan,
                "spy_log_return": _benchmark_return(bench_lookup, "SPY", entry_date, exit_date),
                "qqq_log_return": _benchmark_return(bench_lookup, "QQQ", entry_date, exit_date),
                "eligible_for_training": True,
                "eligible_for_backtest": True,
                "exclusion_reason": None,
                "split": split,
                "holding_days": calendar.trading_days_between(entry_date, exit_date),
            })

        # The last filing has no `next` — record as excluded for transparency.
        last = grp.iloc[-1]
        excluded.append({"internal_company_id": icid,
                         "filing_event_id": last["filing_event_id"],
                         "exclusion_reason": "missing_next_event"})

    samples_df = pd.DataFrame(samples)
    excluded_df = pd.DataFrame(excluded)
    return samples_df, excluded_df


def required_sample_columns() -> tuple[str, ...]:
    return ("sample_id", "internal_company_id", "filing_event_id", "report_event_date",
            "entry_date", "entry_price", "exit_date", "exit_price", "raw_log_return",
            "next_report_event_date", "split", "holding_days")


def build_live_samples(
    filings: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    as_of: date | None = None,
    expected_holding_days: int = 60,
) -> pd.DataFrame:
    """For each company, build a LIVE sample anchored at `as_of` (default: today).

    Unlike `build_open_samples` which anchors at the most recent filing's
    entry_date, this anchors at TODAY: the model uses the latest filed
    fundamentals (same as build_open_samples) but the price/vol features and
    ex_ante scale are computed from prices up to `as_of`. This is the
    "rerun the model right now" mode for live position decisions.
    """
    if filings.empty:
        return pd.DataFrame()
    kept, _ = _dedupe_same_date(filings)
    price_lookup = _price_lookup(prices)
    if as_of is None:
        prices_dates = sorted(pd.to_datetime(prices["date"]).dt.date.unique())
        as_of = prices_dates[-1]
    try:
        anchor = calendar.next_trading_day(as_of - pd.Timedelta(days=1))
    except IndexError:
        anchor = as_of

    rows: list[dict] = []
    for icid, grp in kept.groupby("internal_company_id", sort=False):
        grp = grp.sort_values("event_date").reset_index(drop=True)
        # Only consider filings already accepted on/before anchor (no future-data)
        eligible = grp[pd.to_datetime(grp["event_date"]) <= pd.Timestamp(anchor)]
        if eligible.empty:
            continue
        last = eligible.iloc[-1]
        entry_price = price_lookup.get((icid, anchor))
        if entry_price is None or entry_price <= 0:
            continue
        idx = calendar.trading_dates.searchsorted(pd.Timestamp(anchor), side="left")
        exit_idx = min(idx + expected_holding_days, len(calendar.trading_dates) - 1)
        exit_date = calendar.trading_dates[exit_idx].date()
        rows.append({
            "sample_id": _sample_id(icid, last["filing_event_id"]) + "_LIVE",
            "internal_company_id": icid,
            "cik": last["cik"],
            "ticker_at_event": None,
            "filing_event_id": last["filing_event_id"],
            "report_event_date": last["event_date"],
            "accepted_datetime": last["accepted_datetime"],
            "form_type": last["form_type"],
            "period_end_date": last["report_period_end_date"],
            "entry_date": anchor,
            "entry_price": entry_price,
            "next_filing_event_id": None,
            "next_report_event_date": None,
            "exit_date": exit_date,
            "exit_price": None,
            "raw_log_return": np.nan,
            "raw_simple_return": np.nan,
            "ex_ante_scale": np.nan,
            "target_normalized_signal": np.nan,
            "spy_log_return": None,
            "qqq_log_return": None,
            "eligible_for_training": False,
            "eligible_for_backtest": False,
            "exclusion_reason": "live_prediction_anchor",
            "split": "live",
            "holding_days": expected_holding_days,
        })
    return pd.DataFrame(rows)


def build_open_samples(
    filings: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    expected_holding_days: int = 60,
) -> pd.DataFrame:
    """For each company, build an "open" sample for its MOST RECENT filing.

    The standard `build_event_samples` pipeline pairs each filing with its
    next one (entry after current, exit before next). The latest filing has
    no next, so it is excluded — but that's exactly the position the user
    wants to predict on. This helper synthesizes the "open" sample:

      entry_date = first trading day after the latest filing's event_date
      exit_date  = entry_date + `expected_holding_days` trading days
                   (placeholder for the unknown next filing date)
      raw_log_return / exit_price / target = NaN (unknown)

    The encoder + LambdaRank can still score these rows because they only
    need feature data available at the filing, not the realized return.
    """
    if filings.empty:
        return pd.DataFrame()

    kept, _ = _dedupe_same_date(filings)
    price_lookup = _price_lookup(prices)
    rows: list[dict] = []

    for icid, grp in kept.groupby("internal_company_id", sort=False):
        grp = grp.sort_values("event_date").reset_index(drop=True)
        last = grp.iloc[-1]
        try:
            entry_date = calendar.next_trading_day(last["event_date"])
        except IndexError:
            continue
        entry_price = price_lookup.get((icid, entry_date))
        if entry_price is None or entry_price <= 0:
            continue
        idx = calendar.trading_dates.searchsorted(pd.Timestamp(entry_date), side="left")
        if idx + expected_holding_days >= len(calendar.trading_dates):
            exit_date = calendar.trading_dates[-1].date()
        else:
            exit_date = calendar.trading_dates[idx + expected_holding_days].date()

        rows.append({
            "sample_id": _sample_id(icid, last["filing_event_id"]) + "_OPEN",
            "internal_company_id": icid,
            "cik": last["cik"],
            "ticker_at_event": None,
            "filing_event_id": last["filing_event_id"],
            "report_event_date": last["event_date"],
            "accepted_datetime": last["accepted_datetime"],
            "form_type": last["form_type"],
            "period_end_date": last["report_period_end_date"],
            "entry_date": entry_date,
            "entry_price": entry_price,
            "next_filing_event_id": None,
            "next_report_event_date": None,
            "exit_date": exit_date,
            "exit_price": None,
            "raw_log_return": np.nan,
            "raw_simple_return": np.nan,
            "ex_ante_scale": np.nan,
            "target_normalized_signal": np.nan,
            "spy_log_return": None,
            "qqq_log_return": None,
            "eligible_for_training": False,
            "eligible_for_backtest": False,
            "exclusion_reason": "open_position_forecast",
            "split": "open",
            "holding_days": expected_holding_days,
        })
    return pd.DataFrame(rows)
