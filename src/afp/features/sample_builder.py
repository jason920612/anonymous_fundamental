"""Phase 3: event-driven supervised sample construction (RFC-02)."""

from __future__ import annotations

import hashlib
import math
from datetime import date
from typing import Iterable

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
