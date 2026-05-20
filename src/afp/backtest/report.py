"""Build a backtest report summary comparing strategy to benchmarks (RFC-07 §15)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from afp.backtest.benchmarks import (
    benchmark_daily_returns,
    dollar_volume_weighted_universe_returns,
    equal_risk_active_universe_returns,
    equal_risk_universe_returns,
    equal_weight_active_universe_returns,
    equal_weight_universe_returns,
    random_positive_signal_returns,
    shuffled_signal_returns,
)
from afp.backtest.engine import BacktestResult
from afp.backtest.metrics import summary


def build_report(
    result: BacktestResult,
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame | None,
    trading_days_per_year: int = 252,
    predictions: pd.DataFrame | None = None,
    target_positions: int = 50,
    random_seed: int = 42,
) -> dict:
    daily_returns = result.daily.set_index(pd.to_datetime(result.daily["date"]))["net_return"]
    turnover_series = result.daily.set_index(pd.to_datetime(result.daily["date"]))["turnover"]

    summary_dict = {"strategy": summary(daily_returns, turnover_series, trading_days_per_year)}

    dates = daily_returns.index
    if benchmarks is not None and not benchmarks.empty:
        for symbol in benchmarks["ticker_at_date"].unique():
            br = benchmark_daily_returns(benchmarks, symbol).reindex(dates).fillna(0.0)
            summary_dict[f"benchmark_{symbol.lower()}"] = summary(br, trading_days_per_year=trading_days_per_year)

    ew = equal_weight_universe_returns(prices, dates).reindex(dates).fillna(0.0)
    summary_dict["benchmark_equal_weight_universe"] = summary(
        ew, trading_days_per_year=trading_days_per_year)

    er = equal_risk_universe_returns(prices, dates).reindex(dates).fillna(0.0)
    summary_dict["benchmark_equal_risk_universe"] = summary(
        er, trading_days_per_year=trading_days_per_year)
    # Realistic cost-aware version of equal_risk_universe.
    er_cost = equal_risk_universe_returns(prices, dates).reindex(dates).fillna(0.0)  # placeholder

    dv = dollar_volume_weighted_universe_returns(prices, dates).reindex(dates).fillna(0.0)
    summary_dict["benchmark_dollar_volume_weighted"] = summary(
        dv, trading_days_per_year=trading_days_per_year)

    if predictions is not None and not predictions.empty:
        ewa = equal_weight_active_universe_returns(prices, predictions, dates).reindex(dates).fillna(0.0)
        summary_dict["benchmark_equal_weight_active_universe"] = summary(
            ewa, trading_days_per_year=trading_days_per_year)
        era = equal_risk_active_universe_returns(prices, predictions, dates).reindex(dates).fillna(0.0)
        summary_dict["benchmark_equal_risk_active_universe"] = summary(
            era, trading_days_per_year=trading_days_per_year)
        era_cost = equal_risk_active_universe_returns(
            prices, predictions, dates, transaction_cost_bps=10.0
        ).reindex(dates).fillna(0.0)
        summary_dict["benchmark_equal_risk_active_universe_cost10bps"] = summary(
            era_cost, trading_days_per_year=trading_days_per_year)
        rps = random_positive_signal_returns(prices, predictions, dates,
                                             n_holdings=target_positions, seed=random_seed).fillna(0.0)
        summary_dict["benchmark_random_positive_signal"] = summary(
            rps, trading_days_per_year=trading_days_per_year)
        shf = shuffled_signal_returns(prices, predictions, dates,
                                      n_holdings=target_positions, seed=random_seed).fillna(0.0)
        summary_dict["benchmark_shuffled_signal"] = summary(
            shf, trading_days_per_year=trading_days_per_year)

    return summary_dict


def write_report(summary_dict: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for portfolio_name, metrics in summary_dict.items():
        row = {"portfolio": portfolio_name, **metrics}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    return out / "metrics.csv"
