"""Tests for the new optional engine knobs added in Phase 38-42:
- `vol_estimator=` for sharing a VolEstimator across runs.
- `custom_allocator=` for plugging in any allocator function.
- `regime_filter_cfg` / `regime_temperature_series` for Phase 39 scaling.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from afp.backtest.engine import BacktestConfig, run_backtest
from afp.data.trading_calendar import TradingCalendar
from afp.portfolio.allocation import (
    AllocationResult,
    PortfolioConfig,
    inverse_vol_allocation,
)
from afp.portfolio.regime_filter import RegimeFilterConfig
from afp.portfolio.risk_models import RiskConfig, VolEstimator


def _toy_dataset(seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2020-01-02", periods=300, freq="B")
    companies = [f"c{i}" for i in range(8)]
    rows = []
    for c in companies:
        rets = rng.standard_normal(len(days)) * 0.01
        prices = 100 * np.exp(np.cumsum(rets))
        for d, p in zip(days, prices):
            rows.append((d.date(), c, float(p)))
    prices_df = pd.DataFrame(rows, columns=["date", "internal_company_id", "adjusted_close"])

    # Quarterly entry/exit predictions with positive signal for the first half.
    preds = []
    entry_dates = days[::60]
    exit_dates = days[5::60]
    for i, c in enumerate(companies):
        sig = 0.5 if i < 4 else 0.1
        for e, x in zip(entry_dates, exit_dates):
            preds.append({"internal_company_id": c, "entry_date": e, "exit_date": x,
                          "predicted_signal": sig, "target_normalized_signal": 0.0,
                          "raw_log_return": 0.0, "ex_ante_scale": 0.02})
    pred_df = pd.DataFrame(preds)
    cal = TradingCalendar.from_dates(days.tolist())
    return prices_df, pred_df, cal


def test_run_backtest_accepts_shared_vol_estimator():
    prices, preds, cal = _toy_dataset()
    rcfg = RiskConfig()
    shared = VolEstimator(prices, rcfg)
    bt_cfg = BacktestConfig(start_date="2020-01-02", end_date=None)
    pcfg = PortfolioConfig(min_positive_signal=0.05, target_positions=4,
                            max_positions=8, max_single_stock_weight=0.5,
                            min_position_weight=0.0)
    r1 = run_backtest(preds, prices, cal, bt_cfg, pcfg, rcfg, vol_estimator=shared)
    r2 = run_backtest(preds, prices, cal, bt_cfg, pcfg, rcfg)
    # Should give identical results.
    assert np.allclose(r1.daily["net_return"].to_numpy(),
                       r2.daily["net_return"].to_numpy())


def test_run_backtest_accepts_custom_allocator():
    prices, preds, cal = _toy_dataset()
    rcfg = RiskConfig()
    bt_cfg = BacktestConfig(start_date="2020-01-02", end_date=None)
    pcfg = PortfolioConfig(min_positive_signal=0.05, target_positions=4,
                            max_positions=8, max_single_stock_weight=0.5,
                            min_position_weight=0.0)

    def my_alloc(d, p, ve, c, k, sec):
        # All-cash allocator: never invest.
        return AllocationResult(weights=pd.DataFrame(columns=[
            "internal_company_id", "weight", "predicted_signal", "vol"]),
            cash_weight=1.0, n_candidates=0)

    r = run_backtest(preds, prices, cal, bt_cfg, pcfg, rcfg, custom_allocator=my_alloc)
    # All-cash → net return is just the cash rate (0.0 here).
    assert (r.daily["net_return"].abs() < 1e-9).all()


def test_regime_filter_scales_exposure():
    prices, preds, cal = _toy_dataset()
    rcfg = RiskConfig()
    bt_cfg = BacktestConfig(start_date="2020-01-02", end_date=None)
    pcfg = PortfolioConfig(min_positive_signal=0.05, target_positions=4,
                            max_positions=8, max_single_stock_weight=0.5,
                            min_position_weight=0.0)
    # Constant low-temperature → regime filter floors exposure.
    T = pd.Series(np.full(300, 1e-6),
                  index=pd.date_range("2020-01-02", periods=300, freq="B"))
    cfg = RegimeFilterConfig(enabled=True, scale_floor=0.5, scale_ceiling=1.0,
                             boltzmann_kt=2.0, lookback_days=60, smoothing=0.0)
    r = run_backtest(preds, prices, cal, bt_cfg, pcfg, rcfg,
                     regime_filter_cfg=cfg, regime_temperature_series=T)
    # Cash weight should be non-trivial — exposure scaled down.
    avg_cash = r.daily["cash_weight"].mean()
    assert avg_cash > 0.1
