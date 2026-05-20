"""Phase 11 baseline tests — active universe, dollar-volume, random, shuffled."""

import numpy as np
import pandas as pd
import pytest

from afp.backtest.benchmarks import (
    active_universe_mask,
    dollar_volume_weighted_universe_returns,
    equal_risk_active_universe_returns,
    equal_weight_active_universe_returns,
    random_positive_signal_returns,
    shuffled_signal_returns,
)


@pytest.fixture
def world():
    dates = pd.bdate_range("2024-01-02", periods=80)
    rng = np.random.default_rng(0)
    tickers = list("ABCDEF")
    rows = []
    for t in tickers:
        px = np.cumprod(1 + rng.normal(0.0005, 0.012, len(dates))) * 50
        vol = rng.integers(500_000, 5_000_000, len(dates))
        for i, d in enumerate(dates):
            rows.append({"date": d.date(), "internal_company_id": t,
                         "adjusted_close": px[i], "volume": vol[i]})
    prices = pd.DataFrame(rows)
    preds = pd.DataFrame([
        {"internal_company_id": "A", "predicted_signal": 0.5,  "entry_date": dates[5].date(),
         "exit_date": dates[40].date()},
        {"internal_company_id": "B", "predicted_signal": 0.3,  "entry_date": dates[10].date(),
         "exit_date": dates[60].date()},
        {"internal_company_id": "C", "predicted_signal": 0.1,  "entry_date": dates[20].date(),
         "exit_date": dates[70].date()},
        {"internal_company_id": "D", "predicted_signal": -0.4, "entry_date": dates[15].date(),
         "exit_date": dates[55].date()},
    ])
    return prices, preds, pd.DatetimeIndex(dates)


def test_active_universe_mask_only_true_within_window(world):
    _, preds, dates = world
    mask = active_universe_mask(preds, dates)
    # A is active dates[5..40]
    a_mask = mask["A"]
    assert a_mask.loc[dates[3]] == False
    assert a_mask.loc[dates[20]] == True
    assert a_mask.loc[dates[45]] == False


def test_equal_weight_active_universe_returns_finite(world):
    prices, preds, dates = world
    s = equal_weight_active_universe_returns(prices, preds, dates)
    assert s.index.equals(dates)
    assert s.notna().all()
    assert s.abs().max() < 1.0   # reasonable scale


def test_equal_risk_active_universe_returns_finite(world):
    prices, preds, dates = world
    s = equal_risk_active_universe_returns(prices, preds, dates, lookback_days=20)
    assert s.notna().all()


def test_dollar_volume_weighted_universe_returns_finite(world):
    prices, _, dates = world
    s = dollar_volume_weighted_universe_returns(prices, dates, lookback_days=20)
    assert s.notna().all()


def test_random_positive_signal_returns_finite_and_in_active_set(world):
    prices, preds, dates = world
    s = random_positive_signal_returns(prices, preds, dates, n_holdings=3, seed=42,
                                       lookback_days=20)
    # Should be zero on days before any signal is active
    early = s.loc[s.index < pd.Timestamp(preds["entry_date"].min())]
    assert (early == 0).all()


def test_shuffled_signal_returns_finite(world):
    prices, preds, dates = world
    s = shuffled_signal_returns(prices, preds, dates, n_holdings=3, seed=42, lookback_days=20)
    assert s.notna().all()
