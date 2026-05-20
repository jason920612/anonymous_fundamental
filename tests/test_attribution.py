"""Phase 11 portfolio attribution tests."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from afp.backtest.attribution import (
    build_attribution_report,
    cash_and_turnover_summary,
    concentration,
    dollar_volume_exposure,
    top_contributors,
)


@pytest.fixture
def holdings_and_prices():
    dates = pd.bdate_range("2024-01-02", periods=50)
    rng = np.random.default_rng(0)
    tickers = ["A", "B", "C", "D"]
    price_rows = []
    px = {t: 100.0 for t in tickers}
    for d in dates:
        for t in tickers:
            px[t] *= (1 + rng.normal(0.001, 0.01))
            price_rows.append({"date": d.date(), "internal_company_id": t,
                               "adjusted_close": px[t], "volume": rng.integers(1_000_000, 5_000_000)})
    prices = pd.DataFrame(price_rows)
    holdings_rows = []
    for d in dates:
        for t in tickers:
            w = 0.25 if d.weekday() < 3 else (0.4 if t == "A" else 0.2)
            holdings_rows.append({"date": d.date(), "internal_company_id": t, "weight": w})
    return pd.DataFrame(holdings_rows), prices


def test_top_contributors_returns_ranked_table(holdings_and_prices):
    holdings, prices = holdings_and_prices
    out = top_contributors(holdings, prices, n=4)
    assert len(out) == 4
    assert "total_contribution" in out.columns
    assert out["total_contribution"].is_monotonic_decreasing


def test_concentration_includes_herfindahl_and_top_buckets(holdings_and_prices):
    holdings, _ = holdings_and_prices
    out = concentration(holdings)
    for col in ("date", "herfindahl", "top5_weight", "top10_weight", "n_positions"):
        assert col in out.columns
    assert (out["herfindahl"] > 0).all()
    assert (out["herfindahl"] <= 1.0 + 1e-9).all()


def test_dollar_volume_exposure_per_day(holdings_and_prices):
    holdings, prices = holdings_and_prices
    out = dollar_volume_exposure(holdings, prices, lookback_days=10)
    assert "weighted_avg_dollar_volume" in out.columns


def test_cash_and_turnover_summary_metrics():
    daily = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=10).date,
        "cash_weight": [0.0, 0.1, 0.2, 0.0, 0.5, 0.0, 0.0, 0.1, 0.0, 0.0],
        "turnover": [0.0, 0.05, 0.02, 0.0, 0.2, 0.0, 0.0, 0.03, 0.0, 0.0],
    })
    out = cash_and_turnover_summary(daily, tcost_bps=10.0)
    assert "avg_cash_weight" in out and out["avg_cash_weight"] > 0
    assert "annualized_turnover" in out
    assert "cost_drag_per_year_pct" in out


def test_build_attribution_report(holdings_and_prices):
    holdings, prices = holdings_and_prices
    daily = pd.DataFrame({
        "date": holdings["date"].unique(),
        "cash_weight": 0.0,
        "turnover": 0.01,
    })
    rep = build_attribution_report(holdings, daily, prices, tcost_bps=10.0)
    assert "top_contributors" in rep
    assert "concentration" in rep
    assert "dollar_volume_exposure" in rep
    assert "cash_turnover_summary" in rep
