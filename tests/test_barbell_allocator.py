"""Phase 53: barbell allocator tests."""

from __future__ import annotations

import pandas as pd

from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.barbell_allocator import barbell_allocation


def _candidates(signals):
    n = len(signals)
    return pd.DataFrame({
        "internal_company_id": [f"a{i}" for i in range(n)],
        "predicted_signal": signals,
        "positive_signal": [max(0.0, s) for s in signals],
        "vol_estimate": [0.02] * n,
    })


def test_barbell_weights_sum_to_one():
    cands = _candidates([0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.7, 0.8])
    cfg = PortfolioConfig(min_positive_signal=0.0, target_positions=8,
                           max_positions=8, max_single_stock_weight=0.5,
                           min_position_weight=0.0)
    out = barbell_allocation(cands, cfg, safe_fraction=0.8, concentrated_top_k=3)
    assert abs(out.weights["weight"].sum() - 1.0) < 1e-6


def test_top_k_get_higher_weight():
    cands = _candidates([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cfg = PortfolioConfig(min_positive_signal=0.0, target_positions=8,
                           max_positions=8, max_single_stock_weight=0.5,
                           min_position_weight=0.0)
    out = barbell_allocation(cands, cfg, safe_fraction=0.8, concentrated_top_k=2)
    # Top 2 (signals 0.8, 0.7) should have higher weights than the bottom 2.
    w = out.weights.sort_values("predicted_signal", ascending=False)["weight"].to_numpy()
    assert w[0] > w[-1]
    assert w[1] > w[-1]


def test_empty_returns_full_cash():
    cfg = PortfolioConfig()
    out = barbell_allocation(pd.DataFrame(), cfg)
    assert out.weights.empty
    assert out.cash_weight == 1.0
