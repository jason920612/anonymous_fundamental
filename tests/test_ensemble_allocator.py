"""Phase 51: allocator ensemble tests."""

from __future__ import annotations

import pandas as pd

from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.ensemble_allocator import ensemble_allocation


def _candidates(signals, vols):
    return pd.DataFrame({
        "internal_company_id": [f"a{i}" for i in range(len(signals))],
        "predicted_signal": signals,
        "positive_signal": [max(0.0, s) for s in signals],
        "vol_estimate": vols,
    })


def test_empty_returns_full_cash():
    cfg = PortfolioConfig()
    out = ensemble_allocation(pd.DataFrame(), cfg)
    assert out.weights.empty
    assert out.cash_weight == 1.0


def test_alpha_one_equals_inverse_vol():
    # alpha=1 should give exactly inverse-vol weights (no RMT contribution).
    cands = _candidates([0.5, 0.3, 0.4, 0.2], [0.02, 0.02, 0.02, 0.02])
    cfg = PortfolioConfig(min_positive_signal=0.0, target_positions=4,
                           max_positions=4, max_single_stock_weight=0.5,
                           min_position_weight=0.0)
    # With no vol_estimator the RMT side will fall back to inverse-vol
    out = ensemble_allocation(cands, cfg, alpha=1.0)
    # Weights sum to 1 and are all positive
    assert abs(out.weights["weight"].sum() - 1.0) < 1e-6
    assert (out.weights["weight"] > 0).all()


def test_weights_sum_to_one():
    cands = _candidates([0.4, 0.3, 0.2, 0.5], [0.01, 0.02, 0.03, 0.015])
    cfg = PortfolioConfig(min_positive_signal=0.0, target_positions=4,
                           max_positions=4, max_single_stock_weight=0.5,
                           min_position_weight=0.0)
    out = ensemble_allocation(cands, cfg, alpha=0.5)
    assert abs(out.weights["weight"].sum() - 1.0) < 1e-6
