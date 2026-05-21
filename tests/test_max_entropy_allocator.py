"""Phase 40: maximum-entropy allocator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.max_entropy_allocator import max_entropy_allocation


def _candidates(signals: list[float], vols: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "internal_company_id": [f"a{i}" for i in range(len(signals))],
        "predicted_signal": signals,
        "positive_signal": [max(0.0, s) for s in signals],
        "vol_estimate": vols,
    })


def test_uniform_signal_gives_equal_weights():
    cands = _candidates([0.5] * 10, [0.02] * 10)
    cfg = PortfolioConfig(target_positions=10, max_single_stock_weight=0.5,
                          min_position_weight=0.0)
    out = max_entropy_allocation(cands, cfg, inverse_vol_blend=0.0)
    w = out.weights["weight"].to_numpy()
    assert abs(w.sum() - 1.0) < 1e-6
    assert w.std() < 0.01


def test_higher_signal_gets_higher_weight():
    cands = _candidates([0.1, 0.3, 0.5, 0.7, 0.9], [0.02] * 5)
    cfg = PortfolioConfig(target_positions=3, max_single_stock_weight=0.6,
                          min_position_weight=0.0)
    out = max_entropy_allocation(cands, cfg, inverse_vol_blend=0.0)
    weights = out.weights.sort_values("internal_company_id")["weight"].to_numpy()
    # a0 (lowest signal) < a4 (highest)
    assert weights[0] < weights[4]


def test_effective_n_matches_target():
    n = 20
    signals = list(np.linspace(0.1, 0.9, n))
    cands = _candidates(signals, [0.02] * n)
    cfg = PortfolioConfig(target_positions=10, max_single_stock_weight=0.5,
                          min_position_weight=0.0)
    out = max_entropy_allocation(cands, cfg, inverse_vol_blend=0.0)
    w = out.weights["weight"].to_numpy()
    n_eff = 1.0 / (w ** 2).sum()
    # Should be within 25% of target (binary-search tolerance + cap rounding)
    assert 7 <= n_eff <= 14


def test_empty_input_returns_full_cash():
    cfg = PortfolioConfig()
    out = max_entropy_allocation(pd.DataFrame(), cfg)
    assert out.weights.empty
    assert out.cash_weight == 1.0
