"""Phase 7 tests — constraints, cash handling, negative-signal exclusion."""

import numpy as np
import pandas as pd
import pytest

from afp.portfolio.allocation import (
    PortfolioConfig,
    inverse_vol_allocation,
    select_candidates,
)
from afp.portfolio.signal_restore import add_restored_returns


def _df(rows):
    return pd.DataFrame(rows)


def test_negative_signals_excluded():
    cfg = PortfolioConfig(min_positive_signal=0.05)
    preds = _df([
        {"internal_company_id": "A", "predicted_signal":  0.30, "vol_estimate": 0.02},
        {"internal_company_id": "B", "predicted_signal": -0.40, "vol_estimate": 0.02},
        {"internal_company_id": "C", "predicted_signal":  0.10, "vol_estimate": 0.02},
        {"internal_company_id": "D", "predicted_signal":  0.02, "vol_estimate": 0.02},  # below threshold
    ])
    cands = select_candidates(preds, cfg)
    assert set(cands["internal_company_id"]) == {"A", "C"}


def test_inverse_vol_weights_sum_to_at_most_one_and_respect_caps():
    cfg = PortfolioConfig(min_positive_signal=0.0, max_single_stock_weight=0.20,
                          min_position_weight=0.0, allow_cash=False)
    preds = _df([
        {"internal_company_id": f"S{i}", "predicted_signal": 0.5, "vol_estimate": 0.01 + 0.005 * i}
        for i in range(8)
    ])
    cands = select_candidates(preds, cfg)
    res = inverse_vol_allocation(cands, cfg)
    assert res.weights["weight"].max() <= 0.20 + 1e-9
    assert res.weights["weight"].sum() <= 1.0 + 1e-9


def test_allow_cash_when_few_candidates():
    cfg = PortfolioConfig(min_positive_signal=0.05, max_single_stock_weight=0.05,
                          min_positions=30, allow_cash=True)
    preds = _df([{"internal_company_id": "A", "predicted_signal": 0.5, "vol_estimate": 0.02}])
    cands = select_candidates(preds, cfg)
    res = inverse_vol_allocation(cands, cfg)
    assert res.cash_weight > 0


def test_min_position_weight_drops_tiny_positions():
    cfg = PortfolioConfig(min_positive_signal=0.0, max_single_stock_weight=0.5,
                          min_position_weight=0.10, allow_cash=True)
    preds = _df([
        {"internal_company_id": "A", "predicted_signal": 0.90, "vol_estimate": 0.01},
        {"internal_company_id": "B", "predicted_signal": 0.05, "vol_estimate": 0.20},  # tiny
    ])
    cands = select_candidates(preds, cfg)
    res = inverse_vol_allocation(cands, cfg)
    assert set(res.weights["internal_company_id"]) == {"A"}


def test_add_restored_returns_inverts_signal():
    preds = pd.DataFrame({
        "internal_company_id": ["X"],
        "predicted_signal": [0.5],
        "ex_ante_scale": [0.2],
    })
    out = add_restored_returns(preds, k=2.5)
    # restored = atanh(0.5) * 2.5 * 0.2
    assert out["restored_expected_return"].iloc[0] == pytest.approx(
        np.arctanh(0.5) * 2.5 * 0.2, rel=1e-9)
