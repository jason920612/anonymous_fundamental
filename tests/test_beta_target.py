"""Phase 46: beta-target wrapper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.portfolio.beta_target import (
    BetaTargetConfig,
    _rolling_beta,
    compute_beta_scale,
)


def _series(values, start="2020-01-02"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_beta_one_when_strategy_is_market():
    rng = np.random.default_rng(0)
    m = rng.standard_normal(200) * 0.01
    s = _series(m)
    mkt = _series(m)
    beta = _rolling_beta(s, mkt, 100)
    assert abs(beta - 1.0) < 0.05


def test_beta_two_when_strategy_amplifies_market():
    rng = np.random.default_rng(1)
    m = rng.standard_normal(200) * 0.01
    s = _series(2.0 * m)
    mkt = _series(m)
    beta = _rolling_beta(s, mkt, 100)
    assert 1.8 < beta < 2.2


def test_scale_reduced_when_beta_high():
    rng = np.random.default_rng(2)
    m = rng.standard_normal(200) * 0.01
    s = _series(2.0 * m)
    mkt = _series(m)
    cfg = BetaTargetConfig(enabled=True, lookback_days=100, scale_floor=0.3, blend=0.0)
    scale = compute_beta_scale(s, mkt, cfg, prev_scale=1.0)
    # beta≈2.0 → scale ≈ 0.5
    assert 0.4 < scale < 0.6


def test_scale_unchanged_when_beta_below_one():
    rng = np.random.default_rng(3)
    m = rng.standard_normal(200) * 0.01
    s = _series(0.5 * m)
    mkt = _series(m)
    cfg = BetaTargetConfig(enabled=True, lookback_days=100, scale_floor=0.3, blend=0.0)
    scale = compute_beta_scale(s, mkt, cfg, prev_scale=1.0)
    assert scale == 1.0


def test_disabled_returns_one():
    s = _series([0.001] * 100)
    mkt = _series([0.001] * 100)
    cfg = BetaTargetConfig(enabled=False)
    assert compute_beta_scale(s, mkt, cfg) == 1.0
