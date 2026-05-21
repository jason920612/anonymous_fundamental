"""Phase 44: drawdown-targeting wrapper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.portfolio.dd_target import (
    DrawdownTargetConfig,
    _realized_drawdown,
    compute_dd_scale,
)


def test_no_drawdown_keeps_full_exposure():
    rets = pd.Series([0.001] * 100)
    cfg = DrawdownTargetConfig(enabled=True, dd_trigger=0.10,
                               scale_floor=0.3, blend=0.0)
    scale = compute_dd_scale(rets, cfg, prev_scale=1.0)
    assert scale == 1.0


def test_drawdown_below_trigger_keeps_full_exposure():
    # 5% drawdown (below 10% trigger)
    rets = pd.Series([-0.0005] * 100)
    cfg = DrawdownTargetConfig(enabled=True, dd_trigger=0.10,
                               scale_floor=0.3, blend=0.0)
    dd = _realized_drawdown(rets, 100)
    assert -0.06 < dd < -0.03
    scale = compute_dd_scale(rets, cfg, prev_scale=1.0)
    assert scale == 1.0


def test_drawdown_above_trigger_reduces_exposure():
    # 20% drawdown — past trigger
    rets = pd.Series([-0.002] * 100)
    cfg = DrawdownTargetConfig(enabled=True, dd_trigger=0.10, alpha=1.0,
                               scale_floor=0.3, blend=0.0)
    dd = _realized_drawdown(rets, 100)
    assert dd < -0.10
    scale = compute_dd_scale(rets, cfg, prev_scale=1.0)
    assert scale < 1.0
    assert scale >= 0.3


def test_disabled_returns_one():
    rets = pd.Series([-0.002] * 100)
    cfg = DrawdownTargetConfig(enabled=False)
    assert compute_dd_scale(rets, cfg) == 1.0
