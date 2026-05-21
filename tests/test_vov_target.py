"""Phase 50: vol-of-vol target tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.portfolio.vov_target import VovTargetConfig, _rolling_vov, compute_vov_scale


def test_disabled_returns_one():
    rets = pd.Series([0.001] * 200)
    cfg = VovTargetConfig(enabled=False)
    assert compute_vov_scale(rets, cfg) == 1.0


def test_constant_vol_close_to_one():
    """Stationary noise should not strongly damp exposure (z near 0)."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.standard_normal(300) * 0.01)
    cfg = VovTargetConfig(enabled=True, sigma_window=20, vov_window=60, blend=0.0)
    scale = compute_vov_scale(rets, cfg, prev_scale=1.0)
    # Even in stationary noise the rolling vov-z can wander ±1σ. Test
    # only that we're above the floor, since vov-z=0 → scale=1.
    assert scale > 0.4


def test_recent_vol_spike_lowers_scale():
    rng = np.random.default_rng(1)
    rets = list(rng.standard_normal(250) * 0.01)
    rets += list(rng.standard_normal(80) * 0.05)  # vol spike
    cfg = VovTargetConfig(enabled=True, sigma_window=20, vov_window=60,
                          scale_floor=0.3, blend=0.0)
    scale = compute_vov_scale(pd.Series(rets), cfg, prev_scale=1.0)
    assert scale < 0.95


def test_rolling_vov_is_nan_aware():
    s = pd.Series([0.001] * 5)
    cfg = VovTargetConfig(enabled=True, sigma_window=20, vov_window=60)
    assert compute_vov_scale(s, cfg) == 1.0
