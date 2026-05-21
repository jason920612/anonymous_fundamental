"""Phase 60-61: Gutenberg-Richter + Kuramoto tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.portfolio.gutenberg_richter import (
    GutenbergRichterConfig,
    _foreshock_count,
    compute_gr_scale,
)
from afp.portfolio.kuramoto import (
    KuramotoConfig,
    _kuramoto_r,
    compute_kuramoto_r_series,
    compute_kuramoto_scale,
)


def test_gr_disabled_returns_one():
    rets = pd.Series([0.001] * 400)
    assert compute_gr_scale(rets, GutenbergRichterConfig(enabled=False)) == 1.0


def test_gr_calm_series_returns_high_scale():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.standard_normal(400) * 0.01)
    cfg = GutenbergRichterConfig(enabled=True, count_window=20, history_window=200,
                                  blend=0.0)
    scale = compute_gr_scale(rets, cfg, prev_scale=1.0)
    assert scale > 0.5  # should not collapse on stationary noise


def test_gr_recent_cluster_reduces_scale():
    rng = np.random.default_rng(1)
    rets = list(rng.standard_normal(350) * 0.01)
    # Append a recent cluster of large negative drawdowns
    rets += list(-np.abs(rng.standard_normal(30)) * 0.05)
    cfg = GutenbergRichterConfig(enabled=True, count_window=30, history_window=200,
                                  scale_floor=0.3, blend=0.0)
    scale_recent = compute_gr_scale(pd.Series(rets), cfg, prev_scale=1.0)
    # The cluster should reduce scale somewhat
    assert scale_recent < 1.0


def test_kuramoto_r_unsynchronized_low():
    rng = np.random.default_rng(0)
    # 100 random returns ~ N(0, σ), σ=0.01
    ret = rng.standard_normal(100) * 0.01
    sigma = np.full(100, 0.01)
    r = _kuramoto_r(ret, sigma)
    assert r < 0.5


def test_kuramoto_r_synchronized_high():
    # All assets move the same direction (+1σ)
    ret = np.full(100, 0.01)
    sigma = np.full(100, 0.01)
    r = _kuramoto_r(ret, sigma)
    assert r > 0.9


def test_kuramoto_series_in_unit_interval():
    rng = np.random.default_rng(0)
    n_days, n_assets = 300, 30
    rets = rng.standard_normal((n_days, n_assets)) * 0.01
    prices = np.exp(np.cumsum(rets, axis=0))
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    panel = pd.DataFrame(prices, index=dates,
                         columns=[f"a{i}" for i in range(n_assets)])
    r = compute_kuramoto_r_series(panel, sigma_window=30).dropna()
    assert (r >= 0).all() and (r <= 1).all()
