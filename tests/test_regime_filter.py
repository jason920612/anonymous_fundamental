"""Phase 39: cross-sectional dispersion regime filter tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.portfolio.regime_filter import (
    RegimeFilterConfig,
    compute_regime_scales,
    cross_sectional_temperature,
)


def _build_price_panel(n_days: int, n_assets: int, vol: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((n_days, n_assets)) * vol
    prices = np.exp(np.cumsum(rets, axis=0))
    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates,
                        columns=[f"a{i}" for i in range(n_assets)])


def test_temperature_higher_for_higher_dispersion():
    cold = _build_price_panel(300, 20, vol=0.005, seed=0)
    hot = _build_price_panel(300, 20, vol=0.05, seed=0)
    T_cold = cross_sectional_temperature(cold).dropna().mean()
    T_hot = cross_sectional_temperature(hot).dropna().mean()
    assert T_hot > 5 * T_cold


def test_disabled_returns_unit_scale():
    panel = _build_price_panel(300, 10, vol=0.02, seed=1)
    T = cross_sectional_temperature(panel)
    cfg = RegimeFilterConfig(enabled=False)
    s = compute_regime_scales(T, cfg)
    assert (s == 1.0).all()


def test_cold_regime_scales_down():
    """A trailing high-T period followed by a sudden cold day should drop the scale."""
    rng = np.random.default_rng(7)
    hot_rets = rng.standard_normal((400, 15)) * 0.05
    cold_rets = rng.standard_normal((40, 15)) * 0.001
    rets = np.vstack([hot_rets, cold_rets])
    prices = np.exp(np.cumsum(rets, axis=0))
    dates = pd.date_range("2010-01-01", periods=len(prices), freq="B")
    panel = pd.DataFrame(prices, index=dates,
                         columns=[f"a{i}" for i in range(15)])
    T = cross_sectional_temperature(panel)
    cfg = RegimeFilterConfig(enabled=True, lookback_days=252, boltzmann_kt=2.0)
    s = compute_regime_scales(T, cfg)
    hot_tail = s.iloc[395:400].mean()
    cold_tail = s.iloc[-5:].mean()
    assert cold_tail < hot_tail


def test_scale_clipped_to_floor_and_ceiling():
    rng = np.random.default_rng(11)
    rets = rng.standard_normal((500, 10)) * 0.02
    prices = np.exp(np.cumsum(rets, axis=0))
    dates = pd.date_range("2010-01-01", periods=500, freq="B")
    panel = pd.DataFrame(prices, index=dates,
                         columns=[f"a{i}" for i in range(10)])
    T = cross_sectional_temperature(panel)
    cfg = RegimeFilterConfig(enabled=True, lookback_days=100,
                             scale_floor=0.4, scale_ceiling=0.95)
    s = compute_regime_scales(T, cfg)
    assert s.min() >= 0.4 - 1e-9
    assert s.max() <= 0.95 + 1e-9
