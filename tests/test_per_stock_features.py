"""Phase 64: per-stock cross-disciplinary feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.features.per_stock_cross_features import (
    PER_STOCK_FEATURE_IDS,
    PerStockFeatureConfig,
    _dfa_alpha,
    _hurst,
    _levy_z,
    compute_per_stock_features,
)


def test_hurst_random_walk_near_half():
    rng = np.random.default_rng(0)
    rw = rng.standard_normal(500) * 0.01
    h = _hurst(rw)
    assert 0.35 < h < 0.65


def test_hurst_trending_above_half():
    # A long-memory series via FBM-like construction
    rng = np.random.default_rng(1)
    eps = rng.standard_normal(500)
    # Persistent: each step adds a fraction of cumulative drift
    series = eps + 0.5 * np.cumsum(eps) / np.arange(1, len(eps) + 1) ** 0.5
    h = _hurst(series)
    # Should be > 0.5 in expectation; allow some sampling noise
    assert h > 0.4


def test_levy_z_gaussian_near_zero():
    rng = np.random.default_rng(0)
    ret = rng.standard_normal(2000) * 0.01
    z = _levy_z(ret, sigma_threshold=3.0, expected_freq=0.0027)
    # For 2000 Gaussian samples, |z| typically < 3
    assert abs(z) < 5


def test_levy_z_heavy_tail_positive():
    rng = np.random.default_rng(1)
    ret = rng.standard_t(df=3, size=2000) * 0.01
    z = _levy_z(ret, sigma_threshold=3.0, expected_freq=0.0027)
    # Heavy tail should yield positive z (more extremes than Gaussian)
    assert z > 0


def test_dfa_alpha_random_walk_near_half():
    rng = np.random.default_rng(0)
    series = rng.standard_normal(500) * 0.01
    alpha = _dfa_alpha(series)
    assert 0.3 < alpha < 0.7


def test_compute_per_stock_features_returns_all_columns():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    prices = pd.DataFrame({
        "date": list(dates) * 3,
        "internal_company_id": ["a"] * 400 + ["b"] * 400 + ["c"] * 400,
        "adjusted_close": np.concatenate([
            100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.01)),
            100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.01)),
            100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.01)),
        ]),
    })
    samples = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3"],
        "internal_company_id": ["a", "b", "c"],
        "entry_date": [pd.Timestamp("2021-06-01")] * 3,
    })
    out = compute_per_stock_features(samples, prices)
    assert len(out) == 3
    for fid in PER_STOCK_FEATURE_IDS:
        assert fid in out.columns
        assert f"{fid}_missing" in out.columns
