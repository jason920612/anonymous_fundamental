"""Phase 62: cross-disciplinary feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.features.cross_disciplinary_features import (
    CROSS_FEATURE_IDS,
    CrossFeatureConfig,
    attach_cross_features_to_samples,
    compute_cross_features,
)


def _build_price_panel(n_days: int, n_assets: int, vol: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((n_days, n_assets)) * vol
    prices = np.exp(np.cumsum(rets, axis=0))
    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates,
                        columns=[f"a{i}" for i in range(n_assets)])


def test_cross_features_have_expected_columns():
    panel = _build_price_panel(500, 30, vol=0.01, seed=0)
    feat = compute_cross_features(panel)
    for col in CROSS_FEATURE_IDS:
        assert col in feat.columns


def test_cross_features_finite_for_recent_dates():
    panel = _build_price_panel(800, 30, vol=0.01, seed=1)
    feat = compute_cross_features(panel)
    # Last 100 dates should mostly have finite values
    recent = feat.tail(100)
    for col in CROSS_FEATURE_IDS:
        non_nan = recent[col].dropna()
        assert len(non_nan) > 50  # at least half are computed


def test_attach_cross_features_to_samples_keeps_row_count():
    panel = _build_price_panel(800, 30, vol=0.01, seed=2)
    feat = compute_cross_features(panel)
    samples = pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "entry_date": [pd.Timestamp("2012-01-04"), pd.Timestamp("2013-06-01")],
    })
    out = attach_cross_features_to_samples(samples, feat)
    assert len(out) == 2
    for col in CROSS_FEATURE_IDS:
        assert col in out.columns
        assert f"{col}_missing" in out.columns


def test_kuramoto_r_in_unit_interval():
    panel = _build_price_panel(400, 20, vol=0.02, seed=3)
    feat = compute_cross_features(panel)
    r = feat["f_kuramoto_r"].dropna()
    assert (r >= 0).all() and (r <= 1.001).all()
