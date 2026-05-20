"""Phase 21 — anonymous price-derived features."""

import numpy as np
import pandas as pd
import pytest

from afp.features.price_features import (
    PriceFeatureConfig,
    compute_price_features,
    fit_scaler,
    transform_with_scaler,
)


def _world():
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    rng = np.random.default_rng(0)
    rows = []
    for tic in ("AAA", "BBB"):
        px = 50.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(dates))))
        for i, d in enumerate(dates):
            rows.append({"date": d.date(), "internal_company_id": tic,
                         "adjusted_close": px[i], "volume": 1_000_000})
    prices = pd.DataFrame(rows)
    samples = pd.DataFrame([
        {"sample_id": "S1", "internal_company_id": "AAA", "entry_date": pd.Timestamp("2024-06-03").date()},
        {"sample_id": "S2", "internal_company_id": "BBB", "entry_date": pd.Timestamp("2024-06-03").date()},
    ])
    return samples, prices


def test_price_features_disabled_returns_no_columns():
    samples, prices = _world()
    df, ids = compute_price_features(samples, prices, PriceFeatureConfig(enabled=False))
    assert ids == []


def test_price_features_default_6_signals():
    samples, prices = _world()
    df, ids = compute_price_features(samples, prices, PriceFeatureConfig(enabled=True))
    # 4 horizons + relative momentum + rolling vol = 6
    assert len(ids) == 6
    assert all(fid.startswith("price_feature_") for fid in ids)
    assert df[ids].notna().all().all()


def test_price_features_only_use_past_prices():
    """Poisoning prices on/after entry_date must not change feature values."""
    samples, prices = _world()
    cfg = PriceFeatureConfig(enabled=True)
    a, _ = compute_price_features(samples, prices, cfg)

    px_poisoned = prices.copy()
    earliest = min(samples["entry_date"])
    mask = pd.to_datetime(px_poisoned["date"]) >= pd.Timestamp(earliest)
    px_poisoned.loc[mask, "adjusted_close"] = 1.0
    b, _ = compute_price_features(samples, px_poisoned, cfg)

    cols = [c for c in a.columns if c.startswith("price_feature_")]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_fit_scaler_then_transform_in_range():
    samples, prices = _world()
    df, ids = compute_price_features(samples, prices, PriceFeatureConfig(enabled=True))
    stats = fit_scaler(df, ids)
    scaled = transform_with_scaler(df, ids, stats, clip=(-3.0, 3.0))
    for fid in ids:
        assert scaled[fid].between(-3.0, 3.0).all()
        assert f"{fid}_missing" in scaled.columns


def test_no_human_readable_words_in_feature_ids():
    samples, prices = _world()
    _, ids = compute_price_features(samples, prices, PriceFeatureConfig(enabled=True))
    # All IDs are opaque price_feature_NNN — no momentum/return/return tokens.
    bad = {"momentum", "return", "vol", "trend", "rsi"}
    for fid in ids:
        for token in bad:
            assert token not in fid.lower()
