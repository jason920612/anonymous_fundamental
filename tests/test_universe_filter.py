"""Phase 12 — sample-level point-in-time universe filter."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from afp.features.universe_filter import (
    SampleUniverseConfig,
    apply_sample_universe_filter,
)


def _make_prices(icid: str, dates: pd.DatetimeIndex, base_price: float, volume: int):
    rng = np.random.default_rng(hash(icid) & 0xFFFF)
    returns = rng.normal(0.0005, 0.01, len(dates))
    px = base_price * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "date": [d.date() for d in dates],
        "internal_company_id": icid,
        "ticker_at_date": icid,
        "adjusted_close": px,
        "volume": volume,
    })


@pytest.fixture
def world():
    dates = pd.bdate_range("2018-01-02", "2024-12-31")
    high_liq = _make_prices("HIGH", dates, 100.0, 5_000_000)         # passes all filters
    # Force LOW to stay under $5 by giving it a clear downward drift.
    rng_low = np.random.default_rng(7)
    log_ret_low = rng_low.normal(-0.0008, 0.005, len(dates))
    px_low = 4.0 * np.exp(np.cumsum(log_ret_low))
    low_price = pd.DataFrame({
        "date": [d.date() for d in dates], "internal_company_id": "LOW",
        "ticker_at_date": "LOW", "adjusted_close": px_low, "volume": 5_000_000,
    })
    low_adv = _make_prices("ILLIQ", dates, 100.0, 1_000)             # ADV ~ $100K
    short_hist = _make_prices("NEW", dates[-200:], 100.0, 5_000_000) # < min history
    prices = pd.concat([high_liq, low_price, low_adv, short_hist], ignore_index=True)
    samples = pd.DataFrame([
        {"sample_id": "S_HIGH", "internal_company_id": "HIGH", "entry_date": date(2023, 6, 1)},
        {"sample_id": "S_LOW", "internal_company_id": "LOW", "entry_date": date(2023, 6, 1)},
        {"sample_id": "S_ILLIQ", "internal_company_id": "ILLIQ", "entry_date": date(2023, 6, 1)},
        {"sample_id": "S_NEW", "internal_company_id": "NEW", "entry_date": date(2024, 11, 1)},
    ])
    return samples, prices


def test_filter_disabled_returns_original(world):
    samples, prices = world
    cfg = SampleUniverseConfig(enabled=False)
    kept, dropped = apply_sample_universe_filter(samples, prices, cfg)
    assert len(kept) == 4 and dropped.empty


def test_filter_drops_low_price(world):
    samples, prices = world
    cfg = SampleUniverseConfig(enabled=True, min_adjusted_close_price_usd=5.0,
                               min_avg_daily_dollar_volume_usd=0.0,
                               min_years_price_history=0.0)
    kept, dropped = apply_sample_universe_filter(samples, prices, cfg)
    kept_ids = set(kept["sample_id"])
    assert "S_LOW" not in kept_ids
    assert "S_HIGH" in kept_ids
    assert (dropped["exclusion_reason"] == "filter_below_min_price").any()


def test_filter_drops_low_adv(world):
    samples, prices = world
    cfg = SampleUniverseConfig(enabled=True, min_adjusted_close_price_usd=0.0,
                               min_avg_daily_dollar_volume_usd=1_000_000.0,
                               min_years_price_history=0.0)
    kept, dropped = apply_sample_universe_filter(samples, prices, cfg)
    assert "S_ILLIQ" not in set(kept["sample_id"])
    assert (dropped["exclusion_reason"] == "filter_below_min_adv").any()


def test_filter_drops_short_history(world):
    samples, prices = world
    cfg = SampleUniverseConfig(enabled=True, min_adjusted_close_price_usd=0.0,
                               min_avg_daily_dollar_volume_usd=0.0,
                               min_years_price_history=3.0)
    kept, dropped = apply_sample_universe_filter(samples, prices, cfg)
    assert "S_NEW" not in set(kept["sample_id"])
    assert (dropped["exclusion_reason"] == "filter_insufficient_history").any()


def test_filter_uses_only_past_data(world):
    """Poisoning prices on/after entry_date must not change which samples are kept."""
    samples, prices = world
    cfg = SampleUniverseConfig(enabled=True, min_adjusted_close_price_usd=5.0,
                               min_avg_daily_dollar_volume_usd=1_000_000.0,
                               min_years_price_history=1.0)
    a_kept, _ = apply_sample_universe_filter(samples, prices, cfg)

    px_poisoned = prices.copy()
    px_poisoned["date"] = pd.to_datetime(px_poisoned["date"])
    earliest = min(samples["entry_date"])
    mask = px_poisoned["date"] >= pd.Timestamp(earliest)
    px_poisoned.loc[mask, "adjusted_close"] = 0.01  # would fail price filter
    px_poisoned.loc[mask, "volume"] = 0             # would fail ADV filter
    px_poisoned["date"] = px_poisoned["date"].dt.date

    b_kept, _ = apply_sample_universe_filter(samples, px_poisoned, cfg)
    assert set(a_kept["sample_id"]) == set(b_kept["sample_id"])
