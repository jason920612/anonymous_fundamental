"""Phase 4 tests — scale, target transform, atanh inverse, no-future-price gates."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from afp.features.sample_builder import build_event_samples
from afp.targets.target_transform import TargetConfig, apply, restore
from afp.targets.volatility_scale import (
    ScaleConfig,
    compute_ex_ante_scale_for_samples,
    trailing_daily_vol,
    trailing_event_vol,
)
from tests.fixtures import make_synthetic_world


@pytest.fixture(scope="module")
def world():
    return make_synthetic_world(2012, 2017)


@pytest.fixture(scope="module")
def samples(world):
    samples, _ = build_event_samples(world["filings"], world["prices"], world["calendar"],
                                     world["benchmarks"])
    return samples


def test_trailing_daily_vol_uses_only_prior_dates():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    cutoff = idx[200].date()
    v_full = trailing_daily_vol(s, cutoff, 100)
    s_noise = s.copy()
    s_noise.loc[s_noise.index >= pd.Timestamp(cutoff)] = 999.0  # future values polluted
    v_polluted = trailing_daily_vol(s_noise, cutoff, 100)
    assert v_full == pytest.approx(v_polluted)


def test_trailing_event_vol_requires_minimum_history():
    s = pd.Series([0.01, -0.02, 0.03])
    assert trailing_event_vol(s, 8, 4) is None
    s = pd.Series([0.01, -0.02, 0.03, 0.0, 0.02])
    assert trailing_event_vol(s, 8, 4) is not None


def test_compute_ex_ante_scale_uses_only_past_prices(world, samples):
    """Polluting future prices must not change ex_ante_scale for any sample."""
    cfg = ScaleConfig()
    scaled = compute_ex_ante_scale_for_samples(
        samples, world["prices"], world["calendar"], cfg
    )
    # Poison prices strictly after the latest entry_date — that's "future" relative
    # to every sample, so a leak-free scale calculation must ignore them.
    latest_entry = pd.to_datetime(samples["entry_date"]).max().date()
    px_poisoned = world["prices"].copy()
    mask = px_poisoned["date"] > latest_entry
    px_poisoned.loc[mask, "adjusted_close"] = (
        px_poisoned.loc[mask, "adjusted_close"].to_numpy() * 100.0
    )
    poisoned = compute_ex_ante_scale_for_samples(samples, px_poisoned, world["calendar"], cfg)
    # Scale must be unchanged because we only use d < entry_date
    a = scaled.set_index("sample_id")["ex_ante_scale"].dropna()
    b = poisoned.set_index("sample_id")["ex_ante_scale"].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 0
    diff = (a.loc[common] - b.loc[common]).abs()
    assert diff.max() < 1e-9


def test_target_in_unit_interval(world, samples):
    cfg = ScaleConfig()
    scaled = compute_ex_ante_scale_for_samples(samples, world["prices"], world["calendar"], cfg)
    with_target = apply(scaled, TargetConfig(k=2.5))
    eligible = with_target["target_normalized_signal"].dropna()
    assert (eligible >= -1.0).all() and (eligible <= 1.0).all()


def test_missing_scale_flagged(world, samples):
    cfg = ScaleConfig(daily_vol_lookback_days=99999)  # always insufficient
    scaled = compute_ex_ante_scale_for_samples(samples, world["prices"], world["calendar"], cfg)
    with_target = apply(scaled, TargetConfig(k=2.5))
    assert (with_target["exclusion_reason"] == "missing_ex_ante_scale").any()
    assert (with_target.loc[with_target["exclusion_reason"] == "missing_ex_ante_scale",
                            "eligible_for_training"] == False).all()


def test_atanh_restore_inverts_tanh():
    cfg = TargetConfig(k=2.5)
    for raw in (-0.30, 0.0, 0.42):
        scale = 0.20
        signal = math.tanh(raw / (cfg.k * scale))
        restored = restore(signal, scale, cfg)
        assert restored == pytest.approx(raw, abs=1e-6)
