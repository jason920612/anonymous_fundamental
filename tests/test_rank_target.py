"""Phase 13 — cross-sectional rank target."""

import numpy as np
import pandas as pd
import pytest

from afp.targets.target_transform import TargetConfig, apply


def _make_samples(n_per_q: int = 20, n_q: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        entry = pd.Timestamp(f"2020-{(q % 4) + 1:02d}-01") + pd.DateOffset(months=q // 4 * 12)
        for i in range(n_per_q):
            rows.append({
                "sample_id": f"S_{q}_{i}",
                "entry_date": entry.date(),
                "raw_log_return": float(rng.normal(0.0, 0.1)),
                "ex_ante_scale": 0.15,
                "target_normalized_signal": np.nan,
                "eligible_for_training": True,
                "exclusion_reason": None,
            })
    return pd.DataFrame(rows)


def test_rank_target_is_in_unit_interval():
    samples = _make_samples()
    out = apply(samples, TargetConfig(target_mode="cross_sectional_rank"))
    vals = out["target_normalized_signal"].dropna()
    assert (vals >= -1.0).all() and (vals <= 1.0).all()


def test_rank_target_monotonic_with_realized():
    """Within a quarter, higher raw_log_return ⇒ higher rank target."""
    samples = _make_samples(n_per_q=50, n_q=1)
    out = apply(samples, TargetConfig(target_mode="cross_sectional_rank"))
    sub = out.dropna(subset=["target_normalized_signal"]).sort_values("raw_log_return")
    assert (sub["target_normalized_signal"].diff().dropna() >= -1e-9).all()


def test_rank_target_centered_per_quarter():
    """Mean rank target per quarter should be ~0 (range [-1,1], uniform pct_rank)."""
    samples = _make_samples(n_per_q=50, n_q=4)
    out = apply(samples, TargetConfig(target_mode="cross_sectional_rank"))
    means = out.dropna(subset=["target_normalized_signal"]).groupby(
        pd.to_datetime(out.dropna(subset=["target_normalized_signal"])["entry_date"]).dt.to_period("Q")
    )["target_normalized_signal"].mean()
    assert (means.abs() < 0.05).all()


def test_tanh_mode_unchanged_default():
    samples = _make_samples(n_per_q=10, n_q=2)
    out = apply(samples, TargetConfig(k=2.5, target_mode="tanh_scaled"))
    vals = out["target_normalized_signal"].dropna()
    assert (vals >= -1.0).all() and (vals <= 1.0).all()
    # Sanity: with raw=0, target=0
    near_zero = samples["raw_log_return"].abs().idxmin()
    assert abs(out.loc[near_zero, "target_normalized_signal"]) < 0.5
