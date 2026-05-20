"""Phase 11 diagnostics tests."""

import numpy as np
import pandas as pd
import pytest

from afp.models.diagnostics import (
    build_diagnostic_report,
    decile_table,
    positive_signal_vs_random,
    quarterly_direction_accuracy,
    quarterly_rank_correlation,
    top_minus_bottom_spread,
)


def _make_signal_panel(seed: int, perfect: bool, n_dates: int = 60, n_per_date: int = 25):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="W")
    rows = []
    for d in dates:
        raw = rng.normal(0.0, 0.1, n_per_date)
        sig = np.tanh(raw / 0.25) if perfect else rng.normal(0.0, 0.3, n_per_date)
        target = np.tanh(raw / 0.25)
        for i in range(n_per_date):
            rows.append({
                "predicted_signal": sig[i],
                "raw_log_return": raw[i],
                "target_normalized_signal": target[i],
                "spy_log_return": 0.0,
                "entry_date": d,
                "split": "test",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def perfect_signal():
    """Signal that exactly equals raw_log_return — should produce monotonic deciles."""
    return _make_signal_panel(seed=0, perfect=True)


@pytest.fixture
def noise_signal():
    """Predictions independent of returns — should show flat decile table."""
    return _make_signal_panel(seed=1, perfect=False)


def test_decile_table_monotonic_for_perfect_signal(perfect_signal):
    table = decile_table(perfect_signal, n_buckets=10)
    means = table["raw_log_return_mean"].to_numpy()
    # Last decile must beat first decile by a clear margin
    assert means[-1] > means[0]
    # And should be monotonic (allow tiny inversions due to bin tie-breaking)
    diffs = np.diff(means)
    assert (diffs > 0).sum() >= 7   # at least 7 of 9 differences positive


def test_top_minus_bottom_spread_positive_for_perfect_signal(perfect_signal):
    spread = top_minus_bottom_spread(perfect_signal, n_buckets=10)
    assert spread > 0.05


def test_top_minus_bottom_spread_near_zero_for_noise(noise_signal):
    spread = top_minus_bottom_spread(noise_signal, n_buckets=10)
    assert abs(spread) < 0.05


def test_quarterly_rank_correlation_high_for_perfect_signal(perfect_signal):
    out = quarterly_rank_correlation(perfect_signal)
    assert (out["spearman"].dropna() > 0.5).mean() > 0.5


def test_quarterly_direction_accuracy_includes_quarter_column(perfect_signal):
    out = quarterly_direction_accuracy(perfect_signal)
    assert "quarter" in out.columns
    assert (out["direction_accuracy"] > 0.5).all()


def test_positive_signal_vs_random_detects_signal(perfect_signal):
    # n_holdings small relative to per-date population, so model's top-N is
    # distinguishable from a random N of positives.
    out = positive_signal_vs_random(perfect_signal, n_holdings=3)
    assert out["spread"] > 0
    assert out["model_mean"] > out["random_mean"]


def test_positive_signal_vs_random_near_zero_for_noise(noise_signal):
    out = positive_signal_vs_random(noise_signal, n_holdings=3)
    assert abs(out["spread"]) < 0.05


def test_build_diagnostic_report_packages_everything(noise_signal):
    rep = build_diagnostic_report(noise_signal)
    assert "decile_table" in rep
    assert "top_minus_bottom_log_return" in rep
    assert "quarterly_spearman" in rep
    assert "quarterly_direction_accuracy" in rep
    assert "positive_signal_vs_random" in rep
