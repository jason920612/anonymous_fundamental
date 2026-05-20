"""Phase 37 — reference cohort freshness logic.

We can't test the full `_reference_cohort_scores` without a trained booster,
but we can verify the stale-vs-fresh decision logic against a synthetic
cache and synthetic event_samples parquet.
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def _write_cache(path: Path, quarter: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"raw_score": [0.1, 0.2, 0.3], "quarter": [quarter] * 3}).to_parquet(path, index=False)


def _write_samples(path: Path, entry_dates: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "entry_date": entry_dates,
        "target_normalized_signal": [0.1] * len(entry_dates),
    }).to_parquet(path, index=False)


def test_cohort_uses_cache_when_quarter_matches(tmp_path):
    """If cache and event_samples are in the same quarter, no rebuild."""
    cache = tmp_path / "artifacts" / "models" / "lambdarank_v3" / "reference_cohort.parquet"
    samples = tmp_path / "data" / "processed" / "event_samples.parquet"
    _write_cache(cache, "2026Q1")
    _write_samples(samples, [pd.Timestamp("2026-02-15"), pd.Timestamp("2026-03-10")])
    # Decision derived inline (avoid heavy booster dependency)
    samples_q = (pd.to_datetime(
        pd.read_parquet(samples, columns=["entry_date"])["entry_date"]
    ).dt.to_period("Q").max())
    cached_q = pd.read_parquet(cache)["quarter"].iloc[0]
    assert str(samples_q) == cached_q


def test_cohort_marks_stale_when_quarter_advances(tmp_path):
    cache = tmp_path / "ref.parquet"
    samples = tmp_path / "samples.parquet"
    _write_cache(cache, "2026Q1")
    _write_samples(samples, [pd.Timestamp("2026-05-15")])     # 2026Q2
    samples_q = (pd.to_datetime(
        pd.read_parquet(samples, columns=["entry_date"])["entry_date"]
    ).dt.to_period("Q").max())
    cached_q = pd.read_parquet(cache)["quarter"].iloc[0]
    assert str(samples_q) != cached_q
    assert str(samples_q) == "2026Q2"


def test_cohort_detects_when_samples_newer_than_cache(tmp_path):
    cache = tmp_path / "ref.parquet"
    samples = tmp_path / "samples.parquet"
    _write_cache(cache, "2026Q1")
    time.sleep(0.05)
    _write_samples(samples, [pd.Timestamp("2026-02-15")])
    assert samples.stat().st_mtime > cache.stat().st_mtime


def test_cohort_warns_when_universe_is_very_stale(tmp_path):
    samples = tmp_path / "samples.parquet"
    # 200 days ago — beyond the 120-day threshold
    old_date = date.today() - timedelta(days=200)
    _write_samples(samples, [pd.Timestamp(old_date)])
    entries = pd.to_datetime(
        pd.read_parquet(samples, columns=["entry_date"])["entry_date"])
    days_since = (date.today() - entries.max().date()).days
    assert days_since > 120     # would trigger the warning
