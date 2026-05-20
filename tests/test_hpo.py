"""Phase 18 — time-based HPO honors test holdout."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from afp.models.hpo import time_series_folds


def _samples(n: int = 2000):
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2018-01-01")
    rows = []
    for i in range(n):
        entry = base + pd.Timedelta(days=int(rng.integers(0, 365 * 5)))
        rows.append({
            "sample_id": f"S{i}",
            "entry_date": entry.date(),
            "target_normalized_signal": float(rng.normal(0, 0.5)),
        })
    return pd.DataFrame(rows)


def test_time_series_folds_train_strictly_before_eval():
    samples = _samples()
    folds = time_series_folds(samples, n_folds=3, min_train_months=12)
    assert len(folds) >= 2
    for train_idx, eval_idx in folds:
        t_max = pd.to_datetime(samples.iloc[train_idx]["entry_date"]).max()
        e_min = pd.to_datetime(samples.iloc[eval_idx]["entry_date"]).min()
        assert t_max <= e_min


def test_time_series_folds_no_overlap():
    samples = _samples()
    folds = time_series_folds(samples, n_folds=3, min_train_months=12)
    for train_idx, eval_idx in folds:
        assert len(set(train_idx) & set(eval_idx)) == 0


def test_time_series_folds_train_grows_monotonically():
    samples = _samples()
    folds = time_series_folds(samples, n_folds=4, min_train_months=12)
    train_sizes = [len(train) for train, _ in folds]
    assert all(train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1))
