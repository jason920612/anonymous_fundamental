"""Pair anonymized feature tensors with sample targets for model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TabularDataset:
    """Flat-matrix dataset for tree / linear baselines."""

    X: np.ndarray              # [N, P*F + P*F + 3*P]   (values + missing flags + metadata)
    y: np.ndarray              # [N]
    sample_ids: np.ndarray     # [N]
    feature_columns: list[str]


def build_tabular_dataset(scaled: np.ndarray, missing: np.ndarray,
                          metadata: np.ndarray, sample_ids: list[str],
                          targets: pd.Series, feature_ids: list[str]) -> TabularDataset:
    """Flatten `[N,P,F]` into `[N, P*F*2 + P*3]` matrix and join with targets."""
    N, P, F = scaled.shape
    flat_values = scaled.reshape(N, P * F)
    flat_missing = missing.reshape(N, P * F)
    flat_meta = metadata.reshape(N, P * 3)
    X = np.concatenate([flat_values, flat_missing, flat_meta], axis=1)

    columns: list[str] = []
    for f, fid in enumerate(feature_ids):
        for p in range(P):
            columns.append(f"{fid}_o{p}_value")
    for f, fid in enumerate(feature_ids):
        for p in range(P):
            columns.append(f"{fid}_o{p}_missing")
    for m, name in enumerate(("meta_filing_delay_days", "meta_fiscal_period_index",
                              "meta_months_since_period_end")):
        for p in range(P):
            columns.append(f"{name}_o{p}")

    sid_arr = np.array(sample_ids)
    y = targets.reindex(sid_arr).to_numpy()
    # Filter out NaN-target rows (e.g. missing_ex_ante_scale).
    keep = ~np.isnan(y)
    return TabularDataset(
        X=X[keep], y=y[keep], sample_ids=sid_arr[keep], feature_columns=columns,
    )
