"""Phase 18: time-based hyperparameter search.

Strict guarantees:
  - Search uses train + validation rows only. Test data is NEVER touched.
  - Folds are time-ordered (rolling origin). The k-th fold trains on samples
    with `entry_date <= fold_end[k-1]` and evaluates on the slice
    `fold_end[k-1] < entry_date <= fold_end[k]`.
  - The selected best params are fit once on `train ∪ validation` and the
    resulting model is what gets scored on test.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from afp.features.encoder import AnonymousFeatureEncoder
from afp.models.baselines import build_model
from afp.models.datasets import build_tabular_dataset
from afp.models.metrics import regression_metrics
from afp.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class HPOResult:
    best_params: dict[str, Any]
    best_metric: float
    trials: list[dict[str, Any]] = field(default_factory=list)


def time_series_folds(samples: pd.DataFrame, n_folds: int = 5,
                       min_train_months: int = 24) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of `(train_positional_idx, eval_positional_idx)` for each rolling fold.

    Indices are positional into `samples` (i.e. consumable by `samples.iloc[idx]` or
    `X[idx]` where `X` was built in the same row order). No sorting is required of the
    caller — the function computes masks on the input row order directly.
    """
    entry_ts = pd.to_datetime(samples["entry_date"])
    earliest, latest = entry_ts.min(), entry_ts.max()
    if pd.isna(earliest) or pd.isna(latest):
        return []

    start_cut = earliest + pd.Timedelta(days=min_train_months * 30)
    if start_cut >= latest:
        return []
    fold_ends = pd.date_range(start=start_cut, end=latest, periods=n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = fold_ends[i]
        eval_end = fold_ends[i + 1]
        train_mask = (entry_ts <= train_end).to_numpy()
        eval_mask = ((entry_ts > train_end) & (entry_ts <= eval_end)).to_numpy()
        if train_mask.sum() < 100 or eval_mask.sum() < 20:
            continue
        folds.append((np.where(train_mask)[0], np.where(eval_mask)[0]))
    return folds


def grid_search(
    model_kind: str,
    param_grid: dict[str, list[Any]],
    encoder: AnonymousFeatureEncoder,
    train_val_samples: pd.DataFrame,
    facts: pd.DataFrame,
    metric: str = "spearman",
    n_folds: int = 4,
    sample_weight_kwargs: dict[str, bool] | None = None,
) -> HPOResult:
    """Exhaustive grid search using rolling-origin TS CV on train+val only.

    `metric` is one of the keys in `regression_metrics` output. Higher = better
    for spearman/pearson/direction_accuracy/r2; lower = better for mse/mae.
    """
    feature_ids = encoder.artifact.feature_map.feature_ids
    if encoder.artifact.derived_enabled:
        feature_ids = (list(feature_ids)
                       + [f"derived_yoy_{fid}" for fid in feature_ids]
                       + [f"derived_z_{fid}" for fid in feature_ids])

    scaled, missing, meta, sids = encoder.transform(train_val_samples, facts)
    targets = train_val_samples.set_index("sample_id")["target_normalized_signal"]
    ds = build_tabular_dataset(scaled, missing, meta, sids, targets, feature_ids)
    samples_in = train_val_samples.set_index("sample_id").loc[ds.sample_ids].reset_index()
    folds = time_series_folds(samples_in, n_folds=n_folds)
    if not folds:
        raise RuntimeError("no folds available — extend training window or lower n_folds")

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    higher_better = metric not in {"mse", "mae"}
    best_metric = -math.inf if higher_better else math.inf
    best_params: dict[str, Any] = {}
    trials: list[dict[str, Any]] = []

    sw_kw = sample_weight_kwargs or {}

    for combo in combos:
        params = dict(zip(keys, combo))
        fold_metrics = []
        for train_idx, eval_idx in folds:
            model = build_model(model_kind, {"params": params} if model_kind == "gbt" else params)
            sw = None
            if sw_kw:
                from afp.models.train import compute_sample_weights
                sw = compute_sample_weights(samples_in, ds.sample_ids[train_idx], **sw_kw)
            model.fit(ds.X[train_idx], ds.y[train_idx], sample_weight=sw)
            pred = model.predict(ds.X[eval_idx])
            m = regression_metrics(ds.y[eval_idx], pred)
            fold_metrics.append(m[metric])
        avg = float(np.mean(fold_metrics))
        trials.append({"params": params, metric: avg, "fold_values": fold_metrics})
        log.info("hpo_trial", extra={"params": params, metric: avg})
        improved = avg > best_metric if higher_better else avg < best_metric
        if improved:
            best_metric = avg
            best_params = params

    return HPOResult(best_params=best_params, best_metric=best_metric, trials=trials)
