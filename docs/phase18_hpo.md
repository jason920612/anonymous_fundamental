# Phase 18 — Time-Based Hyperparameter Search

## Why

LightGBM defaults haven't been tuned for this problem. Even modest gains
on the *signal extraction* side (Spearman 0.03 → 0.06) compound through
the portfolio module — equal-risk allocation amplifies tiny rank-correlation
edges into measurable Sharpe.

## Strict Test-Data Discipline

The user emphasized: **test data must never be touched during hyperparameter
selection.** This module enforces that structurally:

- `time_series_folds(samples)` takes only `train ∪ validation` rows (the
  caller is responsible for excluding test before passing the frame).
- Folds are **rolling-origin**: each fold's training window is strictly
  earlier than its evaluation window. No data leakage between folds.
- After search, the final model is fit once on the full train+validation
  set with the best params, then scored on test (in the regular `afp train`
  flow). The test score is not part of the search loop.

## Modules

| Function | Role |
|---|---|
| `afp.models.hpo.time_series_folds(samples, n_folds, min_train_months)` | Rolling-origin CV splits; returns positional `(train_idx, eval_idx)` pairs |
| `afp.models.hpo.grid_search(model_kind, param_grid, encoder, train_val_samples, facts, metric, n_folds, sample_weight_kwargs)` | Exhaustive grid search; returns `HPOResult(best_params, best_metric, trials)` |

The `metric` parameter accepts any key in `regression_metrics`:
`mse`, `mae`, `direction_accuracy`, `pearson`, `spearman`, `r2`. The
search automatically picks "higher better" for everything except mse/mae.

## Usage

```python
from afp.models.hpo import grid_search

train_val = samples_with_target[
    samples_with_target["split"].isin(("train", "validation"))
].dropna(subset=["target_normalized_signal"])

result = grid_search(
    model_kind="gbt",
    param_grid={
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.02, 0.05, 0.10],
        "min_data_in_leaf": [50, 100, 200],
    },
    encoder=encoder,
    train_val_samples=train_val,
    facts=facts,
    metric="spearman",          # higher is better
    n_folds=4,
    sample_weight_kwargs={"by_company": True, "by_quarter": True},
)
# result.best_params -> dict to pass to `afp train --config ... --model-kind ...`
```

## RFC Compliance

- The search uses anonymous feature representations from a *fit* encoder
  (which itself was fit on training data only — RFC-03 §13).
- The model never sees human-readable concept names during HPO.
- Test data is structurally excluded — it cannot enter the search even
  with a coding mistake (the function signature takes a `train_val_samples`
  argument that is contractually the union of train + validation only).

## Acceptance — `tests/test_hpo.py`

1. Every fold has `max(train_entry_date) ≤ min(eval_entry_date)`.
2. Train and eval index sets do not overlap.
3. Training-window size grows monotonically across folds (rolling origin
   correctly extends the past).
