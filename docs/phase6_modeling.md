# Phase 6 — Baseline Models

Maps to **RFC-05** and **RFC-09 Milestone 6**.

## Modules

| Module | Role |
|---|---|
| `afp.models.datasets` | Flatten encoder tensors → tabular matrix; join targets |
| `afp.models.baselines` | `ConstantZero`, `HistoricalMean`, `RidgeModel`, `GradientBoostedTrees` |
| `afp.models.metrics` | `regression_metrics`, `bucket_report` |
| `afp.models.train` | `train_and_predict` orchestrator, `attach_prediction_context` |

## Required Baselines (RFC-05 §4)

`build_model(kind)` accepts:

- `constant_zero` — always predicts 0; guards against portfolio bugs that fabricate
  performance without signal.
- `historical_mean` — predicts the training-set mean of `target_normalized_signal`.
- `ridge` — `sklearn.linear_model.Ridge`. Configurable alpha; defaults to 1.0.
- `gbt` — tries LightGBM first; falls back to `HistGradientBoostingRegressor`
  if LightGBM is not installed. Engine choice is exposed on the instance.

Every model's `predict` output is clipped to `[-1, 1]` so the contract with
Phase 7 holds even if a model is mis-configured.

## Tabular Dataset Construction

`build_tabular_dataset` flattens `[N,P,F]` value + missing tensors and `[N,P,3]`
metadata into one matrix:

```
X = [ values(P*F) | missing(P*F) | metadata(P*3) ]
```

`feature_columns` carries the (anonymous) column names so leakage tests can
re-verify column hygiene. Rows whose target is NaN (e.g. missing ex-ante scale)
are dropped at this layer.

## `train_and_predict`

Inputs:

- a *fit* `AnonymousFeatureEncoder` (Phase 5 artifact)
- `train_samples`, `val_samples`, `test_samples` partitioned upstream
- the full long-format facts table

Outputs (`TrainResult`):

- `model` — fitted estimator
- `model_id` — `"<model_type>_v001"` or user-supplied
- `validation_metrics` — `{mse, mae, direction_accuracy, pearson, spearman, r2}`
- `predictions` — long DataFrame `(sample_id, predicted_signal, split, model_id, …)`
  for all of train/val/test (train kept for diagnostics only)

`attach_prediction_context` adds the columns the portfolio module needs
(`entry_date`, `exit_date`, `ex_ante_scale`, `internal_company_id`) so Phase 7
does not have to re-join samples to predictions.

## Splits

Time-based (RFC-05 §8):

```
train      : entry_date <= train_end_date
validation : train_end_date < entry_date <= validation_end_date
test       : entry_date > validation_end_date
```

Walk-forward retraining is supported by re-calling `train_and_predict` with
different cutoff samples; we intentionally do *not* hide this behind a wrapper
in the minimum useful version.

## Acceptance — verified by `tests/test_models.py`

1. All four baselines train end-to-end on the synthetic world and produce
   predictions in `[-1, 1]`.
2. `constant_zero` predictions are exactly zero.
3. Validation metrics include `mse / mae / direction_accuracy / r2`.
4. `attach_prediction_context` yields the columns Phase 7 expects.
