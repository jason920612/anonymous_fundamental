# Phase 19 — GPU-Accelerated Models

## Why

LightGBM is CPU-only in the shipped pip wheel (its GPU build requires
recompilation against OpenCL or CUDA). To take advantage of the local RTX
4060 8GB without that build pain, Phase 19 adds two GPU-native backends:

- **`xgb_gpu`** — XGBoost histogram booster with `tree_method="hist"` and
  `device="cuda"`. Pip wheel ships with CUDA support; ~5× faster than
  LightGBM CPU on this dataset and supports the same `n_estimators`,
  `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`,
  `min_child_weight` knobs.
- **`mlp_gpu`** — PyTorch dense MLP regressor (`Linear → GELU → Dropout`
  ×2 → `Linear → tanh`). Runs on CUDA when available, falls back to CPU.
  Loss: Huber. Optimizer: AdamW with cosine LR decay. Optionally accepts
  `sample_weight` via per-element loss reweighting.

## How To Use

```yaml
model:
  type: xgb_gpu              # or mlp_gpu
```

Or via CLI:

```bash
afp train --model-kind xgb_gpu --config configs/deployment_v1000.yaml \
          --model-id xgb_gpu_v1000
afp train --model-kind mlp_gpu --config configs/deployment_v1000.yaml \
          --model-id mlp_gpu_v1000
```

XGB params come from `model.xgb_gpu.params` if set, else from the dict at
`model.xgb_gpu`. Same pattern for MLP under `model.mlp_gpu`.

## Defaults

| Backend | Defaults |
|---|---|
| `xgb_gpu` | `n_estimators=500, lr=0.05, max_depth=6, min_child_weight=10, subsample=0.9, colsample_bytree=0.5, tree_method=hist, device=cuda` |
| `mlp_gpu` | `hidden=[512,128], dropout=0.3, epochs=80, batch_size=512, lr=1e-3, weight_decay=1e-4, Huber β=0.1, AdamW + cosine` |

## RFC Compliance

- Model input remains anonymous feature IDs. The GPU backends accept the
  same dense matrix as Ridge / LightGBM; nothing in the column space or
  feature names changes.
- Random seeds are honored (`torch.manual_seed`, `np.random.seed`,
  XGBoost `random_state`).
- Predictions are clipped to `[-1, 1]` by `BaseModel._clip` (same contract
  as every other model).

## Test-Data Discipline

Both GPU backends are trained inside `train_and_predict` which fits only
on `train_samples` and never touches `val_samples` or `test_samples`
during fit. HPO (Phase 18) restricts even further by structurally taking
only `train ∪ validation`.

## Acceptance — `tests/test_gpu_models.py`

1. `xgb_gpu` trains on a synthetic regression problem and outputs
   predictions in `[-1, 1]` (skipped when XGBoost not installed).
2. `mlp_gpu` trains and outputs predictions in `[-1, 1]` (skipped when
   `torch.cuda.is_available()` is False).
