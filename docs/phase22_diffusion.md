# Phase 22 — Conditional Diffusion on Anonymous Features

Implements **RFC-05 §7** (optional conditional diffusion). The user
explicitly asked for it after the Phase 17 ablation showed no model
architecture beat Ridge solo (Sharpe 0.76).

## Why Try Diffusion Here

Every model so far emits a point estimate of the normalized target.
Diffusion can instead produce a **distribution** of plausible future
movements per event, conditional on the anonymous feature representation:

- `mean` → use as `predicted_signal`
- `std` → uncertainty (could drive position sizing)
- `probability_positive` → cleaner long-only filter than `sign(mean)`
- `q05`, `q95` → downside / upside reasoning for the portfolio module

RFC-05 §7.5 sets the acceptance bar: diffusion is only worth keeping if it
improves at least one of rank correlation, top-decile realized return,
portfolio Sharpe, max drawdown, or distribution calibration **without
degrading turnover or stability**.

## Architecture (`afp.models.diffusion.ConditionalDiffusion`)

```
condition encoder:   Linear(n_features → 512) → GELU → Dropout
                  → Linear(512 → 128) → GELU
denoiser ε-network:  input = [noisy_y(1) | cond(128) | t_embed(64)] = 193 dim
                  → Linear(193 → 256) → GELU → Dropout
                  → Linear(256 → 256) → GELU → Dropout
                  → Linear(256 → 1)
```

- **Noise schedule:** linear β ∈ [1e-4, 2e-2] over T=200 steps.
- **Sinusoidal time embedding** of dim 64 (DDPM convention).
- **Loss:** MSE between predicted ε and true noise.
- **Optimizer:** AdamW lr=3e-4, weight_decay=1e-4.
- **Target standardization:** y is normalized by (mean, std) of train
  targets; the sampler unstandardizes after sampling.
- **Inference:** DDPM ancestral sampler, `inference_samples=64` per event,
  mean used as `predicted_signal`.

GPU when available; falls back to CPU. The condition encoder and denoiser
sit on the same device.

## RFC Compliance & Test-Data Discipline

- **Input:** dense anonymous feature matrix (`[N, P*F + missing + meta]`).
  The model has no idea what any feature *means* — it just sees an opaque
  high-dimensional vector. No human concept name reaches the model.
- **Training:** the trainer takes `train_samples` only. `val_samples` and
  `test_samples` flow through the encoder during transform but are never
  presented as labeled targets during fit.
- **Standardization stats** are computed from training data only and
  stored as part of the artifact.

## CLI

There's no `afp diffusion` subcommand because the training loop is
distinct enough (GPU-only sensible at the configured sizes) that the
single-file runner `scripts/run_v1000_diffusion.py` is cleaner:

```bash
python3 scripts/run_v1000_diffusion.py 60   # epochs
afp backtest --config configs/deployment_v1000.yaml \
             --predictions artifacts/predictions/diffusion_v1000.parquet \
             --portfolio-id diffusion_v1000
```

The predictions parquet carries extra columns (`predicted_signal_std`,
`probability_positive`, `q05`, `q95`) which the backtest engine ignores
but the diagnostics layer can pick up later.

## Acceptance — `tests/test_diffusion.py`

1. Diffusion trains end-to-end on a synthetic problem and produces
   predictions in `[-1, 1]`.
2. The sampler produces a non-trivial spread (`std > 0.01`) rather than
   collapsing to point estimates.

## Expected Outcome

Honest expectation given Phase 11/17 evidence:

- Sharpe somewhere in the 0.6-0.8 range — comparable to other models, not
  a step-change.
- Slightly better calibration / distribution diagnostics than tree
  baselines.
- Likely fails the RFC-05 §7.5 acceptance bar — but worth running to
  confirm rather than assume.

The result and verdict go into Phase 23's report alongside the rest of
the Phase 17 ablation summary.
