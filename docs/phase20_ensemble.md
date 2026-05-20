# Phase 20 — Equal-Weight & Rank-Average Ensemble

## Why

Each model architecture captures the (very) weak signal in anonymous
fundamentals differently:
- **Ridge** smooths everything linearly
- **LightGBM/XGB** captures interactions but adds variance
- **MLP** can chase non-linearities but easily overfits

Averaging often reduces variance enough to lift Sharpe. With Spearman of
each individual model in the 0.01-0.05 range, even small variance reduction
is worth trying.

## Module

`afp.models.ensemble`:

- `equal_weight_ensemble(paths, out)` — merges prediction parquets on
  `sample_id × split`, averages `predicted_signal`, clips to `[-1, 1]`.
- `rank_average_ensemble(paths, out)` — converts each model's signal to
  within-split pct_rank, averages ranks, maps to `[-1, 1]` via
  `2 × mean_rank - 1`. More robust to scale differences across models.

CLI: `afp ensemble --predictions a.parquet b.parquet ... --method mean|rank
--out artifacts/predictions/ensemble.parquet`.

The CLI also re-attaches `entry_date / exit_date / ex_ante_scale` from
`event_samples.parquet` so the backtest CLI can read the ensemble
parquet directly.

## RFC Compliance

- Ensembling happens at the *prediction* layer — the model inputs and
  training procedures remain unchanged.
- Models can have wildly different validation IC; ensembling does not
  amplify look-ahead because each member's training-time discipline is
  preserved at its source.

## Phase 17 Result

Mean ensemble of (Ridge, LightGBM static, XGB GPU, MLP GPU) → Sharpe 0.73
(vs Ridge solo 0.76). The ensemble was second-best in Phase 17. Rank-
average ensemble degraded performance because the noisy GBT/MLP rank
distributions pulled the consensus away from Ridge's smoother view.

## Acceptance — `tests/test_ensemble.py`

1. Mean ensemble exactly equals the average of input signals for a
   deterministic three-row case.
2. Rank ensemble produces a centered output in `[-1, 1]` for random
   Gaussian inputs of N=200.
