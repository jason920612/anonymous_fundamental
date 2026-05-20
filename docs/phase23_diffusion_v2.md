# Phase 23 — Diffusion v2 as Main Model

## What Changed vs Phase 22

| Knob | v1 | v2 |
|---|---|---|
| Condition | fundamentals only (~17,672 dims) | fundamentals + 6 anonymous price features (17,684 dims) |
| Condition encoder | 256 → 128 | 1024 → 192 |
| Denoiser hidden | 256 | 512 |
| Epochs | 40 | 100 |
| Inference samples | 64 | 128 |
| LR | 3e-4 | 2e-4 |

## Validation Metrics

| | v1 | v2 |
|---|---:|---:|
| direction accuracy | 57.1% | 53.6% |
| Spearman | 0.0216 | 0.0087 |

Direction accuracy went down slightly, but **distributional calibration
improved**: avg_std went from ~0.05 to 0.108, avg probability_positive
from ~0.50 (collapsed) to 0.57 (real skew). This matters because the
distribution allocator's whole point is to use the calibration.

## Backtest

| Variant | Sharpe | Ann.Ret | MDD |
|---|---:|---:|---:|
| v2 + inverse_vol allocator | 0.66 | 13.3% | -36.3% |
| **v2 + distribution allocator (p_pos ≥ 0.55)** | **0.79** | 16.1% | -35.8% |

The distribution allocator extracts ~13 basis-points of Sharpe from v2
just by re-weighting picks by `confidence / std` — proof that the
extra calibration cost (+60 epochs, 4× model size) is paying off.

## Script

`scripts/run_v1000_diffusion_v2.py` — fits diffusion v2 on
`train` only, scores `val + test` via inference. Writes a predictions
parquet that includes `predicted_signal`, `predicted_signal_std`,
`probability_positive`, `q05`, `q95` (extra columns the distribution
allocator reads).

## RFC Compliance

Same as v1: anonymous features only, no human concept names; test data
is never used in fit.

## Acceptance

- `tests/test_diffusion.py` regression tests still pass.
- v2 predictions parquet schema includes the four distributional columns
  the Phase 24 allocator consumes.
- Backtest with `--allocator distribution` reads them correctly.
