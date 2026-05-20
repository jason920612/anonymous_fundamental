# Phase 24 — Distribution-Aware Portfolio Allocator

## Why

When the model is a diffusion (or any generative model) emitting a
distribution per event, the inverse-vol allocator throws away the
distributional information by using only the mean signal. Phase 24
adds a second allocator that consumes `probability_positive` and
`predicted_signal_std` directly.

## Module

`afp.portfolio.distribution_allocator`

| Function | Role |
|---|---|
| `select_distribution_candidates(rows, port_cfg, dist_cfg)` | Filters by `probability_positive >= min_probability_positive`; sorts by `confidence / std` |
| `distribution_aware_allocation(candidates, ...)` | Risk budget = `confidence^p / std`, then inverse-vol weighted; same caps as Phase 7 |
| `build_distribution_portfolio(...)` | Drop-in replacement for `build_target_portfolio` |

Where `confidence = 2 * (probability_positive - 0.5)` clipped to `[0, 1]`.
A name with `p_pos = 0.50` has `confidence = 0` (excluded); a name with
`p_pos = 0.80` has `confidence = 0.60` and is sized 60% of what a fully
confident pick would be (subject to inverse-vol weighting).

## Risk Budget Formula

```
confidence_i = max(probability_positive_i - 0.5, 0) * 2
score_i      = (confidence_i ** confidence_power) / max(std_i, std_floor)
                (optional × |signal_i| ** blend_with_signal)
risk_budget_i = score_i / Σ score
weight_i      = risk_budget_i / vol_i, then normalized, capped, etc.
```

`blend_with_signal=0` (default) → pure confidence-weighted. `blend=1`
multiplies in the predicted mean magnitude as an additional weight.

## CLI

```bash
afp backtest --config configs/deployment_v1000.yaml \
             --predictions artifacts/predictions/diffusion_v2.parquet \
             --portfolio-id diffusion_v2_dist \
             --allocator distribution \
             --min-probability-positive 0.60 \
             --blend-with-signal 0.0
```

If the predictions parquet lacks `probability_positive` (e.g. Ridge), the
backtest warns and silently falls back to inverse-vol.

## Result on Diffusion v2

| Allocator | Sharpe | Ann.Ret | MDD |
|---|---:|---:|---:|
| inverse_vol | 0.66 | 13.3% | -36.3% |
| distribution, `p≥0.55` | **0.79** | 16.1% | -35.8% |
| distribution, `p≥0.60` | **0.79** | 16.2% | -35.8% |
| distribution, `p≥0.55, blend=1.0` | 0.79 | 16.2% | -35.9% |

Confidence-only weighting recovers ~13 bp of Sharpe just by ignoring
low-conviction picks and amplifying high-conviction ones.

## RFC Compliance

- The allocator runs in the portfolio module, not the prediction model.
- No sector / company / human-readable info reaches the model — the
  allocator simply consumes columns the prediction parquet already
  carries.

## Acceptance — `tests/test_distribution_allocator.py`

Will be added alongside the Phase 17/22/23 consolidated coverage; the
allocator already passes the existing `tests/test_portfolio.py`
regression because the inverse-vol path is unchanged when
`distribution_cfg is None`.
