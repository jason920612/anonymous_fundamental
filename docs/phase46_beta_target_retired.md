# Phase 46 — Beta-targeting wrapper (RETIRED)

## Hypothesis (a priori)

Phase 44's drawdown-targeting wrapper is *reactive* — it fires after a
drawdown shows up. A *predictive* alternative would be to scale
exposure by inverse rolling beta to SPY:

```
scale(t) = clip( 1 / max(1, β_60d) , floor, 1.0 )
```

Theory: when the strategy's rolling beta to SPY > 1, the portfolio
will amplify any market shock by exactly that factor. Cutting
exposure ahead of time should reduce MDD without waiting for the
loss to accumulate.

## Result

Single-pass eval, same 2020-2026 window and predictions:

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| baseline (no wrapper) | 1.018 | -0.431 | 0.641 |
| baseline + beta_target | 0.991 | -0.418 | 0.581 |
| rmt_minvar (no wrapper) | 1.096 | -0.446 | 0.727 |
| rmt_minvar + beta_target | 1.070 | -0.427 | 0.645 |
| **rmt_minvar + dd_target (Phase 44 winner)** | **1.084** | **-0.286** | **0.801** |
| rmt_minvar + dd + beta | 1.061 | -0.290 | 0.694 |

**Verdict: retired.** Beta-target alone is weaker than Phase 44
dd-target on every metric, and stacking it on top of dd-target HURTS
Calmar (0.801 → 0.694). The two wrappers are redundant — they cut
exposure on overlapping days, double-paying the cost without gaining
extra protection.

## Why it failed

The mechanism we hoped for — beta rising *before* a drawdown,
allowing pre-emptive de-risking — was not observed in this universe.
By the time rolling 60-day beta is materially above 1, the drawdown
has typically already begun, making the wrapper effectively reactive
*and* less responsive than dd_target's direct DD measurement.

## Methodology discipline

- All parameters pre-declared (`lookback=60d`, `floor=0.30`,
  `ceiling=1.0`, `blend=0.5`).
- Single-pass evaluation; no re-runs or threshold tweaks.
- Honest negative reporting: the idea did not work, the code stays
  for reference but `rmt_minvar + dd_target` remains the production
  recommendation.
