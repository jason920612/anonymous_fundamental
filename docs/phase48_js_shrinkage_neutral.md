# Phase 48 — James-Stein signal shrinkage (NEUTRAL)

## Hypothesis

Stein (1956) / James-Stein (1961): for N ≥ 3 means estimated from
noisy observations, the JS shrinkage estimator strictly dominates the
MLE in mean-squared error:

```
μ_JS_i = μ_grand + (1 - c)·(μ_i - μ_grand),
c = min(1, (N - 2)·σ²/Σ(μ_i - μ_grand)²)
```

Applying this to the candidate signal vector before the allocator
should produce strictly better (or equal) downstream behaviour for
any allocator that is monotonic in signal magnitude.

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| rmt_minvar (no JS) | 1.096 | -0.446 | 0.727 |
| **js_rmt_minvar** | **1.096** | **-0.446** | **0.727** |
| rmt_minvar + dd_target | 1.084 | -0.286 | 0.801 |
| **js_rmt_minvar + dd_target** | **1.083** | **-0.286** | **0.800** |

**Verdict: NEUTRAL — adds no measurable value, but does not hurt
either.** Keeping in codebase as a free safety net.

## Why neutral

In our setup the signal that reaches the allocator is
`positive_signal = max(0, predicted_signal)`, already filtered to
the top-K candidates. Cross-sectional variance is small (signals are
already concentrated in a narrow range), so the shrinkage factor `c`
solves to near-zero — JS is mathematically barely doing anything to
this distribution.

If we'd applied JS shrinkage *before* the candidate selection step,
on the full predicted_signal distribution, the effect would likely
be larger. We did not attempt that variant because changing the
candidate selector requires re-deriving a non-trivial threshold and
opens a door to test-set tuning.

## Discipline

- Shrinkage factor `c` is closed-form from theory — no tuning.
- Single-pass evaluation.
- Honest neutral result: idea is theoretically valid but empirically
  inactive on this signal distribution.
