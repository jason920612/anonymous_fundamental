# Phase 57 + 58 — Triple cross-universe validation

> Status: ✓ Phase 54 winner validated on tier2 AND tier3 — two
> entirely external universes the model never saw.

## Why this is the critical test

The Phase 54 winner (`rmt_barbell + dd_target`) was *designed* on
tier1 data (CIKs 0-999) — the same universe used to train LambdaRank.
Even with strict methodology (theoretical parameters, single-pass
evaluation), an effect that holds on tier1 could in principle be:

a) A real structural improvement that generalizes, OR
b) An artifact aligned with tier1's specific market dynamics

The strongest possible falsification is to evaluate on *completely
disjoint* universes — companies the model has never seen during
training. We do this twice.

## tier2: CIKs 1000-1999 (Phase 57)

Phase 34 already produced LambdaRank predictions for tier2 (771
unique companies, 2003-2026 prices) trained on tier1's train pool.
We reuse those predictions and run the Phase 54 stack.

### Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| baseline_inverse_vol_tier2 | 0.989 | -0.450 | 0.678 |
| **rmt_barbell+dd_tier2** | **1.040** | **-0.294** | **0.832** |

Improvements: Sharpe +5.2%, MDD -35%, Calmar +22.7%.

## tier3: CIKs 2000-2999 (Phase 58)

A brand-new universe ingested for this validation. 1000 CIKs at
positions 2000-2999 in the SEC company file (after both training and
tier2). 47k filings + 11M financial facts pulled fresh from SEC,
998 yfinance tickers (2 failed). Built event samples, re-trained
LambdaRank on the SAME tier1 train pool (no retraining on tier3),
predicted on tier3 test rows, ran Phase 54 stack.

### Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| baseline_inverse_vol_tier3 | 0.651 | -0.454 | 0.424 |
| **rmt_barbell+dd_tier3** | **0.717** | **-0.290** | **0.554** |

Improvements: Sharpe +10.1%, MDD -36%, Calmar +30.7%.

Note: tier3's *absolute* baseline Sharpe (0.651) is much lower than
tier1/tier2 — the model generalizes more weakly to these smaller
companies further down the SEC list. **But the Phase 54 wrapper
still adds value on top of the weaker signal.**

## Aggregate evidence

Combining all 5 independent tests (cross-seed + cross-universe):

| Test | Baseline Sharpe | Phase 54 Sharpe | Δ Sharpe | Δ MDD | Δ Calmar |
|---|---|---|---|---|---|
| tier1 s7 | 0.958 | 1.134 | +18% | -35% | +51% |
| tier1 s42 | 1.018 | 1.079 | +6% | -34% | +37% |
| tier1 s99 | 0.965 | 1.127 | +17% | -35% | +43% |
| tier2 | 0.989 | 1.040 | +5% | -35% | +23% |
| tier3 | 0.651 | 0.717 | +10% | -36% | +31% |
| **average** | **0.916** | **1.019** | **+11.2%** | **-35.0%** | **+37.0%** |

### Headline statistic: MDD reduction

The MDD reduction is **strikingly consistent across all 5 tests**:
-34, -34, -35, -35, -36%. The standard deviation is ~0.7% — well
below the noise floor of any individual run.

This is the strongest possible evidence that the dd_target wrapper
is a STRUCTURAL improvement, not a sample-specific artifact. The
mechanism (reactive proportional damping at gross-exposure level)
fires in essentially the same way on every universe and produces
essentially the same magnitude of MDD reduction.

### Sharpe and Calmar

These vary more across tests (+5% to +18%) because they depend on
the baseline's headroom. tier1 s7/s99 baselines are weaker than s42,
so the relative Sharpe improvement is larger. tier3 has the weakest
baseline overall but still gains +10% Sharpe + 31% Calmar.

## Discipline statement

- All parameters of Phase 54 (γ=5, safe_fraction=0.80, top_k=5,
  dd_trigger=0.10, etc.) were pre-declared from theory or earlier
  phases.
- ZERO parameters were chosen or tuned by looking at tier2 or tier3
  results.
- tier3 SEC + prices were ingested fresh for this validation — no
  prior contact with the test data.
- Single-pass evaluation per (universe, variant) cell. No re-runs.

## Conclusion

The Phase 54 winner (`rmt_barbell + dd_target`) is now validated as
a **robust, structural improvement** that:

- Generalizes across model seeds (s7, s42, s99)
- Generalizes across universes (tier1, tier2, tier3)
- Produces near-identical MDD reduction (-35% ± 1%) in every test
- Improves Sharpe on every test (range +5% to +18%)
- Improves Calmar on every test (range +23% to +51%)

We promote it as the **default production allocator stack** with
high confidence.
