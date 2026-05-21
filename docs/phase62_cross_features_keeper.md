# Phase 62 — Cross-disciplinary features in LambdaRank (NEW WINNER)

> Status: ✓ keeper. First strategy to beat Phase 54 since it was
> established. Embodies the user's "first principles + train models
> with these methods" guidance.

## Hypothesis

Phase 39, 50, 55, 60, 61 all tried cross-disciplinary signals as
**exposure wrappers**. All failed. Phase 60+61 (Gutenberg-Richter
+ Kuramoto) failed especially badly:

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| rmt_barbell+dd (Phase 54) | 1.079 | -0.286 | 0.875 |
| +GR wrapper | 0.823 | -0.286 | 0.593 |
| +Kuramoto wrapper | 0.676 | -0.282 | 0.480 |
| +GR + Kuramoto | 0.574 | -0.282 | 0.368 |

A static rule firing on these signals **destroys** their value.

**But what if we let the MODEL learn from them?** The LambdaRank
already learns to combine ~17000 anonymous fundamental features. If
we add 6 cross-disciplinary market-state features as additional
columns, the model can decide when each is informative and how to
combine them with fundamentals.

## Cross-disciplinary features

Six anonymous numerical features per sample, computed at
`entry_date` from the universe-wide price panel:

| Feature | Source | Physics analog |
|---|---|---|
| `f_market_temp_z` | cross-sectional dispersion z-score | thermodynamic temperature |
| `f_kuramoto_r` | Kuramoto coupled-oscillator order param | synchronization |
| `f_kuramoto_z` | z-score of `f_kuramoto_r` | regime detection |
| `f_gr_count_z` | Gutenberg-Richter foreshock count z | self-organized criticality |
| `f_market_mode_ratio` | λ_1 / trace of cleaned correlation | RMT dominant mode |
| `f_vov_z` | universe vol-of-vol z-score | GARCH heteroscedasticity |

All causally computed (no lookahead). All anonymous to the model.

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| Phase 54 (s42 baseline) | 1.079 | -0.286 | 0.875 |
| **Phase 62 (rmt_barbell+dd, cross-features model)** | **1.128** | **-0.278** | **0.917** |

vs Phase 54: Sharpe +4.5%, MDD slightly better, Calmar +4.8%.

vs original baseline (inverse-vol): Sharpe +10.8%, MDD -36%,
Calmar +43%.

## Feature importance

Top-30 LambdaRank gains (post-training):

```
feature_idx       gain     interpretation
       17677  303.43      price feature
       17674  115.03      price feature
       17672   76.91      price feature
       17676   72.16      price feature
       17675   64.98      price feature
       17689   59.63      CROSS feature (f_vov_z missing flag)
       17684   49.33      CROSS feature (f_market_temp_z)
       17686   40.43      CROSS feature (f_kuramoto_z)
       17687   21.83      CROSS feature (f_gr_count_z)
        ...      <30     anonymous fundamentals
```

Cross-disciplinary features sit at rank 6-9 by gain, above 99% of
the anonymous fundamental features. The model assigns real
predictive weight to these market-state observables.

## Why static wrappers failed but model training succeeded

Static wrapper (e.g., "if Kuramoto z > 1, scale exposure by 0.6"):
- Triggers on ANY high-z day, even ones that turn out to be
  opportunities (positive selection regime)
- Cannot distinguish "high z but tier1-style mega-cap regime where
  selection still works" from "high z + small-cap meltdown"

Model (LambdaRank with these features):
- Learns INTERACTIONS between cross-disciplinary signals and
  fundamentals
- High Kuramoto z combined with positive fundamental signal → still
  rank highly
- High Kuramoto z + weak fundamentals → suppress
- Effectively: the model implements an adaptive, contextual rule
  that no fixed wrapper threshold can capture

This is the deep lesson: **cross-disciplinary signals are
informative but must be incorporated as model FEATURES, not as
exogenous OVERRIDE rules.**

## Discipline

- All 6 features pre-declared from theory (Phases 39, 50, 60, 61, RMT).
- Causal computation, no lookahead.
- Same LambdaRank training hyperparameters as Phase 29 winner — no
  re-tuning to accommodate the new features.
- Single training run, single backtest evaluation.
- The new predictions are saved as a NEW parquet
  (`lambdarank_v3_cross_features.parquet`) without overwriting the
  original — Phase 54 stays as ablation reference.

## Pending

Cross-universe validation (tier2, tier3) requires re-training each
universe's LambdaRank with its own cross-disciplinary features.
That's a substantial pipeline run; queued as future work.

## Production stack

```yaml
model:
  type: lambdarank_with_cross_features
  features:
    base: encoder/v1 anonymous fundamentals + 6 price features
    cross_disciplinary:
      - f_market_temp_z       # Phase 39 thermodynamic
      - f_kuramoto_r          # Phase 61 synchronization
      - f_kuramoto_z
      - f_gr_count_z          # Phase 60 self-organized criticality
      - f_market_mode_ratio   # RMT dominant eigenvalue
      - f_vov_z               # Phase 50 GARCH vol-of-vol
  training: LambdaRank, max_position=100, lr=0.03, leaves=31

allocator:
  type: rmt_barbell  # Phase 54
  safe_fraction: 0.80
  concentrated_top_k: 5
  rmt:
    risk_aversion: 5.0

wrappers:
  - dd_target:  # Phase 44
      dd_trigger: 0.10
      alpha: 1.0
      scale_floor: 0.30
```
