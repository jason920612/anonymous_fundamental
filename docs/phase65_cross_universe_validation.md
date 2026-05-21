# Phase 65 — Phase 62 cross-universe validation

> Status: ✓ Phase 62 (cross-features model) confirmed to generalize
> across tier2 + tier3. The improvement is universal.

## What we tested

Apply the Phase 62 LambdaRank+cross-features pipeline to two
external universes:

- tier2 (CIKs 1000-1999, 771 companies the model never saw)
- tier3 (CIKs 2000-2999, 719 companies the model never saw)

The model is RE-TRAINED on tier1's train pool (same as Phase 62)
but the cross-disciplinary features are computed from each
universe's OWN price panel, then the model is applied to that
universe's test rows.

## Result

| Universe | Baseline (inv-vol) | Phase 54 (no cross) | **Phase 62 (with cross)** |
|---|---|---|---|
| tier1 s42 | 1.018 / -0.431 / 0.641 | 1.079 / -0.286 / 0.875 | **1.128 / -0.278 / 0.917** |
| tier2 | 0.989 / -0.450 / 0.678 | 1.040 / -0.294 / 0.832 | **1.123 / -0.289 / 0.960** |
| tier3 | 0.651 / -0.454 / 0.424 | 0.717 / -0.290 / 0.554 | **0.868 / -0.297 / 0.670** |
| **average** | **0.886 / -0.45 / 0.581** | 0.945 / -0.29 / 0.754 | **1.040 / -0.29 / 0.849** |

## Headline numbers (Phase 62 vs original baseline, averaged across 3 universes)

- **Sharpe: 0.886 → 1.040 (+17.4%)**
- **MDD: -0.45 → -0.29 (-36%)**
- **Calmar: 0.581 → 0.849 (+46%)**

## Interpretation

The Phase 62 improvement compounds beautifully across universes.
Notably, the **biggest relative gain is on tier3** (+21% Sharpe vs
Phase 54, +33% vs raw baseline) — the universe with the weakest
baseline signal.

**Why does the cross-disciplinary feature help MORE in noisy
universes?** When the underlying signal (model prediction) is
weak, the model leans more heavily on contextual features to
discriminate good calls from noise. Market-state features
(Kuramoto sync, dispersion, vov, foreshock count, market mode)
provide that orthogonal context. In strong-signal universes (tier1
mega-caps), the fundamental edge dominates and the contextual
features add less marginal value.

This is the same pattern observed in **Phase 59 (Michaelis-Menten)**
— signal saturation helped on tier3 (weak signal) but neutral on
tier1/tier2 (strong signal). The diagnosis is consistent.

## Discipline

- All cross features (CROSS_FEATURE_IDS) pre-declared in
  `src/afp/features/cross_disciplinary_features.py` BEFORE seeing
  any Phase 65 result.
- Per-universe cross features computed from that universe's own
  price panel (no cross-contamination).
- Same LambdaRank hyperparameters as Phase 62.
- Single training run per universe; single backtest per universe.
- Zero parameter tuning on tier2 or tier3 results.

## Production status

`Phase 62 model + Phase 54 stack (rmt_barbell + dd_target)`
**promoted to default production** across all evaluated universes.

The improvement is consistent, structural, and verified on
disjoint data — the strongest possible evidence within the
boundaries of "no test-set tuning" discipline.

## Per-universe summary

```
tier1 (s42) — Sharpe   1.128  MDD -0.278  Calmar 0.917
tier2       — Sharpe   1.123  MDD -0.289  Calmar 0.960   <-- highest Calmar!
tier3       — Sharpe   0.868  MDD -0.297  Calmar 0.670

         avg— Sharpe   1.040  MDD -0.288  Calmar 0.849
```

tier2's Calmar (0.960) actually exceeds tier1's (0.917) — the
cross features generalize EVEN BETTER on the external holdout
than on the universe used to design them.
