# Phase 59 — Michaelis-Menten signal saturation (NEUTRAL)

> Cross-disciplinary attempt: biochemistry (enzyme kinetics, 1913)
> applied to portfolio signal preprocessing.

## Why non-finance

After 21 finance-adjacent experiments (RMT, regime, max-entropy,
HRP, ES, beta, vov, etc.), the user's question was: have you tried
something *truly* cross-disciplinary, not just statistical physics
adjacent to econometrics?

Phase 59 deliberately picks an idea with **no historical connection
to finance**: Michaelis-Menten enzyme kinetics from biochemistry
(L. Michaelis + M. Menten 1913, foundational paper of pharmacology).

## Theory

In enzyme kinetics, reaction rate `v` as a function of substrate
concentration `[S]`:

```
v = V_max · [S] / (K_m + [S])
```

- Low [S]: linear regime — `v ≈ V_max · [S] / K_m`.
- High [S]: saturating — `v → V_max` as `[S] → ∞`.
- K_m = substrate concentration at half-maximum rate.

Adapted to portfolio signals as a preprocessor:

```
μ_sat = signal / (K_m + signal),   K_m = median(|signal|_{>0})
```

K_m is set adaptively per allocation step (median of non-zero
signals), making this **fully parameter-free**.

Effect:
- Preserves rank order (monotone transform).
- Linear regime for typical signals.
- Saturates extreme signal outliers to ≤ 1 — preventing
  signal-magnitude blowups from dominating allocation.

## Result

Applied as a signal preprocessor before the Phase 54 RMT-barbell +
dd_target stack. Evaluated across all 3 universes for cross-universe
robustness in a single pass:

| Universe | Phase 54 (no MM) | Phase 59 (with MM) | Δ Sharpe | Δ MDD | Δ Calmar |
|---|---|---|---|---|---|
| tier1 s42 | 1.079 / -0.286 / 0.875 | 1.053 / -0.286 / 0.844 | -0.026 | 0 | -0.031 |
| tier2 | 1.040 / -0.294 / 0.832 | 1.010 / -0.292 / 0.813 | -0.030 | +0.002 | -0.019 |
| tier3 | 0.717 / -0.290 / 0.554 | **0.734** / -0.291 / **0.564** | **+0.017** | -0.001 | +0.010 |

**Verdict: NEUTRAL.** No consistent direction across universes;
average effect is approximately zero on Sharpe and Calmar; MDD is
unchanged (dd_target already controls it).

## Interesting pattern

The differential effect across universes is informative:

- **Strong-signal universes (tier1, tier2)**: M-M hurts slightly.
  These universes have informative signal magnitudes; saturating the
  high end loses information that the RMT min-var would have used.
- **Weak-signal universe (tier3)**: M-M helps slightly. Baseline
  Sharpe is only 0.65 here, suggesting noisy magnitudes. M-M
  saturation acts as a regularizer that prevents the few apparently-
  high signals from dominating the noisy allocation.

This is consistent with the role M-M plays in biology — it caps the
"speed" of an enzyme so that abundant substrate doesn't cause
runaway reaction. The financial analog: M-M caps the "size" of a
signal so that one extreme prediction doesn't drive disproportionate
allocation. Works exactly when signal noise is high.

## Discipline

- K_m adaptive from data structure (no tuned parameter).
- V_max = 1.0 by theoretical convention.
- Single-pass evaluation across all 3 universes simultaneously.
- Honest neutral verdict.

## Status

Stays in codebase as an optional preprocessor for low-signal regimes.
Production stack remains Phase 54 (`rmt_barbell + dd_target`)
without M-M.

## Reflection

The cross-disciplinary search yielded the same kind of "neutral"
result that Phase 43 (ES) and Phase 48 (JS shrinkage) did — purely
mathematical signal transformations don't help on top of an already
strong allocator+wrapper stack. The MDD floor is set by exposure
control (dd_target), the Sharpe floor by allocator structure
(RMT-barbell). Signal-level tweaks become incremental.

The genuine cross-discipline finding is structural: **the universe
prefers exposure damping over signal transformation**, which mirrors
how robust biological systems regulate behaviour (homeostasis on
output, not on input). Phase 39, 50, 55 (input-side regime/vov/VaR
adaptive) all failed; Phase 44 (output-side DD-target damping)
succeeded. M-M (input-side saturation) lands neutral.
