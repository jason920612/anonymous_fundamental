# Phase 47 — Eigenportfolio market-mode subtraction (RETIRED)

## Hypothesis

The dominant eigenvector v₁ of the RMT-cleaned cov matrix is the
"market mode." Signal projected onto v₁ represents market-correlated
alpha that blows up together in drawdowns. Projecting orthogonal to
v₁ before the min-var solve should leave only cross-sectional alpha,
producing a portfolio with lower correlated-shock exposure.

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| eigenport_minvar | 0.774 | -0.373 | 0.455 |
| eigenport_minvar + dd_target | 0.564 | -0.262 | 0.350 |
| (reference: baseline) | 1.018 | -0.431 | 0.641 |
| (reference: rmt_minvar+dd_target) | 1.084 | -0.286 | 0.801 |

**Verdict: retired.**

## Why it failed

For a long-only portfolio, weights must lie in the positive orthant.
The dominant eigenvector v₁ has mostly-positive loadings, so *any*
long-only portfolio has positive exposure to v₁ — orthogonality to
v₁ requires shorts.

What our implementation could do is feed the *signal* orthogonal to
v₁ into the min-var solve. This shrinks signal magnitudes (because
much of the signal lies along v₁), so the solve falls back closer
to equal-weight on the candidate universe. Less alpha capture → lower
Sharpe.

Adding dd_target on top made it worse — both forces de-risked the
exposure, and the doubly-defensive portfolio gives up too much of
the model's edge.

## Methodology

Parameter-free (just eigendecomp). Single-pass evaluation. No tweaks
or re-runs to "rescue" the idea.
