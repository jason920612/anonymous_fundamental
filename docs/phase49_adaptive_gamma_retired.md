# Phase 49 — Adaptive γ via signal entropy (RETIRED)

## Hypothesis

The optimal risk-aversion γ in Markowitz `max(wᵀμ - γ/2 wᵀΣw)`
should depend on how concentrated the signal is:

- Low signal entropy (one clear winner) → low γ → tilt heavily.
- High signal entropy (uniform signal) → high γ → fall back on min-var.

Implementation: map normalized Shannon entropy `H/log(N) ∈ [0,1]`
linearly to a pre-declared envelope `γ ∈ [γ_min, γ_max] = [2, 10]`
(the same envelope used in the Phase-41 sensitivity check).

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| adaptive_rmt | 1.042 | -0.441 | 0.671 |
| adaptive_rmt + dd_target | 1.022 | -0.285 | 0.731 |
| (reference: rmt_minvar γ=5) | **1.096** | -0.446 | **0.727** |
| (reference: rmt_minvar+dd_target) | **1.084** | **-0.286** | **0.801** |

**Verdict: retired.** Both adaptive variants under-perform the
fixed-γ counterparts by ~0.05-0.07 Sharpe.

## Why it failed

In this universe the candidate signal vector (top-100 LambdaRank
scores after `select_candidates` filtering) has consistently *high*
entropy — values are clustered in a narrow range, so normalized
`H/log(N)` is close to 1.0 on most rebalance days. The mapping
therefore drives γ to ≈ 10 (the defensive end of the envelope),
which the Phase-41 γ-sensitivity check showed is the WORST point
of that envelope.

In other words: the theory-driven mapping pointed the wrong
direction for this signal. High entropy here is not a signal that
we should be defensive — it is just a property of the
already-filtered top-K signal vector. Tilting toward γ=5 (or even
γ=2) gives better performance.

## Discipline

- The functional form (linear mapping from normalized entropy)
  was pre-declared.
- The envelope [2, 10] was also pre-declared (same as the
  sensitivity check).
- I did not flip the sign of the mapping after seeing the result.
  Doing so would be test-set fitting. Theory said "high H → defensive";
  reality said the opposite — but I cannot now claim a new theory
  retroactively. The idea is retired.

## What this teaches

A pre-declared theory can be empirically wrong. The discipline isn't
"the theory must work" — it's "we don't tweak after seeing test
results." The retirement of Phase 49 alongside the keepers (38, 44)
shows the methodology is doing its job: pruning bad hypotheses
without fitting them away.
