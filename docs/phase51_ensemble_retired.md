# Phase 51 — Allocator ensemble (RETIRED)

## Hypothesis

Averaging two allocators with different failure modes should produce
a weight vector more robust than either alone:

- Inverse-vol: ignores covariance structure
- RMT min-var: sensitive to small cleaned eigenvalues

Theoretical 50/50 blend should hedge against each failure mode.

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| ensemble (no wrapper) | 1.061 | -0.438 | 0.685 |
| ensemble + dd_target | 1.001 | -0.286 | 0.715 |
| (reference: rmt_minvar+dd_target) | **1.084** | **-0.286** | **0.801** |

**Verdict: retired.** Both ensemble variants under-perform the Phase
44 winner. 50/50 dilutes the RMT advantage without compensating MDD.

## Why it failed

RMT min-var is not "fragile" in this universe — the M-P cleaning
plus the long-only / cap constraints already produce robust weights.
Diluting it with inverse-vol throws away the correlation-aware
signal RMT provides, gaining no risk reduction in exchange.

## Discipline

- α=0.5 was the theoretical mid-point (no preference).
- Not searched over α — single-pass evaluation per α=0.5.
- Honest retirement.
