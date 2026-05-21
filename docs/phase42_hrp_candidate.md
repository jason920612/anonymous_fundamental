# Phase 42 — Hierarchical Risk Parity (candidate)

> Status: code + tests landed; **not yet evaluated**. Awaiting Phase 41
> results before deciding whether to commit a budget to a separate
> evaluation run.

## Why HRP

The diagnosis at the end of Phase 26 was that the production
inverse-vol allocator runs at Sharpe ≈ 1.0 while a same-universe
equal-risk-active benchmark hits Sharpe 1.44 with MDD only -18%. The
gap is dominated by **risk-model fragility**, not predictive quality
of the signal.

Hierarchical Risk Parity (López de Prado 2016) is a structurally
different way of constructing risk-parity weights that bypasses the
single biggest failure mode of mean-variance:

> Markowitz inverts an N×N covariance. The inverse amplifies the
> noise eigendirections (the M-P bulk) by 1/λ — small eigenvalues
> dominate. HRP never inverts.

Instead it:

1. Converts the correlation matrix into a proper metric
   `d_ij = √(½(1 − ρ_ij))`.
2. Single-linkage clusters assets into a tree (≈ the minimum spanning
   tree of the correlation graph — a physics-style "skeleton" of
   pairwise dependencies).
3. Allocates risk-parity weights *recursively* down the tree,
   splitting each branch proportional to its inverse cluster variance.

The procedure is purely topological + inverse-variance; no matrix
inversion ever happens.

## Implementation

- `src/afp/portfolio/hrp_allocator.py`:
  - `_corr_distance` — proper correlation-distance metric.
  - `_single_linkage` — pure-numpy single-linkage agglomerative
    clustering (no SciPy dependency).
  - `_quasi_diagonal_order` — leaf ordering from the linkage tree.
  - `_recursive_bisection` — risk-parity bisection.
  - `hrp_weights(cov)` — orchestrator.
  - `hrp_allocation(...)` / `build_hrp_target_portfolio(...)` —
    drop-in allocator that composes HRP with the same signal tilt
    (`exp(β·μ)`) and caps / sector / cash rules used by Phase 38.

The covariance fed into HRP is the **RMT-cleaned** matrix from Phase
38 — they compose orthogonally: Phase 38 denoises the matrix; Phase
42 avoids inverting it.

## Status

Modules + 4 unit tests landed. Evaluated as part of the v2 run.

## Result

**hrp_rmt: Sharpe 0.830, MDD -0.399, Calmar 0.491**

Underperforms baseline (1.018) and far below rmt_minvar (1.096). The
hierarchical bisection bypasses Σ⁻¹ but also discards the information
about *which* eigendirections carry signal — the signal-tilt
multiplier `exp(β·μ)` cannot recover that lost information. In this
universe HRP is **dominated** by the direct RMT min-var solution.

We retire HRP for this pipeline.
