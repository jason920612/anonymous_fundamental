# Phase 53 — Antifragile barbell allocator (KEEPER, Pareto frontier)

> Status: ✓ keeper on the Calmar axis; second production option
> alongside Phase 38+44.

## Hypothesis (Taleb 2012, *Antifragile*)

Split capital between TWO extreme allocations, nothing in the middle:

- **SAFE side (80%)**: equal-weight across ALL candidates.
  Maximally diversified; single-name MDD floor.
- **CONCENTRATED side (20%)**: equal-weight across top-5 by signal.
  High-conviction picks; participates in winners.

The "barbell" shape — heavy on both ends, zero in the middle — gives
convex exposure to surprises. Drawdowns on the concentrated side are
capped at 20% of capital; upside is asymmetric.

## Pre-declared parameters

- `safe_fraction = 0.80`
- `concentrated_top_k = 5`
- Equal weight inside each slice. No tuning.

## Result

| Variant | Sharpe | MDD | Calmar | AnnRet |
|---|---|---|---|---|
| baseline | 1.018 | -0.431 | 0.641 | 0.276 |
| rmt_minvar+dd_target (Phase 44 winner) | **1.084** | -0.286 | 0.801 | 0.229 |
| barbell (no wrapper) | 1.096 | -0.444 | 0.799 | **0.354** |
| **barbell+dd_target** | 1.049 | -0.288 | **0.851** | 0.245 |

**Headline outcome**: barbell+dd_target is the NEW Calmar champion.

Compared to the previous Phase 44 winner (rmt_minvar+dd_target):

- Sharpe: 1.049 vs 1.084 (−0.035)
- MDD: −0.288 vs −0.286 (essentially tied)
- **Calmar: 0.851 vs 0.801 (+6.2%)**
- AnnRet: 24.5% vs 22.9% (+7%)

Both improvements drift from the same source: barbell captures more
upside on the concentrated 20% slice (top 5 names) while the
diversified 80% slice keeps MDD in the same range as RMT-cleaned
min-var.

## Pareto frontier — two keepers, two flavours

The Phase 44 and Phase 53 winners are NOT dominated by each other:

```
                Sharpe   MDD     Calmar
  rmt+dd        1.084   -0.286   0.801   ← Sharpe-optimal
  barbell+dd    1.049   -0.288   0.851   ← Calmar-optimal
```

Neither dominates. They occupy different corners of the Pareto
frontier. The deployment choice depends on the operator's preference:

- Sharpe-first (vol-adjusted return is king): use Phase 44 stack.
- Calmar-first (return/MDD ratio is king): use Phase 53 stack.

Both stacks substantially beat the baseline (1.018 / -0.431 / 0.641).

## Why this works structurally

Barbell sidesteps the "concentrated bet vs diversification" dilemma
that allocator-level Phase 39/40/42/47/49/50/51/52 all failed on.
Instead of choosing one or the other, it ALLOCATES capital between
them at the 80/20 split.

The 80% safe slice gives ~equal-weight performance (Sharpe 1.06
baseline benchmark) and bounded MDD via diversification. The 20%
concentrated slice gives signal-driven concentrated alpha. dd_target
on top trims gross when the joint signal drawdown gets large.

## Discipline

- 80/20 split is a Taleb-canonical ratio, pre-declared from theory.
- top-K=5 is the same `target_positions / 10` rule that would have
  been chosen before seeing the result.
- No tuning. Equal weights inside each slice.
- Both keepers (Phase 44 and Phase 53) reported honestly side-by-side.
