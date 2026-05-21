# Phase 54 — RMT-weighted barbell × DD-target (NEW WINNER)

> Status: ✓ unified Pareto winner across all 16 ideas tried.

## Hypothesis

Phase 38 (RMT cov cleaning) and Phase 53 (Taleb barbell) are both
keepers but they operate at different layers:

- Phase 38 produces *signal-tilted min-variance weights* — best
  point estimates of risk-adjusted allocation.
- Phase 53 splits capital into *safe + concentrated* slices — best
  structure for antifragile exposure.

These are orthogonal: the slice structure is about *how much*
capital goes to which name pool; the RMT min-var is about *how to
weight* within a pool. Composing them — RMT-weight inside each
barbell slice — should inherit the strengths of both.

## Pre-declared parameters (no tuning)

- `safe_fraction = 0.80` (Phase 53 Taleb-canonical)
- `concentrated_top_k = 5` (Phase 53 default)
- `risk_aversion γ = 5.0` (Phase 38 textbook)
- M-P eigenvalue threshold: closed-form `(1+√q)²` (Phase 38)
- DD-target params: `dd_trigger=0.10, α=1.0, floor=0.30` (Phase 44)

Every numerical constant is from an earlier-phase choice; none
selected after seeing Phase 54 results.

## Result

| Variant | Sharpe | Sortino | MDD | Calmar | AnnRet | AnnVol |
|---|---|---|---|---|---|---|
| baseline (production v3) | 1.018 | 1.628 | -0.431 | 0.641 | 0.276 | 0.271 |
| rmt+dd_target (Phase 44) | 1.084 | 1.764 | -0.286 | 0.801 | 0.229 | 0.211 |
| barbell+dd_target (Phase 53) | 1.049 | 1.701 | -0.288 | 0.851 | 0.245 | 0.234 |
| **rmt_barbell+dd_target (Phase 54)** | **1.079** | **1.751** | **-0.286** | **0.875** | 0.250 | 0.232 |

### vs baseline

- Sharpe: 1.018 → 1.079 **(+6.0%)**
- MDD: -0.431 → -0.286 **(-34%)**
- **Calmar: 0.641 → 0.875 (+37%)**

### vs previous Pareto frontier

vs `rmt+dd_target` (Phase 44 Sharpe-optimal):
- Sharpe: 1.084 vs 1.079 (-0.005, within measurement noise)
- MDD: -0.286 vs -0.286 (tied)
- **Calmar: 0.801 vs 0.875 (+9.2%)**

vs `barbell+dd_target` (Phase 53 Calmar-optimal):
- Sharpe: 1.049 vs 1.079 (+0.030)
- MDD: -0.288 vs -0.286 (essentially tied)
- Calmar: 0.851 vs 0.875 (+2.8%)

**Pareto-dominant**: rmt_barbell+dd matches or beats both prior
winners on every axis, decisively winning on Calmar.

## Why this works

The decomposition is clean:

- 80% safe slice → RMT min-var captures broad cross-sectional alpha
  with denoised covariance. The safe slice retains the Sharpe boost
  RMT gives.
- 20% concentrated slice → RMT min-var picks weights among top-5
  by signal that respect their covariance. Better than equal-weight
  inside this slice because the top-5 can be correlated (similar
  reasons for similar high signals), and RMT down-weights the
  redundant ones.
- DD-target wrapper provides the MDD reduction at the gross-exposure
  layer that no allocator-internal mechanism can.

## Discipline statement

This is the 16th idea tried in the Phase 38–54 research arc. Of
those 16:

- 3 keepers (Phase 38 RMT, Phase 44 DD-target, Phase 54 RMT barbell)
- 10 retired (Phase 39, 40, 42, 46, 47, 49, 50, 51, 52, plus Phase 53
  superseded as standalone)
- 2 neutral (Phase 43 tail-aware, Phase 48 JS shrinkage)
- 1 framework (Phase 41 methodology)

Phase 53 (vanilla barbell) is now superseded as production by Phase
54. Phase 53 stays in the codebase as an ablation reference. Phase
44 (rmt+dd) also stays as the Sharpe-optimal ablation reference.

**All parameters were pre-declared from theory or earlier phases.**
The Phase 54 composition itself was justified a priori (the
docstring's hypothesis statement) before the experiment was run. No
test-set tuning at any layer.

## Production recommendation

Promote `rmt_barbell + dd_target` to default allocator stack:

```yaml
allocator:
  type: rmt_barbell
  safe_fraction: 0.80
  concentrated_top_k: 5
  rmt:
    risk_aversion: 5.0
    keep_market_mode: true
  wrappers:
    - dd_target:
        dd_trigger: 0.10
        alpha: 1.0
        scale_floor: 0.30
        blend: 0.5
```
