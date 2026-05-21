# Phase 44 — Drawdown-targeting exposure wrapper (next experiment)

## Context (what Phase 41-43 told us)

Across 12 allocator variants (RMT, HRP, tail-aware, max-entropy,
regime filter, baseline), **every** strategy has MDD in the range
**-0.40 to -0.45**. The MDD floor is independent of the allocator —
it is set by the structural facts of (1) long-only exposure, (2)
2020 COVID + 2022 bear market events, (3) the underlying signal's
beta-1 nature.

Therefore: changing how risk is distributed across names will not
materially change MDD. To reduce MDD we must change the **gross
exposure as a function of realized state**.

## Approach: drawdown-conditioned exposure

Track the realized strategy drawdown over a rolling window:

```
DD(t) = (V(t) - max V[t-W:t]) / max V[t-W:t]          (≤ 0)
```

When `|DD(t)|` exceeds a threshold, scale gross exposure toward
cash. Formally:

```
exposure(t) = clip( 1 - α * max(|DD(t)| - DD_trigger, 0) / DD_trigger ,
                    floor, 1.0 )
```

This is purely *reactive* — no forecast, no peek at future. The
two parameters:

- `DD_trigger`: threshold at which de-risking starts. Set to **0.10**
  — historically the 25th percentile of major equity drawdowns
  before recovery patterns shift.
- `α`: rate of de-risking. Set to **1.0** — at 2×DD_trigger (i.e.
  -20% DD), exposure is fully at the floor.
- `floor`: minimum exposure. Set to **0.30** — same as Phase 39, to
  prevent total capitulation on whipsaws.

All three are pre-declared from theory; we will not tune on val/test.

## Why this might work (theory)

Two physics analogies:

1. **Negative feedback dampening**: classical control theory.
   Drawdown is an error signal; cutting exposure is the corrective
   action. Provided the corrective action is *proportional* and
   doesn't overshoot, it reduces oscillation amplitude.

2. **Energy dissipation in a damped oscillator**: portfolio equity
   trajectory is a noisy harmonic oscillator around the long-term
   drift. Drawdown corresponds to kinetic energy peaks; coupling
   exposure to recent drawdown is analogous to viscous damping —
   it preferentially removes the high-frequency oscillation
   amplitude (large drawdowns) without much affecting the long-
   term drift (annual return).

## Predicted side-effect

Expected behaviour:

- Sharpe: roughly unchanged or slightly reduced. De-risking near
  drawdown bottoms occasionally misses the recovery, but rest of
  the time exposure is at 100%.
- MDD: meaningfully reduced — that's the whole point.
- Calmar: should *improve* because Calmar = return / |MDD| and we
  trade a fraction of return for a larger fraction of MDD.

## Implementation plan

Module: `src/afp/portfolio/dd_target.py` — pure config + a
`compute_dd_scale(daily_returns_so_far) → float` function, modeled
after `vol_target.compute_portfolio_vol_scale`.

Integration: extend the backtest engine with a single new optional
kwarg `dd_target_cfg`, applied AFTER the existing `regime_filter`
and `vol_target` scaling. Multiplicative; stacks cleanly.

Evaluation: `scripts/run_phys_v3_experiments.py` — 3 variants:
- baseline + dd_target
- rmt_minvar + dd_target
- rmt_minvar + dd_target + vol_target  (composes wrappers)

Methodology discipline: parameters pre-declared above; eval is
single-pass on the 2020-2026 window.

## Results

Backtest window 2020-01-02 → 2026-03-18 (same as Phase 41), 10 bps
per-trade cost, frozen `lambdarank_v3_p100_s42.parquet` predictions.

| Variant | Sharpe | Sortino | MDD | Calmar | AnnRet | AnnVol |
|---|---|---|---|---|---|---|
| baseline (no wrapper) | 1.018 | 1.628 | -0.431 | 0.641 | 0.276 | 0.271 |
| rmt_minvar (no wrapper) | 1.096 | 1.768 | -0.446 | 0.727 | 0.324 | 0.296 |
| baseline + dd_target | 0.906 | 1.446 | -0.285 | 0.622 | 0.177 | 0.195 |
| **rmt_minvar + dd_target** | **1.084** | **1.764** | **-0.286** | **0.801** | 0.229 | 0.211 |
| rmt_minvar + dd_target + vol_target | 1.028 | 1.687 | -0.258 | 0.798 | 0.206 | 0.200 |

### Headline outcome

**`rmt_minvar + dd_target`** vs baseline:

- Sharpe: 1.018 → 1.084 (+6.5%)
- MDD: -0.431 → -0.286 (**-34%**)
- **Calmar: 0.641 → 0.801 (+25%)**

Both axes improve simultaneously. The combination is non-trivial:
- RMT (Phase 38) alone gives +Sharpe but hurts MDD slightly.
- DD-target alone on baseline gives -MDD but hurts Sharpe.
- Stacking them: the DD-target compensates for the slight MDD
  worsening from RMT, and the RMT Sharpe gain compensates for the
  Sharpe drag from DD-target. Net: better on both.

### Adding vol-target

The third variant adds vol-targeting on top. MDD drops further to
-0.258 (best of any variant tested), but Sharpe gives up 0.056.
Calmar is essentially tied at 0.798. Whether the extra MDD reduction
is worth the Sharpe cost is a deployment-policy choice; the
single-wrapper `rmt_minvar + dd_target` is the cleaner default.

## Discipline check

- All 3 DD-target parameters were pre-declared in this doc (`dd_trigger=0.10`,
  `α=1.0`, `floor=0.30`). No tuning iteration was run; this is the
  one-shot result.
- The vol-target parameters were *also* pre-declared (target 20%,
  min/max 0.30/1.10). No sweep.
- We do not commit to a γ ≠ 5 for the RMT layer despite γ=2
  scoring higher in the γ-sensitivity check — that would be test-set
  selection. γ=5 stays as the textbook default.

## Production recommendation

Promote `rmt_minvar + dd_target` to the default allocator stack:

```yaml
allocator:
  type: rmt_minvar
  rmt:
    risk_aversion: 5.0     # textbook default; γ-sensitivity check spans 1.04-1.12 Sharpe
    keep_market_mode: true
  wrappers:
    - dd_target:
        dd_trigger: 0.10
        alpha: 1.0
        scale_floor: 0.30
        blend: 0.5
```
