# Phase 55 — Vol-aware adaptive DD trigger (RETIRED)

## Hypothesis

Replace the static `dd_trigger=0.10` (Phase 44) with a vol-adaptive
formula:

```
trigger(t) = clip( z · σ_60d · sqrt(252/12) , 0.05, 0.20 )
```

with `z=2.0` (standard 2σ VaR event), `σ_60d` the rolling 60-day
strategy vol, and the horizon set to 1 month.

Theory: in high-vol regimes, larger drawdowns are statistically
"normal" within a 1-month horizon, so the trigger should rise to
match. In low-vol regimes, the same DD is more anomalous, so the
trigger should fall.

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| rmt_barbell + dd_target static (Phase 54 winner) | **1.079** | **-0.286** | **0.875** |
| rmt_barbell + dd_target adaptive trigger | 0.987 | -0.366 | 0.728 |

**Verdict: retired.** Adaptive trigger is strictly worse on all axes.

## Why it failed

The theoretical mechanism (high-vol regime → larger trigger) has the
**opposite** practical effect from what we want:

- In high-vol regimes (e.g., 2020 COVID, 2022 bear), the trigger
  rises (e.g., from 10% toward 20%) → DD-target wrapper de-risks
  LATER, allowing more loss to accumulate.
- In low-vol regimes (e.g., 2017-style trending market), the trigger
  falls (e.g., to 5%) → wrapper de-risks earlier, costing return.

The static 10% threshold avoids both pathologies. The VaR theoretical
framework "what is a 2σ event over 1 month" is not the right framing
for drawdown protection — we want the OPPOSITE response (tighter
when stressed, looser when calm). Phase 39 already discovered this
sign-flip on a different axis (cross-sectional dispersion); Phase 55
confirms it again.

## Discipline

- Parameters pre-declared from VaR theory (z=2, 1-month horizon).
- Single-pass evaluation.
- Honest retirement; we do not flip the sign on the formula after
  seeing the result.

## Status

`rmt_barbell + dd_target` (static 10% trigger, Phase 54) remains the
unified Pareto winner.
