# Phase 50 — Vol-of-volatility target (RETIRED)

## Hypothesis

GARCH-style heteroscedasticity: vol is itself volatile. A rolling
z-score of `vov(t) = std(σ_t)` should fire EARLIER than reactive
drawdown-tracking when regime instability is brewing, allowing
pre-emptive de-risking.

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| rmt_minvar + vov_target | 0.951 | -0.446 | 0.616 |
| rmt_minvar + dd + vov | 0.852 | -0.286 | 0.576 |
| (reference: rmt_minvar+dd_target) | **1.084** | **-0.286** | **0.801** |

**Verdict: retired.** Both vov-target variants under-perform; adding
vov on top of dd_target makes it strictly worse on Sharpe and Calmar.

## Why it failed

The rolling `vov(t)` series is itself noisy — small-sample
estimation error in the 20d rolling σ propagates through into the
60d rolling std-of-σ. The resulting z-score wanders ±1σ even under
stationary noise, triggering unnecessary exposure cuts during
periods that turn out to be opportunities.

Adding vov on top of dd_target stacks two reactive damping mechanisms
that fire on overlapping (but not identical) sets of days, giving up
return without gaining additional MDD protection.

## Discipline

- Boltzmann form `exp(-max(0, z) / kT)` with `kT=2.0` mirrors the
  Phase 39 regime filter — same theoretical envelope.
- Pre-declared windows (σ=20d, vov=60d) from standard GARCH practice.
- Single-pass evaluation; no re-runs or threshold tweaks.
