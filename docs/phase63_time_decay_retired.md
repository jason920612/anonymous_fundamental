# Phase 63 — Time-decay signal (RETIRED)

## Hypothesis (first principles)

The signal is generated at filing time. Market participants
gradually incorporate the same fundamental information into prices.
By the time the next filing arrives, the current signal is fully
stale. Therefore: weight the signal exponentially by exp(-age/τ)
with τ = 63 days (one trading quarter, median filing cadence).

## Result

| Universe | Phase 54 (no decay) | Phase 63 (with decay) | Δ Sharpe |
|---|---|---|---|
| tier1 s42 | 1.079 / -0.286 / 0.875 | 0.792 / -0.294 / 0.559 | **-0.29** |
| tier2 | 1.040 / -0.294 / 0.832 | 0.667 / -0.286 / 0.526 | **-0.37** |
| tier3 | 0.717 / -0.290 / 0.554 | 0.643 / -0.293 / 0.470 | -0.07 |

**Verdict: retired across all 3 universes.**

## Why first principles was wrong

The hypothesis assumed signal information decays with time. In
reality, the fundamental data point (a quarterly earnings filing)
remains informative until the next filing because:

1. The strategy explicitly rebalances at the *next* filing — by
   construction, the position is closed at exit_date when fresh
   info arrives.
2. Within the holding window, the original fundamental ratio is
   still the most recent verified data point.
3. Decaying the signal pushes capital toward cash unnecessarily,
   sacrificing alpha capture without commensurate risk reduction.

The first-principles claim "signal information decays" was
intuitive but EMPIRICALLY FALSE for event-driven fundamental
investing. The data wins over the prior.

## Lesson

First principles can mislead when applied without empirical
sanity check. The signal staleness intuition would apply if the
market reacted within hours/days to the filing — but with our
quarterly rebalance, the position IS the bet that the filing's
edge persists until next filing. Decaying erodes the very thing
we're betting on.

## Discipline

- τ=63d set from median filing cadence (theoretical, not tuned).
- Exponential decay form chosen for smooth degradation.
- Single-pass eval across 3 universes.
- Honest retirement: theoretically clean idea, empirically wrong.
