# Phase 56 — Cross-seed robustness validation of Phase 54 winner

> Status: ✓ confirmed across 3 independent LambdaRank seeds.

## Question

Phase 54 declared `rmt_barbell + dd_target` as a Pareto-dominant
winner (Sharpe 1.079, MDD -0.286, Calmar 0.875) on the s42 predictions.
Is this:

a) A real structural improvement that generalizes to other seeds, or
b) A lucky alignment with seed 42's specific prediction noise?

If (b), the improvement would not be reproducible in production. We
need to falsify (b) before promoting the stack.

## Method

Re-run the Phase 54 stack (rmt_barbell allocator + dd_target wrapper)
on TWO additional LambdaRank seeds (s7, s99). All pre-declared
parameters identical. Compare each seed's winner vs its own baseline
inverse-vol.

The s7 and s99 predictions use `max_position=50` (vs s42's `=100`),
so the *absolute* Sharpe levels differ. We compare relative
improvements (Δ%) instead of absolutes.

## Results

| Seed | Baseline | rmt_barbell+dd | Δ Sharpe | Δ MDD | Δ Calmar |
|---|---|---|---|---|---|
| s7 | 0.958 / -0.418 / 0.621 | **1.134** / **-0.272** / **0.937** | +18.4% | -35% | +51% |
| s42 (headline) | 1.018 / -0.431 / 0.641 | 1.079 / -0.286 / 0.875 | +6.0% | -34% | +37% |
| s99 | 0.965 / -0.420 / 0.606 | 1.127 / -0.274 / 0.864 | +16.8% | -35% | +43% |
| **Average** | — | — | **+13.7%** | **-35%** | **+43.7%** |

## Conclusion

✓ **The Phase 54 effect is HIGHLY consistent across seeds.**

- MDD reduction is essentially identical across all three seeds
  (-34/-35/-35%). The drawdown protection is structural, not noise.
- Sharpe improvement varies (6-18%) but is positive in all three.
- Calmar improvement is large and consistent (+37 to +51%).

Note: s42 baseline (Sharpe 1.018) is the strongest of the three;
s7 and s99 baselines are weaker (~0.96), giving more room for
relative improvement. The winner stack converges to a tight range
(1.08–1.13 Sharpe) across all three seeds — the improvement
"funnels" all seeds toward a similar high-Sharpe regime.

## Discipline

- All parameters were the SAME pre-declared values used in Phase 54.
- No tuning to s7 or s99 results.
- The s7/s99 prediction parquets pre-existed (from earlier seed-
  stability experiments in Phase 29-33); they are not new training
  runs.
- Equivalent test set as before (2020-2026); the only change is the
  source of predicted_signal.

## Production status

Phase 54's `rmt_barbell + dd_target` is now **validated as
seed-robust**. We promote it as the new production allocator stack
with high confidence that the improvement is reproducible.
