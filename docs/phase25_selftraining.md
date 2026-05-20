# Phase 25 — Self-Training with Backtest Reward

## What It Does

1. Train diffusion v1/v2 on the train split (cross-validated by calendar
   quartile to avoid trivial leakage).
2. For each train sample, compute reward:
   ```
   agreement_i = +1 if sign(predicted_i) == sign(realized_i), else -1
   confidence_i = |probability_positive_i - 0.5| * 2
   reward_i = agreement_i * confidence_i * |realized_log_return_i|
   ```
   - Correctly-predicted, high-conviction events score positively.
   - Confidently-wrong predictions score negatively.
   - Low-confidence predictions stay near zero.

3. Convert reward → sample weight:
   ```
   weight_i = exp(reward_scale * reward_i)
   weights renormalized to mean 1
   ```
   `reward_scale = 8.0` by default. A correct, high-conviction 5%
   move gets weight ≈ exp(0.4) ≈ 1.49; a confidently wrong 5% move
   gets weight ≈ exp(-0.4) ≈ 0.67.

4. Retrain diffusion v3 on the full train set with these weights
   (2× the per-fold epoch budget so the final model sees comparable
   gradient updates).

The leave-one-quartile-out predictions in step 1 prevent the reward
from rewarding overfit predictions of each sample's own target.

## RFC Compliance

- The model still sees only anonymous features.
- Reward uses **realized log return** from the training samples — that's
  point-in-time data the train split already has access to. No future
  / test information leaks.
- Test data is never used in any of the 5 sub-fits.

## Result on v1000

Diffusion v3 self-train (vs v2):

| Allocator | v2 Sharpe | v3 Sharpe | v2 MDD | v3 MDD |
|---|---:|---:|---:|---:|
| inverse_vol | 0.66 | 0.68 | -36.3% | -34.9% |
| distribution, p≥0.55 | 0.79 | 0.75 | -35.8% | -37.2% |

- Self-training slightly improved inverse-vol Sharpe (0.66 → 0.68) and
  notably improved **MDD** (-36.3% → -34.9%, the lowest of any variant
  in the leaderboard).
- With the distribution allocator, however, v3 underperformed v2 (0.79 →
  0.75). The reward-weighted retraining smoothed the probability_positive
  distribution too aggressively for the allocator to discriminate.
- Net: self-training is **a drawdown reducer**, not a Sharpe improver,
  on this universe.

## Honest Assessment

The reward signal is too noisy at this universe scale (1842 train rows ×
4818 features) for the reweighting to break through the inherent
information bound of fundamentals. The MDD improvement is real but small.

A more promising version would be:
- run actual backtest, compute per-quarter portfolio Sharpe as the
  reward, then reweight samples by their *contribution* to those
  Sharpe quarters
- iterate over multiple rounds with decay
- or use REINFORCE policy-gradient on the actual portfolio weight
  function

We didn't pursue those because they'd dwarf the entire codebase in
complexity for what diagnostics suggest is a fundamentally
information-bound problem.

## Acceptance

- `scripts/run_v1000_diffusion_selftrain.py` runs end-to-end on GPU
  (5 fits total) and writes `artifacts/predictions/diffusion_v3_self.parquet`
  with the same schema as v2.
- Backtest CLI works against it with both `--allocator inverse_vol` and
  `--allocator distribution`.
