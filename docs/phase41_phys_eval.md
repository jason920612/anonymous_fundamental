# Phase 41 — Single-pass evaluation: physics-inspired allocators

> Methodology: NO val/test tuning. Each allocator's hyperparameters
> were set a priori from theory (see Phases 38-40 docs). One backtest
> per variant, no re-runs.

## Setup

- **Predictions**: frozen `lambdarank_v3_p100_s42.parquet` from the
  Phase 33 honest-methodology run (Sharpe 1.02 on the
  2020-2026 backtest window).
- **Universe**: same 1000-CIK training pool — no new model training.
- **Transaction cost**: 10 bps per trade (existing config).
- **Window**: 2020-01-02 → 2026-03-18 (val+test combined per the
  existing config — the only "production" backtest window we have).

## Variants

| # | Variant | What's different |
|---|---|---|
| A | `baseline_inverse_vol` | The current Phase 26 production allocator |
| B | `rmt_minvar` | Phase 38 RMT-cleaned signal-tilted min-variance |
| C | `max_entropy` | Phase 40 Boltzmann allocator, `inverse_vol_blend=0.5` |
| D | `max_entropy_lowvol_blend` | Phase 40 with `inverse_vol_blend=1.0` |
| E | `baseline + regime` | A multiplied by the Phase 39 daily exposure scale |
| F | `rmt_minvar + regime` | B + Phase 39 |
| G | `max_entropy + regime` | C + Phase 39 |
| H | `max_entropy_lowvol + regime` | D + Phase 39 |

## Results

Backtest window 2020-01-02 → 2026-03-18 (1603 trading days), 10 bps
per-trade cost, identical predictions across all 8 variants.

| # | Variant | Sharpe | Sortino | MDD | Calmar | AnnRet | AnnVol |
|---|---|---|---|---|---|---|---|
| **1** | **rmt_minvar** | **1.096** | **1.768** | -0.446 | **0.727** | 0.324 | 0.296 |
| 2 | rmt_minvar+regime | 1.062 | 1.694 | -0.445 | 0.656 | 0.292 | 0.275 |
| 3 | baseline_inverse_vol | 1.018 | 1.628 | -0.431 | 0.641 | 0.276 | 0.271 |
| 4 | baseline+regime | 0.983 | 1.553 | -0.431 | 0.577 | 0.248 | 0.253 |
| 5 | max_entropy | 0.830 | 1.307 | -0.402 | 0.489 | 0.197 | 0.237 |
| 6 | max_entropy+regime | 0.796 | 1.240 | -0.402 | 0.440 | 0.177 | 0.222 |
| 7 | max_entropy_lowvol_blend | 0.774 | 1.209 | -0.395 | 0.449 | 0.177 | 0.229 |
| 8 | max_entropy_lowvol+regime | 0.739 | 1.140 | -0.395 | 0.401 | 0.158 | 0.214 |

## Findings

### ✓ RMT covariance cleaning works (Phase 38)

The Marchenko-Pastur noise filter pushed Sharpe from 1.018 → 1.096
(+0.078, +7.7%) and Calmar from 0.641 → 0.727. Annualized return
jumped from 27.6% → 32.4% — the cleaned Σ⁻¹ inside Markowitz
concentrated weight on names with informative covariance structure
rather than the high-variance bulk of M-P noise eigenmodes.

The MDD went slightly worse (-0.446 vs -0.431) — RMT does **not**
target drawdown directly. The Sharpe gain came from return, not
volatility.

### ✗ Thermodynamic regime filter hurts consistently (Phase 39)

The cross-sectional dispersion / Boltzmann-damping idea was
theoretically appealing but reduces Sharpe on **every** allocator
it's combined with by ~0.035. The mechanism failure: in this
universe, the major drawdown periods (COVID March 2020, 2022 bear)
were characterized by *high* cross-sectional dispersion (different
stocks reacted very differently), not low. The "cold = correlated =
avoid" mapping does not match this universe's MDD structure.

The detector should be retired for this prediction model unless
re-formulated with a different temperature observable.

### ✗ Max-entropy / Boltzmann allocator is too defensive (Phase 40)

`target_positions=50` produces an N_eff ≈ 50, which is significantly
more diversified than the current allocator's effective concentration.
The result is lower per-position bet sizes and a structural Sharpe
penalty of ~0.18. The inverse-vol blend (`blend=1.0`) makes it worse,
not better. MDD does improve slightly (-0.40 vs -0.43) but the cost
in return is too high.

## Discipline check

- No parameter was tuned to val/test results. Phase-38 γ=5.0 is a
  textbook default; Phase-39 kT=2.0 is a theoretical
  `exp(-0.5)≈0.61` at 1σ; Phase-40 β is solved structurally from the
  existing `target_positions` config.
- Eight variants were run a single time. No re-runs to "find" better
  parameters.
- Two of the three ideas (regime filter, max-entropy) **failed**
  against the baseline; we accept that result and retire those paths
  rather than tweak them.

## Conclusion + next direction

**Keep**: Phase 38 (RMT covariance cleaning) — promote to a primary
allocator option in the next deployment.

**Retire**: Phase 39 (regime filter) and Phase 40 (max-entropy).

**Open question**: how to reduce MDD without sacrificing the Sharpe
gain from RMT. The Phase-42 (HRP) and Phase-43 (tail-aware ES)
candidates were prepared in parallel for this exact reason. The v2
evaluation (`scripts/run_phys_v2_experiments.py`) runs HRP + tail-
aware + RMT γ-sensitivity (4 variants) under the same single-pass
methodology.

## Discipline statement

Each parameter has a *prior* justification (Phase 38-40 docs); none
were tuned to the test window. If the leaderboard shows no
improvement, we accept that result. If it does improve, we accept
that result. No iteration is allowed within Phase 41.

If a follow-up phase (e.g., 42) wishes to combine the *winners* of
this phase with **other** changes, that new phase will repeat the
"set parameters from theory then evaluate once" rule.
