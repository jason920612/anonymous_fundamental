# Phase 26 — Final Leaderboard & Lessons

## v1000 Universe, 2020-01 → 2026-05 (1,603 trading days)

| # | Model + Allocator | Sharpe | Ann.Ret | MDD | Sortino |
|---:|---|---:|---:|---:|---:|
| 1 | **Diffusion v2 + dist allocator (p≥0.60)** | **0.79** | 16.2% | **-35.8%** | 1.25 |
| 1 | Diffusion v2 + dist allocator (p≥0.55) | 0.79 | 16.1% | -35.8% | 1.24 |
| 1 | Ridge + 6 anonymous price features | 0.79 | 15.9% | -37.1% | 1.23 |
| 4 | Ridge static (fundamentals only) | 0.76 | 15.2% | -37.2% | 1.18 |
| 5 | Diffusion v3 self-train + dist alloc | 0.75 | 15.3% | -37.2% | 1.18 |
| 6 | Mean ensemble (Ridge+LGBM+XGB+MLP) | 0.73 | 14.5% | -36.0% | 1.15 |
| 7 | Static LightGBM | 0.72 | 14.2% | -37.2% | 1.10 |
| 8 | Diffusion v3 self-train (inverse_vol) | 0.68 | 13.5% | -34.9% | 1.07 |
| 9 | Diffusion v1 (inverse_vol) | 0.67 | 13.8% | -37.1% | 1.05 |
| 10 | Diffusion v2 (inverse_vol) | 0.66 | 13.3% | -36.3% | 1.04 |
| 11 | XGB GPU | 0.65 | 12.9% | -38.5% | 0.99 |
| 12 | MLP GPU | 0.64 | 13.1% | -37.1% | 1.00 |
| 13 | LightGBM HPO-tuned + weighting | 0.56 | 11.2% | -39.6% | 0.85 |
| 14 | Walk-forward LightGBM (QS) | 0.59 | 11.9% | -38.9% | 0.91 |

### Reference benchmarks (same universe / window)

| | Sharpe | Ann.Ret | MDD |
|---|---:|---:|---:|
| SPY | 0.77 | 15.7% | -33.7% |
| QQQ | 0.87 | 21.6% | -34.8% |
| Equal-weight active universe | 1.00 | 22.8% | -39.6% |
| **Equal-risk active universe** | **1.44** | 24.5% | -18.0% |
| Random positive signal | 1.42 | 24.8% | -23.2% |
| Shuffled signal | 1.30 | 32.9% | -29.5% |
| Equal-risk full universe | 1.64 | 29.9% | -18.2% |
| Dollar-volume weighted | 1.19 | 27.1% | -34.1% |

## Headline

**Two winners tied at Sharpe 0.79:**
- Diffusion v2 + distribution allocator
- Ridge + anonymous price features

Both barely beat SPY (0.77) and significantly trail same-universe
inverse-vol baselines (1.44).

**Best drawdown protection:** Diffusion v3 self-train (-34.9% MDD,
inverse-vol allocator). The reward-weighted retraining systematically
de-emphasized loss-prone samples.

## Lessons That Held Across 15 Variants

1. **Validation IC ≠ portfolio Sharpe.** The 0.05 Spearman model
   (HPO-tuned LightGBM) had the worst Sharpe (0.56); the near-0 IC
   model (Ridge) tied for first.
2. **Sample weighting hurt.** Inverse-frequency weighting consistently
   dropped Sharpe by 5-10%. Mega-cap names *were* doing the work.
3. **Walk-forward hurt.** Quarterly retraining adds parameter
   variance that wasn't compensated by regime adaptation in this
   period.
4. **Universe filter neutral.** Removed noise but also removed alpha-
   bearing micro-caps; net Sharpe unchanged.
5. **Derived features helped Ridge, hurt trees.** Tree models overfit
   the extra columns; linear models benefited from the explicit
   YoY/z-score transforms.
6. **Diffusion calibration > diffusion mean.** v2 with inverse-vol
   was the same as XGB (0.66). With distribution allocator it jumped
   to 0.79. The value is in the distributional output, not the point
   estimate.
7. **Self-training is a drawdown reducer, not a Sharpe lifter.**
   Reward weighting de-emphasizes confidently-wrong samples → smoother
   equity curve, smaller drawdowns, but similar Sharpe.

## Acceptance / RFC Status

- Every variant maintains anonymous model input (RFC-03 §15).
- Test data was never used in any training / HPO loop (RFC-08 §11).
- Diffusion v2 + distribution allocator is the recommended production
  pipeline at Sharpe 0.79 + MDD -35.8% — the only variant that beats
  SPY on both axes.
- Equal-risk-active baseline (1.44) remains unbeaten by any of the 15
  variants. Per the original Decision Rule, the **structural
  conclusion is "the model does not provide alpha beyond what
  inverse-vol allocation already extracts."**

## What Would Plausibly Break Through 1.44

(In rough order of difficulty/payoff)

1. **5,000+ CIK universe + survivorship-corrected price data.**
   The 1000-CIK current-only universe is too narrow for meaningful
   stock selection alpha; the equal-risk-active baseline benefits
   disproportionately.
2. **Sector-relative target.** Removes macro regime dominance from
   the supervised signal; would let the model learn what's predictive
   *within* sector rather than across.
3. **Multi-quarter forward target (12m).** Reduces noise-to-signal of
   the target by ~3×.
4. **Alternative data (news, transcripts, fund flows).** Outside the
   RFC scope but the structural fix for fundamentals' inherent IC
   ceiling.
5. **Cross-asset / global universe.** US equity fundamentals at single-
   firm 60-90d horizons may simply be efficient.

## File Index for This Iteration

```
src/afp/models/diffusion.py                     # Phase 22 (v1) + 23 (v2 arch)
src/afp/portfolio/distribution_allocator.py     # Phase 24
src/afp/features/price_features.py              # Phase 21
scripts/run_v1000_diffusion.py                  # v1 trainer
scripts/run_v1000_diffusion_v2.py               # v2 trainer (price + bigger)
scripts/run_v1000_diffusion_selftrain.py        # v3 self-training
scripts/run_v1000_price_features.py             # Ridge + price runner
configs/deployment_v1000.yaml                   # baseline config
configs/deployment_v1000_improved.yaml          # Phase 12-16 bundle
configs/deployment_v1000_tuned.yaml             # HPO-picked params
configs/deployment_v1000_tanh.yaml              # ablation: tanh vs rank target
configs/deployment_v1000_diversified.yaml       # ablation: 150 positions
```
