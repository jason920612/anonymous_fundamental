# Phase 11 — lgbm_v1000_static Summary
## Strategy vs Benchmarks
- **Strategy**: Sharpe 0.72 | Ann.Ret 14.2% | Ann.Vol 19.6% | MDD -37.2%
- Equal-risk **active universe** (strictest): Sharpe 1.48 | Ann.Ret 25.3% | Ann.Vol 17.1% | MDD -17.9%
- Equal-weight active universe: Sharpe 1.00 | Ann.Ret 22.8% | Ann.Vol 23.0% | MDD -39.6%
- Equal-risk full universe: Sharpe 1.64 | Ann.Ret 29.9% | Ann.Vol 18.2% | MDD -18.2%
- Equal-weight full universe: Sharpe 1.06 | Ann.Ret 21.3% | Ann.Vol 20.1% | MDD -34.8%
- Dollar-volume weighted: Sharpe 1.19 | Ann.Ret 27.1% | Ann.Vol 22.8% | MDD -34.1%
- Random positive signal: Sharpe 1.42 | Ann.Ret 24.8% | Ann.Vol 17.5% | MDD -23.2%
- Shuffled signal: Sharpe 1.30 | Ann.Ret 32.9% | Ann.Vol 25.3% | MDD -29.5%
- SPY: Sharpe 0.77 | Ann.Ret 15.7% | Ann.Vol 20.4% | MDD -33.7%
- QQQ: Sharpe 0.87 | Ann.Ret 21.6% | Ann.Vol 24.9% | MDD -34.8%

## Decision
**A — Strategy does NOT beat equal-risk-active baseline.**

Per the Phase 11 decision rule: do not tune advanced models, do not add autoencoder, do not add diffusion. Improve data, universe, target, or portfolio design first.

## Signal Diagnostics (test split)
- top minus bottom decile mean realized log return: **-0.0050**
- positive-signal vs random-positive baseline spread: **-0.0000** (t≈-4.96, n_trials=25)

### Decile table
| Decile | mean pred | mean realized log ret | count |
|---|---:|---:|---:|
| D1 | -0.241 | +0.0333 | 1696 |
| D2 | -0.147 | +0.0305 | 1695 |
| D3 | -0.100 | +0.0367 | 1696 |
| D4 | -0.063 | +0.0353 | 1695 |
| D5 | -0.031 | +0.0327 | 1696 |
| D6 | +0.002 | +0.0244 | 1695 |
| D7 | +0.033 | +0.0269 | 1695 |
| D8 | +0.069 | +0.0315 | 1696 |
| D9 | +0.115 | +0.0424 | 1695 |
| D10 | +0.197 | +0.0283 | 1696 |

### Quarterly Spearman: mean **-0.008** | 12/25 quarters positive

### Quarterly direction accuracy: mean **49.6%**

## Portfolio Attribution
- avg cash weight: **0.5%**
- max cash weight: **95.3%**
- annualized turnover: **7.6×**
- cost drag/year: **1.51%**

### Top 10 Contributors
| internal_company_id | total_contribution |
|---|---:|
| COMP0001849056 | +0.0840 |
| COMP0001674101 | +0.0400 |
| COMP0001375365 | +0.0395 |
| COMP0001326380 | +0.0376 |
| COMP0000936395 | +0.0355 |
| COMP0001035983 | +0.0300 |
| COMP0000874238 | +0.0296 |
| COMP0001837240 | +0.0282 |
| COMP0000034088 | +0.0253 |
| COMP0000107263 | +0.0249 |

### Concentration
- mean Herfindahl: **0.0123** (>0.10 means top-heavy)
- mean top-10 weight share: **20.1%**

## Caveats
- survivorship handling: imperfect / best-effort (universe drawn from current SEC ticker file)
- price source: yfinance — research prototype quality, not production grade
- test window: 2020-01 → 2026-05; longer/sub-period decomposition pending
- restatements: as-of-download snapshots only
