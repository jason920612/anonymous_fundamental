# Phase 11 — lgbm_v1000_improved Summary
## Strategy vs Benchmarks
- **Strategy**: Sharpe 0.67 | Ann.Ret 13.3% | Ann.Vol 19.9% | MDD -38.2%
- Equal-risk **active universe** (strictest): Sharpe 1.44 | Ann.Ret 24.5% | Ann.Vol 17.0% | MDD -18.0%
- Equal-weight active universe: Sharpe 0.96 | Ann.Ret 21.7% | Ann.Vol 22.6% | MDD -39.8%
- Equal-risk full universe: Sharpe 1.64 | Ann.Ret 29.9% | Ann.Vol 18.2% | MDD -18.2%
- Equal-weight full universe: Sharpe 1.06 | Ann.Ret 21.3% | Ann.Vol 20.1% | MDD -34.8%
- Dollar-volume weighted: Sharpe 1.19 | Ann.Ret 27.1% | Ann.Vol 22.8% | MDD -34.1%
- Random positive signal: Sharpe 1.50 | Ann.Ret 25.9% | Ann.Vol 17.3% | MDD -17.6%
- Shuffled signal: Sharpe 1.21 | Ann.Ret 27.4% | Ann.Vol 22.7% | MDD -29.0%
- SPY: Sharpe 0.77 | Ann.Ret 15.7% | Ann.Vol 20.4% | MDD -33.7%
- QQQ: Sharpe 0.87 | Ann.Ret 21.6% | Ann.Vol 24.9% | MDD -34.8%

## Decision
**A — Strategy does NOT beat equal-risk-active baseline.**

Per the Phase 11 decision rule: do not tune advanced models, do not add autoencoder, do not add diffusion. Improve data, universe, target, or portfolio design first.

## Signal Diagnostics (test split)
- top minus bottom decile mean realized log return: **-0.0062**
- positive-signal vs random-positive baseline spread: **0.0000** (t≈2.90, n_trials=25)

### Decile table
| Decile | mean pred | mean realized log ret | count |
|---|---:|---:|---:|
| D1 | -0.362 | +0.0337 | 1581 |
| D2 | -0.190 | +0.0329 | 1580 |
| D3 | -0.113 | +0.0219 | 1580 |
| D4 | -0.056 | +0.0277 | 1581 |
| D5 | -0.005 | +0.0287 | 1580 |
| D6 | +0.044 | +0.0409 | 1580 |
| D7 | +0.092 | +0.0298 | 1581 |
| D8 | +0.145 | +0.0338 | 1580 |
| D9 | +0.212 | +0.0313 | 1580 |
| D10 | +0.341 | +0.0275 | 1581 |

### Quarterly Spearman: mean **+0.005** | 11/25 quarters positive

### Quarterly direction accuracy: mean **52.3%**

## Portfolio Attribution
- avg cash weight: **0.5%**
- max cash weight: **100.0%**
- annualized turnover: **7.4×**
- cost drag/year: **1.48%**

### Top 10 Contributors
| internal_company_id | total_contribution |
|---|---:|
| COMP0001050446 | +0.0629 |
| COMP0001035983 | +0.0328 |
| COMP0001341439 | +0.0305 |
| COMP0001137789 | +0.0295 |
| COMP0001408710 | +0.0289 |
| COMP0000006951 | +0.0272 |
| COMP0000106040 | +0.0237 |
| COMP0000050863 | +0.0237 |
| COMP0000820318 | +0.0225 |
| COMP0000033213 | +0.0216 |

### Concentration
- mean Herfindahl: **0.0117** (>0.10 means top-heavy)
- mean top-10 weight share: **18.6%**

## Caveats
- survivorship handling: imperfect / best-effort (universe drawn from current SEC ticker file)
- price source: yfinance — research prototype quality, not production grade
- test window: 2020-01 → 2026-05; longer/sub-period decomposition pending
- restatements: as-of-download snapshots only
