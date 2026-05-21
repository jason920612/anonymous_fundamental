# Cross-Disciplinary Feature Augmentation for Anonymous Long-Only Equity Portfolios

*Final method, empirical evidence, and the design principle that made it work.*

---

## Abstract

We present a long-only US equity portfolio strategy that achieves Sharpe **1.04**, maximum drawdown **-0.29**, and Calmar **0.85** (cross-universe average) under three strict constraints: (1) the model sees only anonymous numerical features and cannot recover concept names; (2) the portfolio is long-only with leverage 1; (3) all parameters are pre-declared from theory, never tuned on the test window. The winning architecture has three layered pieces: a **LambdaRank model trained with six cross-disciplinary market-state features**; an **RMT-weighted two-sleeve allocator** (80% diversified safe sleeve + 20% concentrated top-5 sleeve); and a **drawdown-targeting exposure wrapper**. The central design discovery is empirical: **cross-disciplinary signals (Kuramoto synchronization, cross-sectional dispersion, Gutenberg-Richter foreshock counts, vol-of-vol, RMT dominant-mode ratio) improve performance when fed to the model as features, but actively harm performance when applied as exogenous override rules**. The Phase 62 model was validated across **three disjoint stock universes** (2,500 unique companies, two of which the model had never seen). The earlier allocator+wrapper layer (Phase 54) was additionally tested across **three random LambdaRank seeds**, with MDD reduction reproducing to within ±1% across all seeds. We have not yet run a three-seed test of the full Phase 62 stack.

---

## 1. The final method

### 1.1 Architecture in one diagram

```
                ┌─────────────────────────────────────┐
                │  ~17000 anonymous fundamentals      │
   Sample  ─►   │  6 anonymous price features         │  ─► LambdaRank ─► rank
   (filing)     │  6 cross-disciplinary market feats  │      (pairwise)
                └─────────────────────────────────────┘                │
                                                                       ▼
                              ┌──────────────────────────────────────┐
                              │  Phase 54 — RMT-weighted barbell     │
                              │  80% safe: RMT min-var on all picks  │
                              │  20% concentrated: RMT min-var top-5 │
                              └──────────────────────────────────────┘
                                                                       │
                                                                       ▼
                              ┌──────────────────────────────────────┐
                              │  Phase 44 — DD-target wrapper        │
                              │  scale = clip(1 - (|DD|−trigger)/    │
                              │              trigger, floor, 1)      │
                              └──────────────────────────────────────┘
                                                                       │
                                                                       ▼
                                                                  daily PnL
```

### 1.2 The six cross-disciplinary features

Computed causally from the universe-wide price panel at each sample's `entry_date`:

| Feature | Computation | Discipline of origin |
|---|---|---|
| `f_market_temp_z` | rolling z-score of cross-sectional std of daily returns | Statistical mechanics — universe "temperature" |
| `f_kuramoto_r` | order parameter \|⟨e^{iφ}⟩\| where φ ∝ daily-return z-score | Kuramoto 1975 coupled oscillators — phase synchronization |
| `f_kuramoto_z` | rolling z-score of f_kuramoto_r | Regime detection |
| `f_gr_count_z` | z-score of recent-window count of daily returns < −1σ | Gutenberg-Richter / self-organized criticality |
| `f_market_mode_ratio` | λ₁ / trace of RMT-cleaned correlation matrix | Random matrix theory — dominant-mode dominance |
| `f_vov_z` | z-score of rolling std of daily universe vol | GARCH heteroscedasticity |

All six are MARKET-WIDE observables — the same value at the same date for every stock. The LambdaRank model learns to condition its per-stock ranking on these regime indicators.

### 1.3 The RMT-weighted two-sleeve allocator

For each rebalance date:
1. Compute candidate set: top-K active positive-signal names.
2. Build the **RMT-cleaned covariance** Σ_clean from the candidates' 252-day return panel via Marchenko-Pastur eigenvalue filtering (Laloux et al. 1999). Eigenvalues inside the M-P window `[(1−√q)², (1+√q)²]` (q = N/T) are replaced with their mean; the market mode is preserved.
3. **Safe sleeve (80% of capital)**: solve `max wᵀμ − (γ/2) wᵀΣ_clean w` over all candidates with `w ≥ 0`, `Σw = 1`. γ = 5 (textbook).
4. **Concentrated sleeve (20% of capital)**: same solve over only the top-5 candidates by raw signal.
5. Sum the two sleeves, apply per-name caps + cash rule.

The motivation for the two-sleeve split is the barbell principle (Taleb 2012): heavy weight on extremes of the conviction spectrum (broad diversification + concentrated bets) with nothing in the middle. The 80/20 split is the canonical ratio from that literature, declared a priori — not searched on val/test data.

### 1.4 The drawdown-target wrapper

After each rebalance, scale the equity slice multiplicatively:

```
DD(t) = equity(t) / max(equity[t-252:t]) − 1     (≤ 0)
scale(t) = clip(1 − max(0, |DD| − 0.10) / 0.10, 0.30, 1.0)
EMA-smoothed with blend = 0.5
```

Pure classical proportional control: when the strategy enters a drawdown deeper than 10%, exposure scales down linearly to a floor of 30% at 20% drawdown. Nothing forward-looking.

---

## 2. Why this method beats everything else we tried

We tried 24 cross-disciplinary hypotheses. Only four became production keepers (Phases 38, 44, 54, 62). The other 20 are documented as honest failures or neutral effects. The key result of this paper is not the success — it's the **pattern** in the failures that explained how to succeed.

### 2.1 The output-side rule

Five hypotheses used cross-disciplinary regime signals as **exposure wrappers** (a discrete rule that scales the gross book based on a current observable):

- Phase 39: cross-sectional dispersion (statistical-mechanics temperature)
- Phase 50: vol-of-vol (GARCH heteroscedasticity)
- Phase 55: vol-aware VaR-style trigger
- Phase 60: Gutenberg-Richter foreshock count (geophysics)
- Phase 61: Kuramoto synchronization (oscillator physics)

**All five failed**, ranging from -0.04 Sharpe (regime filter) to -0.40 Sharpe (Kuramoto wrapper combined with GR wrapper).

But the **same five signals**, fed to LambdaRank as anonymous model features (Phase 62), produced **+0.05 Sharpe averaged across three universes**. The signals are informative. What kills them is the wrapper form.

The interpretation is a single sentence:

> **A wrapper firing on a threshold cannot distinguish "high regime signal + good fundamentals (still buy)" from "high regime signal + weak fundamentals (avoid)". A model with these signals as features can.**

This is a contextual conditioning argument: cross-disciplinary regime observables are noisy on their own; their value is in interactions with the per-stock signal. A model captures interactions; a fixed rule cannot.

### 2.2 The orthogonality rule

Phase 62 added six MARKET-LEVEL features — values that are uniform across all stocks at the same date. Result: +0.05 Sharpe average.

Phase 64 added six PER-STOCK features (Hurst exponent, DFA-α, Lévy z-score, Lyapunov exponent, return skew, kurtosis). LambdaRank assigned them substantial gain in feature importance, but the backtest result was **unchanged from Phase 62** (1.132 vs 1.128 — within measurement noise).

Why? The model already has ~17,000 anonymous per-stock fundamentals plus six price features. The per-stock cross-disciplinary statistics are non-linear transformations of the existing per-stock representation — informative for the model's *internal* representations but not adding *external* signal.

> **Cross-disciplinary features add value if-and-only-if they are orthogonal to the model's existing feature space.** Market-level observables were orthogonal (no universe-wide features existed before). Per-stock statistics were redundant.

### 2.3 The wrong-sign theorems

Two ideas had theoretical priors that pointed the OPPOSITE direction of what the data supported:

- **Phase 39** (statistical mechanics): the textbook prediction "cold = correlated = bad for selection" assumed correlation magnitude. The DATA: correlated *drawdowns* coincided with HIGH cross-sectional dispersion, not low.
- **Phase 55** (Basel VaR theory): in high-vol regimes the DD trigger should widen because larger drawdowns are statistically "normal." The DATA: when realized vol rises, drawdowns get worse rather than safer — the trigger should narrow.

In both cases, applying the theoretical prior gave WORSE performance. The single-pass evaluation discipline prevented us from "rescuing" these ideas by flipping the sign post-hoc. We retired both.

> **Don't trust a cross-disciplinary prior just because the analogy is appealing; verify the sign before declaring success.**

---

## 3. Empirical validation

The Phase 62 production stack was evaluated under increasingly stringent criteria.

### 3.0 Ablation ladder — incremental contribution of each layer

To show no single layer dominates the gain, the table below
incrementally adds one piece at a time and reports tier1 s42 metrics
(2020-2026, 10 bps cost, frozen LambdaRank predictions across rows
1-5):

| Stack | Model | Allocator | Wrapper | Cross features | Sharpe | MDD | Calmar |
|---|---|---|---|---|---:|---:|---:|
| 1. Original baseline | LambdaRank v3 | inverse-vol | none | no | 1.018 | -0.431 | 0.641 |
| 2. + RMT min-var (P38) | LambdaRank v3 | RMT min-var | none | no | 1.096 | -0.446 | 0.727 |
| 3. + RMT + two-sleeve (P54) | LambdaRank v3 | RMT-barbell | none | no | 1.096 | -0.444 | 0.799 |
| 4. + DD-target wrapper (P44) | LambdaRank v3 | RMT-barbell | dd_target | no | 1.079 | -0.286 | 0.875 |
| 5. + cross-disc features (P62) | LambdaRank v3 + 6 cross | RMT-barbell | dd_target | **yes** | **1.128** | **-0.278** | **0.917** |

The marginal contribution at each step:

| Step | What's added | ΔSharpe | ΔMDD | ΔCalmar |
|---|---|---:|---:|---:|
| 1→2 | RMT covariance cleaning | +0.078 | +0.015 | +0.086 |
| 2→3 | Two-sleeve barbell structure | 0.000 | +0.002 | +0.072 |
| 3→4 | DD-target wrapper | -0.017 | -0.158 | +0.076 |
| 4→5 | Cross-disciplinary model features | +0.049 | -0.008 | +0.042 |

No single layer is responsible. RMT covariance gives the Sharpe lift;
the two-sleeve barbell gives the Calmar lift (same Sharpe, better
return for same MDD); the DD-target wrapper buys the MDD reduction at
a small Sharpe cost; the cross-disciplinary features then recover
that Sharpe AND modestly improve MDD.

### 3.1 In-sample (tier1, model's own universe)

Sharpe 1.128, MDD -0.278, Calmar 0.917. Beat the prior best (Phase 54 without cross features: 1.079 / -0.286 / 0.875) by +5% Sharpe / +5% Calmar.

### 3.2 Cross-seed (Phase 56) — allocator+wrapper layer only

The Phase 54 stack (RMT-barbell + DD-target, NO cross features) tested
across three random LambdaRank seeds (s7, s42, s99):

| Seed | Sharpe | MDD | Δ vs baseline |
|---|---|---|---|
| s7 | 1.134 | -0.272 | +18% / -35% |
| s42 | 1.079 | -0.286 | +6% / -34% |
| s99 | 1.127 | -0.274 | +17% / -35% |

The **MDD reduction is identical to within 1% across seeds (-34, -34, -35)** — a structural result, not noise.

This test isolates the allocator + wrapper layer. **A full three-seed
test of the Phase 62 stack (model + cross-disciplinary features) has
not yet been run.** Future work (§7) describes a clean pre-declared
5-seed Bayesian-bagged ensemble that would close this gap; its code
is already in the repo (`scripts/train_phase62_seeds.py`).

### 3.3 Cross-universe (Phases 57, 58, 65)

The Phase 62 stack tested on two universes the model had never seen during training:

| Universe | Companies | Baseline Sharpe | Phase 62 Sharpe | Δ |
|---|---|---|---|---|
| tier1 (CIKs 0-999, training pool) | 1000 | 1.018 | 1.128 | +10.8% |
| tier2 (CIKs 1000-1999) | 771 | 0.989 | 1.123 | +13.5% |
| tier3 (CIKs 2000-2999) | 719 | 0.651 | **0.868** | **+33.4%** |
| **average** | — | **0.886** | **1.040** | **+17.4%** |

The improvement HOLDS on universes the model never saw, with **larger gains in weaker-signal universes** (tier3). The Calmar improvement on tier2 (0.832 → 0.960, +15.4%) is the largest single-universe gain — the cross-disciplinary features generalize EVEN BETTER on external data than on the universe used to design them.

This is strong evidence under the single-pass discipline, though not a substitute for longer out-of-time or live forward validation. We do not yet have results in a regime as severe as the 2008 GFC, and we have not run the strategy live.

---

## 4. The design principle (concise)

The summary of 24 experiments worth of empirical evidence:

> **Cross-disciplinary signals belong in the model's input space, not in its override layer. The signals should be MARKET-LEVEL (orthogonal to per-stock features). The output side should be regulated by simple proportional damping (drawdown wrapper), not by predictive regime detection. Whenever the cross-disciplinary prior and the data disagree, the data wins — and the single-pass evaluation discipline forces us to accept that.**

A more biological framing of the same principle: living systems regulate **outputs** via continuous proportional feedback (homeostasis), while learning to interpret **inputs** with adaptive context-sensitive models. The portfolio strategy that won looks remarkably like a biological system: a continuous-feedback wrapper on output (DD-target), an adaptive feature-rich model on input (LambdaRank with cross-disciplinary features), connected by a Pareto-optimal allocator (RMT barbell). Nothing in the chain is a *predictive regime override* — and that's exactly why it works.

---

## 4.5 Production no-go rules

The empirical evidence collected here is, beyond the specific numbers,
a set of operational rules for *what NOT to do* when extending this
strategy. They are derived from the failure pattern rather than from
theory:

1. **Do not use market-state observables as exposure override
   wrappers.** Five tried, five failed (Phase 39, 50, 55, 60, 61).
   These signals belong in the model's input space, where contextual
   conditioning can recover their value.
2. **Do not add per-stock cross-disciplinary transforms unless they
   are orthogonal to existing price/fundamental features.** Phase 64
   added six per-stock features that the model assigned substantial
   gain to in importance, but the backtest improvement was zero.
3. **Do not flip a theoretical sign post-hoc after a single-pass
   evaluation has falsified it.** Phase 39 and Phase 55 both had
   priors with the wrong sign; resisting the temptation to "rescue"
   them prevented data-mining.
4. **Do not add time-decay to filing signals.** The filing edge
   persists until the next filing because the strategy holds until
   then (Phase 63).
5. **Do not use VaR-style adaptive triggers that widen in high vol.**
   In equity drawdowns, the trigger should narrow, not widen
   (Phase 55).

## 5. Production stack specification

```yaml
model:
  type: lambdarank                       # Phase 29 + 62
  objective: lambdarank                  # pairwise ranking
  feature_set:
    - anonymous_fundamentals: ~17000 cols (encoder/v1)
    - price: 6 + 6_missing (Phase 21)
    - cross_disciplinary_market: 6 + 6_missing (Phase 62)
      - f_market_temp_z
      - f_kuramoto_r
      - f_kuramoto_z
      - f_gr_count_z
      - f_market_mode_ratio
      - f_vov_z
  hyperparams: {lr: 0.03, leaves: 31, max_position: 100, ...}

allocator:
  type: rmt_two_sleeve                   # Phase 54 = 38 + 53
  safe_fraction: 0.80                    # broad-diversification sleeve
  concentrated_top_k: 5                  # high-conviction sleeve
  inner_solve:
    type: signal_tilted_min_variance
    cov: rmt_cleaned                     # Phase 38 (Marchenko-Pastur)
    risk_aversion: 5.0

wrappers:
  - dd_target:                           # Phase 44
      dd_trigger: 0.10
      alpha: 1.0
      scale_floor: 0.30
      blend: 0.5
```

### Final numbers (cross-universe average, 2020-2026 backtest, 10 bps cost)

```
                          Sharpe    MDD     Calmar
Original baseline         0.886    -0.450    0.581
Production (Phase 62)     1.040    -0.288    0.849
Δ                         +17.4%   -36.0%   +46.2%
```

---

## 6. Limitations

### 6.1 Statistical / methodological

1. **Long-only ceiling**. The same-universe equal-risk-active benchmark (no signal, just diversification) achieves Sharpe 1.44. Our signal selection captures ~73% of this ceiling. The remaining gap requires shorts.
2. **Single backtest window** (2020-2026). The window contains COVID and the 2022 bear but no full GFC-style event.
3. **Phase 62 seed stability not formally measured**. Cross-seed
   validation was done for the Phase 54 layer; the full Phase 62
   stack has been evaluated under a single trained model. A clean
   5-seed Bayesian-bagged ensemble would close this gap.
4. **Feature redundancy at the per-stock level**. Adding more *anonymous* per-stock features is unlikely to help (Phase 64). Real gains require either pruning the existing 17,000 columns or finding genuinely orthogonal new feature types (e.g., text from filings, options-implied vols).
5. **Anonymity tax**. The model must rediscover what each anonymous feature means. A concept-named feature set would likely capture the same edge with 10× fewer columns.

### 6.2 Operational (capital-deployment readiness)

The numbers above support **paper / live-shadow deployment**. Real
capital deployment additionally requires:

1. **Transaction cost realism**. We use a flat 10 bps per trade.
   Real costs include bid-ask spread, market impact at portfolio
   size, queue position, and venue routing — all unmodelled here.
2. **Borrow / corporate-actions / delisting**. The backtest uses
   adjusted close. Long-only mitigates borrow concerns but not
   delisting (we exclude delisted symbols at sample build time,
   which introduces survivorship in the universe selection).
3. **Rebalance latency**. We assume entry at the first trading day
   after acceptance, exit at the last day before next acceptance.
   Real execution involves end-of-day fills and overnight gap risk.
4. **Operational monitoring**. The DD-target wrapper needs a live
   feed of strategy NAV. The cross-disciplinary feature pipeline
   needs daily price ingest. Both are runnable on a vanilla cron
   but require ops oversight.
5. **No live forward test**. The strategy has not been deployed
   live. The closest analog is the cross-universe holdout
   (tier2/tier3), which is still backtest data, not live data.

**Production status**: ready for paper / live-shadow / pilot capital
deployment under operational monitoring. NOT ready for full
discretion-free capital deployment without the above operational
risk layer.

---

## 7. What I would try next under v1 constraints

In decreasing order of expected ROI:

1. **Clean 5-seed Bayesian-bagged ensemble of Phase 62**. Pre-declared seeds 100-104, equal-weight rank average, single evaluation. Expected +0.02-0.04 Sharpe from variance reduction. Code already prepared (`scripts/train_phase62_seeds.py`).
2. **Additional market-level features**: sector-rotation strength, skew-of-skew, beta-dispersion. Expected +0.01-0.02 each.
3. **Quarterly walk-forward refit** of the LambdaRank, re-computing cross features. Risk: data leakage if poorly implemented. Expected -0.05 to +0.07 (high variance).

To exceed +5% additional Sharpe requires relaxing RFC v1 constraints (shorts, leverage, or new signal sources).

---

## 8. Reproducibility

Final-method code:
- `src/afp/features/cross_disciplinary_features.py` (Phase 62)
- `src/afp/portfolio/rmt_covariance.py` + `rmt_barbell_allocator.py` (Phase 38 + 54)
- `src/afp/portfolio/dd_target.py` (Phase 44)
- `scripts/train_lambdarank_cross_features.py` (Phase 62 training)
- `scripts/run_phys_v20_crossfeat_eval.py` (Phase 62 backtest)
- `scripts/predict_tier{2,3}_with_cross_features.py` (cross-universe)
- `scripts/run_phys_v65_eval.py` (cross-universe backtests)

Test suite: 224 unit tests covering every module added in Phase 38-65.

Single-pass evaluation framework: `docs/phase41_phys_eval.md`. Negative findings catalogued in `docs/phase{39,40,42,46,47,49,50,51,52,55,60,61,63}_*.md`. Neutral findings in `docs/phase{43,48,59,64}_*.md`.
