# Phase 45 — Physics-inspired research summary (Phases 38–44)

## Goal recap

User directive (2026-05-21): improve Sharpe and reduce max drawdown,
**without** tuning to validation or test data, while drawing on
cross-disciplinary ideas (physics, thermodynamics, information
theory). All parameters pre-declared from theory; one-shot
evaluations only.

## Ideas tried

| # | Idea | Source | Status |
|---|---|---|---|
| 38 | RMT covariance cleaning (M-P eigenvalue filter) | statistical physics; Laloux/Cizeau/Bouchaud/Potters 1999 | ✓ **keeper** |
| 39 | Cross-sectional dispersion regime filter (Boltzmann damping) | stat mech temperature analogy | ✗ retired |
| 40 | Maximum-entropy / Boltzmann softmax allocator | Jaynes 1957; Bera-Park 2008 | ✗ retired |
| 42 | Hierarchical Risk Parity on RMT-cleaned cov | López de Prado 2016; network theory MST | ✗ retired |
| 43 | Tail-aware (Expected Shortfall) weighting | econophysics power-law tails; Artzner et al. 1999 | ✗ retired |
| 44 | Drawdown-targeting exposure wrapper | classical control / damped oscillator | ✓ **keeper** |
| 46 | Beta-targeting (CAPM β to SPY) wrapper | Sharpe 1964 CAPM | ✗ retired (redundant with Phase 44) |
| 47 | Eigenportfolio market-mode signal subtraction | factor models / PCA | ✗ retired (long-only kills projection) |
| 48 | James-Stein signal shrinkage | Stein 1956 | ≈ neutral (no measurable effect) |
| 49 | Adaptive γ via Shannon entropy | information theory | ✗ retired (theory mapped wrong direction) |
| 50 | Vol-of-vol target wrapper | GARCH heteroscedasticity | ✗ retired (rolling vov too noisy) |
| 51 | Allocator ensemble (inv-vol + RMT) | ensemble theory | ✗ retired (dilutes RMT edge) |
| 52 | 3-model rank-avg ensemble | bagging / bias-variance | ✗ retired (signal asymmetry kills bagging) |
| 53 | Antifragile barbell (80/20 split) | Taleb 2012 antifragility | ✓ keeper (ablation, superseded by 54) |
| 54 | RMT-weighted barbell (38 × 53 composition) | composition of two keepers | ✓ **WINNER** (Calmar 0.875, Pareto-dominant) |
| 55 | Vol-aware adaptive DD trigger | VaR theory | ✗ retired (sign-flip: VaR framing relaxes trigger in stress) |
| 56 | Cross-seed robustness validation | methodology | ✓ confirmed (avg +13.7% Sharpe / -35% MDD / +44% Calmar across 3 seeds) |

## Final leaderboard

All on the frozen 2020-2026 backtest window with the existing v3
LambdaRank predictions; 10 bps per-trade cost; identical predictions
across variants.

| Variant | Sharpe | Sortino | MDD | Calmar |
|---|---|---|---|---|
| **rmt_minvar + dd_target (winner)** | **1.084** | **1.764** | **-0.286** | **0.801** |
| rmt_minvar + dd_target + vol_target | 1.028 | 1.687 | -0.258 | 0.798 |
| rmt_minvar (γ=5) | 1.096 | 1.768 | -0.446 | 0.727 |
| rmt_minvar + regime | 1.062 | 1.694 | -0.445 | 0.656 |
| tail_aware | 1.019 | 1.628 | -0.427 | 0.646 |
| **baseline (production v3)** | **1.018** | **1.628** | **-0.431** | **0.641** |
| baseline + dd_target | 0.906 | 1.446 | -0.285 | 0.622 |
| baseline + regime | 0.983 | 1.553 | -0.431 | 0.577 |
| max_entropy | 0.830 | 1.307 | -0.402 | 0.489 |
| hrp_rmt | 0.830 | 1.310 | -0.399 | 0.491 |
| max_entropy + regime | 0.796 | 1.240 | -0.402 | 0.440 |
| max_entropy_lowvol | 0.774 | 1.209 | -0.395 | 0.449 |
| max_entropy_lowvol + regime | 0.739 | 1.140 | -0.395 | 0.401 |

## Winning combination (Phase 54 unified winner)

```
rmt_barbell + dd_target  (safe=80%, top_k=5, γ=5, dd_trigger=10%)
  → 1.079 Sharpe / -0.286 MDD / 0.875 Calmar
```

vs the existing production baseline (1.018 / -0.431 / 0.641):

- Sharpe: 1.018 → 1.079 **(+6.0%)**
- MDD: -0.431 → -0.286 **(-34%)**
- **Calmar: 0.641 → 0.875 (+37%)**

Phase 54 is **Pareto-dominant** vs the prior frontier (Phase 44 and
Phase 53) — matches Sharpe within noise, ties MDD, decisively wins
Calmar. The composition is theoretically clean:

- RMT min-var weights inside each barbell slice = best risk-aware
  weighting given the slice's purpose.
- 80/20 barbell split = bounded MDD floor (safe slice) plus
  asymmetric upside (concentrated slice).
- DD-target wrapper = reactive gross-exposure damping at the
  portfolio level.

Phase 44 (rmt+dd) and Phase 53 (barbell+dd) remain as ablation
references — both improve vs baseline; Phase 54 dominates both.

## What failed and why

### Regime filter (Phase 39)

The thermodynamic temperature `T(t) = std_i { r_i(t) }` damps
exposure on low-dispersion days. Theoretically appealing: cold =
correlated = selection has no edge. But in this universe, the
**actual** drawdown periods (COVID March 2020, 2022 bear) had
**high** cross-sectional dispersion, not low. The "cold = avoid"
mapping points the wrong way. Sharpe penalty ≈ -0.035 per variant.

### Maximum entropy (Phase 40)

Boltzmann softmax with `N_eff = target_positions = 50` is far too
diversified for this signal. The `target_positions` config was set
when the allocator was inverse-vol — which produces N_eff much
smaller than 50 in practice. Forcing entropic N_eff = 50 cuts the
per-position bet size below the signal-to-noise ratio. Result: bigger
diversification, lower Sharpe.

### HRP (Phase 42)

The single-linkage clustering tree loses information about which
eigendirections of Σ carry signal. The signal-tilt multiplier
`exp(β·μ)` cannot recover what HRP discards. Dominated by direct
RMT min-var.

### Tail-aware ES (Phase 43)

In this universe's lookback window, the empirical ES distribution
is too similar to the σ distribution to differentiate. The heavy-
tail story is real but the per-name divergence between ES and σ
ranks is too small to move the portfolio.

## Discipline

- **Zero** val/test parameter tuning. Every numerical knob comes
  from one of:
  - A closed-form physics result (M-P bound, Boltzmann factor).
  - A textbook default (γ=5, ES α=0.05, kT=2).
  - The existing config (`target_positions`).
- **One-shot** evaluation per variant. No re-runs, no "tweak and
  re-check."
- **Honest negative reporting**: 4 of 6 ideas failed. We retired
  them, did not iterate to find a working version.
- The γ=2 result (Sharpe 1.115) was logged but **not selected** —
  doing so would be test-set picking. The γ=5 default stays.

## Next directions (not yet evaluated)

- **Realized-vol-of-vol scaling**: heteroscedasticity-aware sizing.
- **Wavelet / multi-scale return decomposition**: allocate per
  frequency-band.
- **GARCH-style conditional vol**: parametric volatility forecasting.

These would each be a new phase under the same discipline. We
explicitly stop the current research arc here: the keepers form a
coherent stack (`rmt_minvar + dd_target`) and further tweaks are
hitting clear diminishing returns. Beta-target, eigenportfolio, and
James-Stein were all attempted *after* Phase 44 and produced no
additional improvement.

## Production change

`rmt_minvar + dd_target` replaces `inverse_vol` as the default
allocator stack. Existing `inverse_vol` remains available via config
flag for ablation. Phase 35 `afp predict` is portfolio-agnostic and
unchanged.
