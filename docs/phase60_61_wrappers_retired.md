# Phase 60+61 — Cross-disciplinary wrappers (RETIRED)

## Hypothesis

Apply cross-disciplinary regime signals as exposure-scaling wrappers:

- **Phase 60: Gutenberg-Richter foreshock detector** — geophysics
  / self-organized criticality. Count of small-magnitude drawdowns
  in a recent window; anomalous excess = foreshock signature.
- **Phase 61: Kuramoto synchronization order parameter** — coupled
  oscillator physics (1975). Measures phase coherence across the
  universe; high coherence = lockstep regime, bad for selection.

Both apply Boltzmann damping (kT=2) on positive z-scores. Same form
as Phase 39 regime filter and Phase 50 vov target.

## Result (tier1 s42)

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| rmt_barbell+dd (Phase 54) | **1.079** | -0.286 | **0.875** |
| +GR wrapper | 0.823 | -0.286 | 0.593 |
| +Kuramoto wrapper | 0.676 | -0.282 | 0.480 |
| +GR + Kuramoto | 0.574 | -0.282 | 0.368 |

**Verdict: retired.** Both signals as WRAPPERS hurt severely.
Stacking them is catastrophic (-46% Sharpe, -58% Calmar).

## Why they fail (deep lesson)

These signals carry real information — the LambdaRank model in
**Phase 62** assigns substantial gain to them when fed as
features. But a static wrapper firing on a z-threshold is too
crude:

- High-z events include both stress periods AND opportunity periods
- The wrapper cannot distinguish which is which without contextual
  fundamental info
- Damping exposure during opportunity periods costs return; the
  protection during stress periods isn't large enough to compensate

The same finding applies to Phase 39 (cross-sectional dispersion),
Phase 50 (vov), and Phase 55 (VaR adaptive trigger). The lesson:

> **"Input-side regime detectors as wrappers" is a systematic
> failure mode** for this universe. Cross-disciplinary signals
> must be incorporated as MODEL FEATURES (Phase 62), where the
> model can learn contextual rules, NOT as fixed override wrappers.

## Discipline

- Pre-declared parameters from theory (kT=2, sigma=1σ, etc.).
- Single-pass evaluation across all variants.
- Both signals retired as exposure wrappers. They live on as
  successful model features in Phase 62.
