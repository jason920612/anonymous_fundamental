# Technical Docs

Per-phase technical documentation for the anonymous fundamental portfolio
research system. Each file corresponds to one RFC and one implementation
milestone.

| Phase | RFC | Doc | Source Package |
|---|---|---|---|
| 1 | RFC-00, §8 (skeleton) | [phase1_repo_skeleton.md](phase1_repo_skeleton.md) | `afp.utils` + configs |
| 2 | RFC-01 | [phase2_data_ingestion.md](phase2_data_ingestion.md) | `afp.data.*` |
| 3 | RFC-02 | [phase3_event_dataset.md](phase3_event_dataset.md) | `afp.features.sample_builder` |
| 4 | RFC-04 | [phase4_target_normalization.md](phase4_target_normalization.md) | `afp.targets.*` |
| 5 | RFC-03 | [phase5_anonymous_features.md](phase5_anonymous_features.md) | `afp.features.*` |
| 6 | RFC-05 | [phase6_modeling.md](phase6_modeling.md) | `afp.models.*` |
| 7 | RFC-06 | [phase7_portfolio.md](phase7_portfolio.md) | `afp.portfolio.*` |
| 8 | RFC-07 | [phase8_backtest.md](phase8_backtest.md) | `afp.backtest.*` |
| 9 | RFC-08 | [phase9_experiments.md](phase9_experiments.md) | `afp.experiments.*` |
| 10 | RFC-09 §13-§17 | [phase10_deployment.md](phase10_deployment.md) | `afp.cli.*`, `afp.data.price_client_yfinance`, Dockerfile, CI |
| 11 | RFC-07 §17, RFC-09 §16 | [phase11_validation.md](phase11_validation.md) | resumable ingest, stronger baselines, signal diagnostics, attribution |
| 12 | RFC-01 §6 | [phase12_universe_filter.md](phase12_universe_filter.md) | point-in-time liquidity / price / history filter at sample-build |
| 13 | RFC-02 §10 | [phase13_cross_sectional_target.md](phase13_cross_sectional_target.md) | cross-sectional rank target (2*pct_rank - 1) |
| 14 | RFC-03 §4 | [phase14_derived_features.md](phase14_derived_features.md) | anonymous YoY + z-score derived features |
| 15 | RFC-05 | [phase15_sample_weighting.md](phase15_sample_weighting.md) | inverse-frequency sample weights |
| 16 | RFC-06 §12 | [phase16_sector_caps.md](phase16_sector_caps.md) | SIC sector parsing + sector cap allocator |
| 17 | RFC-09 §16 | [phase17_improvement_results.md](phase17_improvement_results.md) | 10-variant ablation; **Ridge solo wins at 0.76 Sharpe** |
| 18 | RFC-05 §17 | [phase18_hpo.md](phase18_hpo.md) | time-based hyperparameter search (test-data safe) |
| 19 | RFC-05 §5 | [phase19_gpu.md](phase19_gpu.md) | XGBoost CUDA + PyTorch MLP on GPU |
| 20 | RFC-05 §10 | [phase20_ensemble.md](phase20_ensemble.md) | mean + rank-average ensembling |
| 21 | RFC-00 §11 (lifted) | [phase21_price_features.md](phase21_price_features.md) | anonymous price-derived features (trailing returns + vol) |
| 22 | RFC-05 §7 | [phase22_diffusion.md](phase22_diffusion.md) | conditional DDPM on anonymous features (GPU) |
| 23 | RFC-05 §7 | [phase23_diffusion_v2.md](phase23_diffusion_v2.md) | bigger diffusion + price condition (GPU) |
| 24 | RFC-06 §9 | [phase24_distribution_allocator.md](phase24_distribution_allocator.md) | confidence/std-weighted long-only allocator |
| 25 | RFC-05 §10 | [phase25_selftraining.md](phase25_selftraining.md) | reward-weighted self-training loop |
| 26 | RFC-07 §17 | [phase26_final_leaderboard.md](phase26_final_leaderboard.md) | **final leaderboard + lessons (Sharpe 0.79)** |
| 29 | RFC-09 §17 | [phase29_lambdarank_breakthrough.md](phase29_lambdarank_breakthrough.md) | LambdaRank pairwise loss → Sharpe 0.96 breakthrough |
| 33 | methodology | [phase33_methodology_correction.md](phase33_methodology_correction.md) | corrected test-snooping; honest val-driven Sharpe 1.00 |
| 34 | RFC-08 §11 | [phase34_external_universe_test.md](phase34_external_universe_test.md) | tier2 cross-universe holdout (1000 new CIKs) — Sharpe 0.99-1.03 |
| 35 | UX | [phase35_predict_tool.md](phase35_predict_tool.md) | `afp predict TICKER` — user-facing prediction CLI |
| 36 | UX | [phase36_cache_refresh.md](phase36_cache_refresh.md) | auto cache freshness check (SEC + yfinance) |
| 37 | UX | [phase37_reference_cohort_refresh.md](phase37_reference_cohort_refresh.md) | auto reference-cohort refresh + staleness warnings |
| 38 | RFC-06 §9 (lifted) | [phase38_rmt_covariance.md](phase38_rmt_covariance.md) | Marchenko-Pastur covariance cleaning + signal-tilted min-variance |
| 39 | RFC-06 §9 (lifted) | [phase39_regime_filter.md](phase39_regime_filter.md) | thermodynamic regime detector (cross-sectional dispersion → Boltzmann scale) |
| 40 | RFC-06 §9 (lifted) | [phase40_max_entropy_allocator.md](phase40_max_entropy_allocator.md) | Jaynes max-entropy / Boltzmann allocator |
| 41 | methodology | [phase41_phys_eval.md](phase41_phys_eval.md) | single-pass evaluation: physics-inspired allocators |
| 42 | (candidate) | [phase42_hrp_candidate.md](phase42_hrp_candidate.md) | Hierarchical Risk Parity — pending Phase 41 readout |
| 43 | (candidate) | [phase43_tail_aware_candidate.md](phase43_tail_aware_candidate.md) | Tail-aware (Expected Shortfall) allocator — pending Phase 41 readout |
| 44 | RFC-06 §9 (lifted) | [phase44_drawdown_target.md](phase44_drawdown_target.md) | **drawdown-targeting wrapper — KEEPER (Sharpe 1.08, MDD -0.29, Calmar 0.80)** |
| 45 | meta | [phase45_phys_summary.md](phase45_phys_summary.md) | **physics-inspired research summary: 6 ideas tried, 2 keepers, +25% Calmar** |
| 46 | (retired) | [phase46_beta_target_retired.md](phase46_beta_target_retired.md) | beta-targeting wrapper — dominated by Phase 44 dd_target, no marginal value |
| 47 | (retired) | [phase47_eigenportfolio_retired.md](phase47_eigenportfolio_retired.md) | eigenportfolio market-mode subtraction — long-only constraint kills the projection |
| 48 | (neutral) | [phase48_js_shrinkage_neutral.md](phase48_js_shrinkage_neutral.md) | James-Stein signal shrinkage — neutral (no-op on this signal distribution) |
| 49 | (retired) | [phase49_adaptive_gamma_retired.md](phase49_adaptive_gamma_retired.md) | adaptive γ via Shannon entropy — theory mapped the wrong direction, retired |
| 50 | (retired) | [phase50_vov_retired.md](phase50_vov_retired.md) | vol-of-vol target — too noisy, over-damps. arc saturated at Phase 44 |
| 51 | (retired) | [phase51_ensemble_retired.md](phase51_ensemble_retired.md) | allocator ensemble (inv-vol + RMT) — dilutes RMT edge, retired |
| 52 | (retired) | [phase52_ensemble3_retired.md](phase52_ensemble3_retired.md) | 3-model rank-avg ensemble — lowest MDD but bad Sharpe, retired |
| 53 | (keeper, ablation) | [phase53_barbell_keeper.md](phase53_barbell_keeper.md) | antifragile barbell — Calmar 0.851. Superseded by Phase 54 |
| 54 | **(WINNER)** | [phase54_rmt_barbell_winner.md](phase54_rmt_barbell_winner.md) | **RMT-weighted barbell + DD-target — Sharpe 1.08, MDD -0.29, Calmar 0.875 (Pareto-dominant)** |
| 55 | (retired) | [phase55_adaptive_trigger_retired.md](phase55_adaptive_trigger_retired.md) | vol-aware adaptive DD trigger — VaR-style framing sign-flips, static 10% wins |
| 56 | (validation) | [phase56_seed_robustness.md](phase56_seed_robustness.md) | **cross-seed (s7/s42/s99) robustness — Phase 54 win confirmed: +13.7% Sharpe / -35% MDD / +44% Calmar avg** |
| 57+58 | (validation) | [phase57_58_cross_universe.md](phase57_58_cross_universe.md) | **triple cross-universe (tier2 + tier3) — Phase 54 wins on ALL 5 tests (3 seeds × tier1 + tier2 + tier3); avg MDD -35% ± 1%** |
| 59 | (neutral) | [phase59_mm_saturation_neutral.md](phase59_mm_saturation_neutral.md) | Michaelis-Menten enzyme kinetics (1913 biochemistry) signal saturation — neutral, helps tier3 (low-signal), neutral elsewhere |
| 60+61 | (retired) | [phase60_61_wrappers_retired.md](phase60_61_wrappers_retired.md) | Gutenberg-Richter foreshock + Kuramoto sync as WRAPPERS — failed catastrophically (lesson: feed to model, not wrapper) |
| 62 | **(NEW WINNER)** | [phase62_cross_features_keeper.md](phase62_cross_features_keeper.md) | **LambdaRank trained WITH cross-disciplinary features — Sharpe 1.128, MDD -0.278, Calmar 0.917 (beats Phase 54)** |
| 63 | (retired) | [phase63_time_decay_retired.md](phase63_time_decay_retired.md) | first-principles time-decay signal — empirically wrong, decays the very edge we're betting on |
| 64 | (neutral) | [phase64_perstock_neutral.md](phase64_perstock_neutral.md) | per-stock cross-disciplinary features (Hurst, DFA, Lévy, Lyapunov) — redundant with existing fundamentals |
| 65 | (validation) | [phase65_cross_universe_validation.md](phase65_cross_universe_validation.md) | **Phase 62 cross-universe validation: tier1/2/3 all confirm. Avg Sharpe 1.04, MDD -0.29, Calmar 0.85 — +17%/-36%/+46% vs baseline** |
| 67 | UX | [phase67_refresh_all.md](phase67_refresh_all.md) | **`afp refresh-all` — one-command pipeline (ingest + features + train + artifacts), predict CLI auto-picks Phase 62 model** |
| 68 | UX | [phase68_daemon_telegram.md](phase68_daemon_telegram.md) | **`afp daemon` — 24h S&P 500 watcher + Telegram bot notifications; Phase 62 is now the default model** |

## Running the whole system

```bash
pip install -e ".[prices,dev]"
pytest -q                              # 127 tests across phases 1..25
python -m afp.experiments.smoke_run    # synthetic-data end-to-end pipeline
afp run-pipeline --skip-ingest         # full CLI pipeline on already-ingested data
afp diagnostics --predictions artifacts/predictions/diffusion_v2.parquet
```

For the best Phase-26 production pipeline:

```bash
python3 scripts/run_v1000_diffusion_v2.py 100       # GPU, ~20 min
afp backtest --config configs/deployment_v1000.yaml \
             --predictions artifacts/predictions/diffusion_v2.parquet \
             --portfolio-id diffusion_v2_dist \
             --allocator distribution \
             --min-probability-positive 0.60
```

The smoke run produces an experiment directory under
`artifacts/experiments/<experiment_id>/` and appends a row to
`artifacts/experiments/registry.csv`.

## Status — Final Result (1000-CIK universe, 2020-01 → 2026-05)

Joint best (Sharpe 0.79): **diffusion v2 + distribution allocator**
and **Ridge + 6 anonymous price features**. Diffusion variant has lower
MDD (-35.8% vs -37.1%). Both beat SPY (0.77) but trail same-universe
equal-risk-active baseline (1.44).

15 variants explored across phases 12-25 (filter, rank target, derived
features, sample weighting, sector caps, HPO, XGB-GPU, MLP-GPU, ensembles,
price features, diffusion v1/v2/v3, distribution allocator, self-training).
See [phase26_final_leaderboard.md](phase26_final_leaderboard.md) for the
complete table and the structural ceiling analysis.
