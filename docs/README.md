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
