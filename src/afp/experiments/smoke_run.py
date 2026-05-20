"""End-to-end smoke run: synthetic data → predictions → backtest → registered experiment.

This is the canonical "everything is wired" sanity check. Invoked by
`python -m afp.experiments.smoke_run` or by `tests/test_smoke_pipeline.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from afp.backtest.engine import BacktestConfig, run_backtest
from afp.backtest.report import build_report, write_report
from afp.experiments.registry import (
    ExperimentRun,
    append_registry,
    build_experiment_metadata,
)
from afp.experiments.validation import (
    gate_anonymous_columns,
    gate_backtest_accounting,
    gate_event_date_ordering,
    gate_predictions_in_range,
    gate_target_range,
    leakage_checklist,
)
from afp.features.encoder import AnonymousFeatureEncoder
from afp.features.sample_builder import build_event_samples
from afp.models.train import attach_prediction_context, train_and_predict
from afp.portfolio.allocation import PortfolioConfig
from afp.portfolio.risk_models import RiskConfig
from afp.targets.target_transform import TargetConfig, apply as apply_target
from afp.targets.volatility_scale import ScaleConfig, compute_ex_ante_scale_for_samples
from afp.utils.config import load_config


def run() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))   # so tests/fixtures.py is importable
    from tests.fixtures import make_synthetic_world

    cfg = load_config(ROOT / "configs" / "default.yaml")
    world = make_synthetic_world(2012, 2017)

    samples, _ = build_event_samples(
        world["filings"], world["prices"], world["calendar"], world["benchmarks"],
        train_end_date="2014-12-31", validation_end_date="2015-12-31",
    )
    gate_event_date_ordering(samples)

    scaled = compute_ex_ante_scale_for_samples(samples, world["prices"], world["calendar"],
                                               ScaleConfig())
    samples = apply_target(scaled, TargetConfig(k=cfg["target"]["k"]))
    gate_target_range(samples)

    enc = AnonymousFeatureEncoder(periods_back=4, min_company_count=2, min_sample_coverage_pct=0.1)
    train_samples = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"])
    enc.fit(train_samples, world["facts"])

    res = train_and_predict(
        "ridge", enc, train_samples,
        samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"]),
        samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"]),
        world["facts"],
    )
    gate_predictions_in_range(res.predictions)
    # Confirm the dense column space is anonymous
    scaled_arr, missing_arr, meta_arr, sids = enc.transform(train_samples, world["facts"])
    dense = enc.to_dense_frame(scaled_arr, missing_arr, meta_arr, sids)
    gate_anonymous_columns(dense.columns.tolist())

    preds = attach_prediction_context(res.predictions, samples)
    preds = preds[preds["split"].isin(("validation", "test"))].dropna(subset=["entry_date", "exit_date"])

    bt_cfg = BacktestConfig(start_date="2015-01-05", end_date="2018-06-29",
                            transaction_cost_bps_per_trade=10.0)
    pcfg = PortfolioConfig(min_positive_signal=0.0, max_single_stock_weight=0.5,
                           min_positions=1, target_positions=4, max_positions=4,
                           min_position_weight=0.0)
    result = run_backtest(preds, world["prices"], world["calendar"], bt_cfg, pcfg,
                          RiskConfig(lookback_days=120))
    gate_backtest_accounting(result.daily)
    report = build_report(result, world["prices"], world["benchmarks"])

    exp_id, meta = build_experiment_metadata(
        cfg,
        owner="smoke", purpose="smoke pipeline run",
        model_type="ridge", feature_version="v1",
        target_version="v1", portfolio_method="inverse_vol",
        repo_root=ROOT,
    )
    exp_dir = ROOT / "artifacts" / "experiments"
    run_dir = ExperimentRun(exp_dir, exp_id)
    run_dir.save_resolved_config(cfg)
    run_dir.save_metadata(**meta, validation_metrics=res.validation_metrics)
    run_dir.save_metrics({"backtest_report": report})
    run_dir.save_table(result.daily, "backtest_daily")
    run_dir.save_table(result.trades, "backtest_trades")
    run_dir.save_table(res.predictions, "predictions")
    checklist = leakage_checklist({
        "Did the model see any future financial filing?": "no",
        "Did the model see any future price?": "no",
        "Was target scale computed using future returns?": "no",
        "Were scaler statistics fit only on training data?": "yes",
        "Were hyperparameters selected on test data?": "no",
        "Was universe membership point-in-time or disclosed as imperfect?": "disclosed",
        "Were restatements controlled or disclosed?": "disclosed",
    })
    run_dir.save_report(
        f"# Experiment {exp_id}\n\n"
        f"Config hash: `{meta['config_hash']}`\n\n"
        f"## Validation metrics\n```\n{res.validation_metrics}\n```\n\n"
        f"## Backtest summary\n```\n{report['strategy']}\n```\n\n"
        f"{checklist}\n"
    )
    append_registry(exp_dir / "registry.csv", {
        **{k: v for k, v in meta.items() if not isinstance(v, dict)},
        "strategy_sharpe": report["strategy"]["sharpe"],
        "strategy_max_drawdown": report["strategy"]["max_drawdown"],
    })

    return {"experiment_id": exp_id, "metrics": res.validation_metrics,
            "backtest": report["strategy"]}


if __name__ == "__main__":
    out = run()
    print(out)
