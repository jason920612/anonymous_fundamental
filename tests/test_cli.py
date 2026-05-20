"""Phase 10 tests — CLI dispatch + walk-forward + end-to-end CLI pipeline.

The synthetic world is materialized to disk to mimic the production layout, then
the build-dataset → train → backtest CLI chain is exercised via `afp.cli.main`.
SEC + price ingestion subcommands are tested only at the parser level (they
require network/yfinance) — their internals are covered elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from afp.cli.main import build_parser, main
from tests.fixtures import make_synthetic_world


def test_parser_recognizes_all_subcommands():
    parser = build_parser()
    args = parser.parse_args(["ingest-sec", "--limit-ciks", "3"])
    assert args.command == "ingest-sec" and args.limit_ciks == 3
    for cmd in ("ingest-prices", "build-dataset", "train", "backtest",
                "run-pipeline", "walk-forward"):
        parsed = parser.parse_args([cmd] + (["--predictions", "p.parquet"] if cmd == "backtest" else []))
        assert parsed.command == cmd


@pytest.fixture
def materialized_world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    world = make_synthetic_world(2012, 2017)
    proc = Path("data/processed")
    proc.mkdir(parents=True, exist_ok=True)
    world["filings"].to_parquet(proc / "filings.parquet", index=False)
    world["facts"].to_parquet(proc / "financial_facts_long.parquet", index=False)
    world["prices"].to_parquet(proc / "prices_daily.parquet", index=False)
    world["benchmarks"].to_parquet(proc / "benchmarks_daily.parquet", index=False)
    cal_df = pd.DataFrame({"date": [d.date() for d in world["calendar"].trading_dates],
                           "is_trading_day": True})
    cal_df.to_parquet(proc / "trading_calendar.parquet", index=False)
    # Minimal config writes (smaller features so fit is cheap)
    cfg_dir = Path("configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "default.yaml").write_text(_make_test_config())
    return tmp_path


def _make_test_config() -> str:
    return """
project: {random_seed: 42}
paths: {data_processed: data/processed, artifacts: artifacts, reports: reports}
data:
  sec: {user_agent: "test"}
  prices: {source: csv, csv_root: data/raw/prices, cache_root: data/raw/cache,
           start_date: "2012-01-01", end_date: null, request_pause_seconds: 0}
  benchmarks: {symbols: [SPY, QQQ]}
  forms: {primary: [10-Q, 10-K], include_amendments_as_events: false}
features:
  mapping_version: v1
  periods_back: 3
  min_company_count: 2
  min_sample_coverage_pct: 0.1
  max_missing_pct: 99.0
target:
  k: 2.5
  scale:
    method: hybrid_daily_event
    daily_vol_lookback_days: 252
    event_vol_lookback_events: 8
    min_event_history: 4
    daily_weight: 0.7
    event_weight: 0.3
    min_ex_ante_scale: 0.02
    max_ex_ante_scale: 1.0
model:
  type: ridge
  random_seed: 42
  splits:
    train_end_date: "2014-12-31"
    validation_end_date: "2015-12-31"
    test_start_date: "2016-01-01"
  ridge: {alpha: 1.0}
portfolio:
  long_only: true
  allow_cash: true
  leverage: 1.0
  signal: {min_positive_signal: 0.0, signal_power: 1.0, k: 2.5, restore_clip_abs: 0.99,
           use_restored_return_for_sorting: false}
  positions: {min_positions: 1, target_positions: 4, max_positions: 4,
              max_single_stock_weight: 0.5, min_position_weight: 0.0}
  risk: {allocation_method: inverse_vol, risk_lookback_days: 120,
         min_daily_vol: 0.005, max_daily_vol: 0.10, covariance_shrinkage_lambda: 0.5}
  sector: {use_sector_caps: false, max_sector_weight: 0.30}
  rebalance: {frequency: daily_event_driven, execution_price: same_day_close}
  costs: {transaction_cost_bps_per_trade: 10, cash_return: 0.0}
backtest:
  start_date: "2015-01-05"
  end_date: "2018-06-29"
  initial_value: 1.0
  benchmarks: [SPY, QQQ]
  trading_days_per_year: 252
"""


def test_cli_build_train_backtest_chain(materialized_world):
    rc = main(["build-dataset"])
    assert rc == 0
    assert Path("data/processed/event_samples.parquet").exists()
    assert Path("artifacts/feature_encoder/v1/anonymous_feature_map.parquet").exists()

    rc = main(["train", "--model-kind", "ridge", "--model-id", "ridge_test"])
    assert rc == 0
    assert Path("artifacts/predictions/ridge_test.parquet").exists()

    rc = main(["backtest", "--predictions", "artifacts/predictions/ridge_test.parquet",
               "--portfolio-id", "ridge_test"])
    assert rc == 0
    assert Path("reports/backtest/ridge_test/daily.parquet").exists()
    assert Path("reports/backtest/ridge_test/metrics.csv").exists()


def test_walk_forward_produces_stitched_predictions(materialized_world):
    main(["build-dataset"])
    rc = main([
        "walk-forward",
        "--model-kind", "ridge",
        "--start", "2015-01-01",
        "--end", "2017-12-31",
        "--frequency", "QS",
        "--min-train-months", "12",
        "--out", "artifacts/predictions/walk_ridge.parquet",
    ])
    assert rc == 0
    out = pd.read_parquet("artifacts/predictions/walk_ridge.parquet")
    assert not out.empty
    assert "window" in out.columns
    # Multiple windows produced
    assert out["window"].nunique() >= 2
    # Predicted signals in range
    assert out["predicted_signal"].between(-1, 1).all()
