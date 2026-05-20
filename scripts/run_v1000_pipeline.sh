#!/usr/bin/env bash
# Phase 11: complete 1000-CIK pipeline after `afp ingest-sec` finishes.
# Usage: scripts/run_v1000_pipeline.sh [model_kind]
set -euo pipefail

MODEL_KIND="${1:-lightgbm}"
CONFIG="configs/deployment_v1000.yaml"
PRED="artifacts/predictions/walk_v1000.parquet"
PORTFOLIO_ID="walk_v1000"

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "[$(date +%T)] ingest-prices"
python3 -m afp.cli.main ingest-prices --config "$CONFIG" --limit 1000

echo "[$(date +%T)] build-dataset"
python3 -m afp.cli.main build-dataset --config "$CONFIG"

echo "[$(date +%T)] walk-forward retraining"
python3 -m afp.cli.main walk-forward \
    --config "$CONFIG" \
    --model-kind "$MODEL_KIND" \
    --frequency QS \
    --start 2020-01-02 \
    --min-train-months 36 \
    --out "$PRED"

echo "[$(date +%T)] backtest"
python3 -m afp.cli.main backtest \
    --config "$CONFIG" \
    --predictions "$PRED" \
    --portfolio-id "$PORTFOLIO_ID"

echo "[$(date +%T)] diagnostics"
python3 -m afp.cli.main diagnostics \
    --predictions "$PRED" \
    --portfolio-id "$PORTFOLIO_ID"

echo "[$(date +%T)] finalize"
python3 -m afp.cli.main finalize --portfolio-id "$PORTFOLIO_ID"

echo "[$(date +%T)] done — reports/backtest/$PORTFOLIO_ID/summary.md"
