# Phase 9 — Experiment Management

Maps to **RFC-08** and **RFC-09 Milestone 9**.

## Modules

| Module | Role |
|---|---|
| `afp.experiments.validation` | Pipeline gates + leakage checklist renderer |
| `afp.experiments.registry` | `ExperimentRun`, `make_experiment_id`, `append_registry`, `build_experiment_metadata` |
| `afp.experiments.smoke_run` | End-to-end wired pipeline used as the canonical smoke test |

## Validation Gates

`GateFailure` is raised on any violation; pipelines should call gates at the
boundaries between phases.

| Gate | Asserts |
|---|---|
| `gate_event_date_ordering` | RFC-02 §17.1: `entry > report`, `entry < exit`, `exit < next_report` |
| `gate_target_range` | `target_normalized_signal ∈ [-1, 1]` |
| `gate_predictions_in_range` | `predicted_signal ∈ [-1, 1]` |
| `gate_anonymous_columns` | RFC-03 §15.1: no forbidden financial words in model column names |
| `gate_portfolio_weights` | RFC-06: long-only, `weight ≤ max_single_stock_weight`, `Σw ≤ leverage` |
| `gate_backtest_accounting` | `net_return == gross - cost`, `portfolio_value > 0` |

## Experiment ID

```
{YYYYMMDD_HHMMSS}_{owner}_{model_type}_{feature_version}_{target_version}_{portfolio_method}_{config_hash[:8]}
```

`config_hash` is the canonical sha256 over the resolved config (Phase 1's
`afp.utils.config.config_hash`), so two runs with identical configs share the
same trailing prefix — useful for grep'ing the registry.

## ExperimentRun

Each run owns `artifacts/experiments/<experiment_id>/` and writes:

```
resolved_config.yaml      ← full merged config
metadata.json             ← experiment_id, owner, purpose, config_hash, git state, model metrics
metrics.json              ← backtest summary
predictions.parquet       ← model predictions
backtest_daily.parquet    ← per-day equity/cash/turnover
backtest_trades.parquet   ← trade log
report.md                 ← human-readable summary + leakage checklist
```

The CSV-backed registry (`artifacts/experiments/registry.csv`) appends one row
per run: experiment_id, config_hash, git_commit, model_type, sharpe,
max_drawdown, etc.

## Leakage Checklist (RFC-08 §11)

`leakage_checklist(answers)` renders the canonical seven questions as Markdown
bullets so every report carries the disclosure. Default answers from the smoke
run are `no / no / no / yes / no / disclosed / disclosed`.

## Smoke Run

`python -m afp.experiments.smoke_run`:

1. Loads `configs/default.yaml`.
2. Builds the synthetic 6-year, 4-company world.
3. Runs the full pipeline Phase 2 → 8.
4. Calls every gate above between phases.
5. Persists a complete experiment directory and appends to the registry.

This run is the integration test (`tests/test_experiments.py::test_smoke_pipeline_runs_end_to_end`).

## Acceptance — verified by `tests/test_experiments.py`

- Each gate passes on well-formed inputs and raises on violations.
- `build_experiment_metadata` returns an id ending in the config hash prefix.
- `append_registry` is create-or-append idempotent and preserves order.
- `leakage_checklist` renders all seven RFC-mandated questions.
- The smoke run completes and returns metrics + a registered experiment ID.
