"""Phase 9 tests — gates, registry, smoke run."""

from pathlib import Path

import pandas as pd
import pytest

from afp.experiments.registry import append_registry, build_experiment_metadata
from afp.experiments.validation import (
    GateFailure,
    gate_anonymous_columns,
    gate_backtest_accounting,
    gate_event_date_ordering,
    gate_portfolio_weights,
    gate_predictions_in_range,
    gate_target_range,
    leakage_checklist,
)


def test_gate_event_date_ordering_passes_on_well_formed_samples():
    samples = pd.DataFrame({
        "report_event_date": ["2020-01-01", "2020-05-01"],
        "entry_date": ["2020-01-03", "2020-05-04"],
        "exit_date": ["2020-04-30", "2020-07-30"],
        "next_report_event_date": ["2020-05-01", "2020-07-31"],
    })
    gate_event_date_ordering(samples)


def test_gate_event_date_ordering_fails_on_bad_order():
    samples = pd.DataFrame({
        "report_event_date": ["2020-01-01"],
        "entry_date": ["2020-01-01"],
        "exit_date": ["2020-04-30"],
        "next_report_event_date": ["2020-05-01"],
    })
    with pytest.raises(GateFailure):
        gate_event_date_ordering(samples)


def test_gate_target_range_passes_in_unit_interval():
    s = pd.DataFrame({"target_normalized_signal": [-0.9, 0.0, 0.5, 0.99]})
    gate_target_range(s)


def test_gate_target_range_fails_out_of_range():
    s = pd.DataFrame({"target_normalized_signal": [-1.5, 0.0]})
    with pytest.raises(GateFailure):
        gate_target_range(s)


def test_gate_predictions_in_range():
    p = pd.DataFrame({"predicted_signal": [0.1, -0.9, 0.99]})
    gate_predictions_in_range(p)
    with pytest.raises(GateFailure):
        gate_predictions_in_range(pd.DataFrame({"predicted_signal": [1.1]}))


def test_gate_anonymous_columns_blocks_human_tokens():
    with pytest.raises(GateFailure):
        gate_anonymous_columns(["sample_id", "Revenues_o0_value"])
    gate_anonymous_columns(["sample_id", "feature_000001_o0_value", "meta_filing_delay_days_o0"])


def test_gate_portfolio_weights_enforces_cap_and_long_only():
    weights = pd.DataFrame({"internal_company_id": ["A", "B"], "weight": [0.4, 0.5]})
    gate_portfolio_weights(weights, max_single=0.5, leverage=1.0)
    with pytest.raises(GateFailure):
        gate_portfolio_weights(pd.DataFrame({"internal_company_id": ["A"], "weight": [0.6]}),
                               max_single=0.5)
    with pytest.raises(GateFailure):
        gate_portfolio_weights(pd.DataFrame({"internal_company_id": ["A"], "weight": [-0.1]}),
                               max_single=0.5)


def test_gate_backtest_accounting_catches_net_mismatch():
    daily = pd.DataFrame({
        "portfolio_value": [1.0, 1.01],
        "gross_return": [0.0, 0.012],
        "transaction_cost_return": [0.0, 0.002],
        "net_return": [0.0, 0.010],
    })
    gate_backtest_accounting(daily)
    bad = daily.copy()
    bad.loc[1, "net_return"] = 0.011  # inconsistent
    with pytest.raises(GateFailure):
        gate_backtest_accounting(bad)


def test_registry_metadata_includes_hash_and_id():
    cfg = {"project": {"random_seed": 42}, "target": {"k": 2.5}}
    exp_id, meta = build_experiment_metadata(
        cfg, owner="t", purpose="p", model_type="ridge",
        feature_version="v1", target_version="v1", portfolio_method="inv_vol",
        repo_root=Path("."))
    assert exp_id.endswith(meta["config_hash"][:8])
    assert meta["random_seed"] == 42


def test_registry_append_creates_and_grows(tmp_path):
    p = tmp_path / "registry.csv"
    append_registry(p, {"experiment_id": "exp1", "sharpe": 0.5})
    append_registry(p, {"experiment_id": "exp2", "sharpe": 0.7})
    df = pd.read_csv(p)
    assert list(df["experiment_id"]) == ["exp1", "exp2"]


def test_leakage_checklist_renders_all_questions():
    md = leakage_checklist({})
    for q in ("future financial filing", "future price", "future returns",
              "training data", "test data", "point-in-time", "restatements"):
        assert q in md.lower()


def test_smoke_pipeline_runs_end_to_end():
    from afp.experiments.smoke_run import run
    out = run()
    assert "experiment_id" in out
    assert "metrics" in out
    assert "backtest" in out
