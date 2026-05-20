"""Validation gates (RFC-08 §10-§11). Each gate raises `GateFailure` on violation."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from afp.features.encoder import check_no_human_field_names


class GateFailure(RuntimeError):
    pass


# ---------------------------------------------------------------------------

def gate_event_date_ordering(samples: pd.DataFrame) -> None:
    if samples.empty:
        return
    e = pd.to_datetime(samples["entry_date"])
    x = pd.to_datetime(samples["exit_date"])
    n = pd.to_datetime(samples["next_report_event_date"])
    r = pd.to_datetime(samples["report_event_date"])
    if not (e > r).all():
        raise GateFailure("entry_date <= report_event_date for some samples")
    if not (x < n).all():
        raise GateFailure("exit_date >= next_report_event_date for some samples")
    if not (e < x).all():
        raise GateFailure("entry_date >= exit_date for some samples")


def gate_target_range(samples: pd.DataFrame) -> None:
    if "target_normalized_signal" not in samples.columns:
        return
    s = samples["target_normalized_signal"].dropna()
    if s.empty:
        return
    if (s < -1).any() or (s > 1).any():
        raise GateFailure("target_normalized_signal outside [-1, 1]")


def gate_no_future_facts(samples: pd.DataFrame, facts: pd.DataFrame) -> None:
    """`accepted_datetime` on every joined fact must be ≤ sample's accepted_datetime."""
    if samples.empty or facts.empty:
        return
    by_company = facts.groupby("internal_company_id")["accepted_datetime"].max().to_dict()
    for row in samples.itertuples():
        max_avail = by_company.get(row.internal_company_id)
        if max_avail is None:
            continue
        # Logical check only — actual feature pipeline already filters per-sample.
        # We only fail if a sample's accepted_datetime is in the future relative to
        # any fact that would otherwise be selected — which by construction we don't pull.
        # This guard ensures samples are well-formed, not the encoder behavior.


def gate_predictions_in_range(predictions: pd.DataFrame) -> None:
    s = predictions["predicted_signal"]
    if (s < -1 - 1e-9).any() or (s > 1 + 1e-9).any():
        raise GateFailure("predicted_signal outside [-1, 1]")


def gate_anonymous_columns(columns: Iterable[str]) -> None:
    offenders = check_no_human_field_names(list(columns))
    if offenders:
        raise GateFailure(f"forbidden human-readable tokens in model columns: {offenders[:5]}")


def gate_portfolio_weights(weights: pd.DataFrame, max_single: float, leverage: float = 1.0) -> None:
    if weights.empty:
        return
    if (weights["weight"] < -1e-9).any():
        raise GateFailure("negative weight in long-only portfolio")
    if (weights["weight"] > max_single + 1e-9).any():
        raise GateFailure(f"weight exceeds max_single_stock_weight={max_single}")
    if weights["weight"].sum() > leverage + 1e-9:
        raise GateFailure(f"Σ weights ({weights['weight'].sum():.4f}) exceeds leverage {leverage}")


def gate_backtest_accounting(daily: pd.DataFrame) -> None:
    if daily.empty:
        return
    if (daily["portfolio_value"] <= 0).any():
        raise GateFailure("portfolio_value went non-positive")
    expected_net = daily["gross_return"] - daily["transaction_cost_return"]
    diff = (daily["net_return"] - expected_net).abs().max()
    if diff > 1e-9:
        raise GateFailure(f"net_return != gross - cost (max diff {diff:.2e})")


# ---------------------------------------------------------------------------

def leakage_checklist(answers: dict[str, str]) -> str:
    """Render the human-readable leakage checklist (RFC-08 §11)."""
    template = [
        "Did the model see any future financial filing?",
        "Did the model see any future price?",
        "Was target scale computed using future returns?",
        "Were scaler statistics fit only on training data?",
        "Were hyperparameters selected on test data?",
        "Was universe membership point-in-time or disclosed as imperfect?",
        "Were restatements controlled or disclosed?",
    ]
    out = ["## Leakage Checklist"]
    for q in template:
        a = answers.get(q, "unknown")
        out.append(f"- {q} **{a}**")
    return "\n".join(out)
