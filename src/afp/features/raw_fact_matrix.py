"""Assemble per-sample, periods_back × n_features fact tensors (RFC-03 §5-§7)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from afp.features.anonymous_mapping import AnonymousFeatureMap


@dataclass
class FactMatrixConfig:
    periods_back: int = 8


def _eligible_facts_for_sample(facts: pd.DataFrame, sample_accepted: str) -> pd.DataFrame:
    """RFC-02 §11: fact.accepted_datetime ≤ sample.accepted_datetime."""
    return facts[facts["accepted_datetime"] <= sample_accepted]


def _select_latest_per_period_concept(eligible: pd.DataFrame) -> pd.DataFrame:
    """RFC-03 §7 dedup: within (anonymous_feature_id, period_end_date), prefer
    10-K > 10-Q > others, then latest accepted_datetime, non-null value."""
    df = eligible.dropna(subset=["value"]).copy()
    df["_form_rank"] = (df["form_type"] == "10-K").astype(int) * 2 + \
                       (df["form_type"] == "10-Q").astype(int)
    df = df.sort_values(["anonymous_feature_id", "period_end_date",
                         "_form_rank", "accepted_datetime"],
                        ascending=[True, True, False, False])
    df = df.drop_duplicates(subset=["anonymous_feature_id", "period_end_date"], keep="first")
    return df.drop(columns=["_form_rank"])


def build_sample_matrices(
    samples: pd.DataFrame,
    facts: pd.DataFrame,
    feature_map: AnonymousFeatureMap,
    config: FactMatrixConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (values, missing, metadata, sample_ids).

    Shapes:
      values   [N, periods_back, F]   raw numeric values (NaN where missing)
      missing  [N, periods_back, F]   {0,1} missing flag
      metadata [N, periods_back, M]   periodic timing metadata (filing_delay_days etc.)
      sample_ids [N]
    """
    annotated = feature_map.join(facts)
    feature_ids = feature_map.feature_ids
    fid_to_col = {fid: i for i, fid in enumerate(feature_ids)}
    F = len(feature_ids)
    P = config.periods_back

    N = len(samples)
    values = np.full((N, P, F), np.nan, dtype=np.float64)
    missing = np.ones((N, P, F), dtype=np.int8)
    metadata = np.zeros((N, P, 3), dtype=np.float64)  # [filing_delay, fiscal_period_index, months_since_period_end]
    sample_ids: list[str] = []

    # Index facts per company for speed
    facts_by_company: dict[str, pd.DataFrame] = {
        icid: grp for icid, grp in annotated.groupby("internal_company_id", sort=False)
    }

    for i, row in enumerate(samples.itertuples()):
        sample_ids.append(row.sample_id)
        company_facts = facts_by_company.get(row.internal_company_id)
        if company_facts is None or company_facts.empty:
            continue
        eligible = _eligible_facts_for_sample(company_facts, row.accepted_datetime)
        if eligible.empty:
            continue
        kept = _select_latest_per_period_concept(eligible)

        # period_offset assignment — distinct period_end_date sorted descending
        period_ends = (kept["period_end_date"].dropna().sort_values(ascending=False)
                       .drop_duplicates().tolist())
        for offset, period_end in enumerate(period_ends[:P]):
            period_rows = kept[kept["period_end_date"] == period_end]
            for r in period_rows.itertuples():
                col = fid_to_col.get(r.anonymous_feature_id)
                if col is None:
                    continue
                v = r.value
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                values[i, offset, col] = float(v)
                missing[i, offset, col] = 0

            # metadata for this period_offset uses the most recent row in the period
            r = period_rows.iloc[0]
            metadata[i, offset, 0] = _filing_delay_days(r["accepted_datetime"], r["period_end_date"])
            metadata[i, offset, 1] = _fiscal_period_index(r.get("fiscal_period"))
            metadata[i, offset, 2] = _months_between(row.entry_date, r["period_end_date"])

    return values, missing, metadata, sample_ids


# ---------------------------------------------------------------------------

def _filing_delay_days(accepted: str | None, period_end: str | None) -> float:
    if not accepted or not period_end:
        return 0.0
    a = pd.Timestamp(accepted).date()
    p = pd.Timestamp(period_end).date()
    return float(min(180, max(0, (a - p).days)))


def _fiscal_period_index(fp: str | None) -> float:
    if not fp:
        return 0.0
    mapping = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}
    return float(mapping.get(str(fp).upper(), 0))


def _months_between(entry_date, period_end) -> float:
    if entry_date is None or period_end is None:
        return 0.0
    a = pd.Timestamp(entry_date).date()
    b = pd.Timestamp(period_end).date()
    months = (a - b).days / 30.4375
    return float(min(24.0, max(0.0, months)))
