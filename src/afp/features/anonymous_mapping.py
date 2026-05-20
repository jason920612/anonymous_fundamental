"""Stable anonymous feature ID mapping (RFC-03 §3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AnonymousFeatureMap:
    """Map `(taxonomy, concept_name, unit)` ↔ `feature_NNNNNN`."""

    mapping_version: str = "v1"
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["anonymous_feature_id", "taxonomy", "concept_name", "unit",
                 "mapping_version", "first_seen_at", "active"]))

    @classmethod
    def fit(cls, facts: pd.DataFrame, mapping_version: str = "v1",
            min_company_count: int = 100,
            min_sample_coverage_pct: float = 1.0,
            max_missing_pct: float = 99.0,
            num_samples: int | None = None) -> "AnonymousFeatureMap":
        """Assign deterministic anonymous IDs to surviving features.

        Filters per RFC-03 §4.2-§4.3:
          - keep features present in at least `min_company_count` companies;
          - keep features whose coverage ≥ `min_sample_coverage_pct` of `num_samples`
            (only when `num_samples` is provided);
          - drop features with > `max_missing_pct` missingness (same constraint).
        """
        df = facts[["taxonomy", "concept_name", "unit", "internal_company_id"]].copy()
        df = df.dropna(subset=["concept_name", "unit"])
        group_cols = ["taxonomy", "concept_name", "unit"]
        stats = df.groupby(group_cols)["internal_company_id"].nunique().rename("company_count")
        stats = stats.reset_index()

        if num_samples is not None and num_samples > 0:
            coverage = df.groupby(group_cols).size().rename("fact_count").reset_index()
            stats = stats.merge(coverage, on=group_cols, how="left")
            stats["coverage_pct"] = 100.0 * stats["fact_count"] / num_samples
        else:
            stats["coverage_pct"] = float("inf")
            stats["fact_count"] = 0

        keep = (stats["company_count"] >= min_company_count) & \
               (stats["coverage_pct"] >= min_sample_coverage_pct) & \
               ((100.0 - stats["coverage_pct"]) <= max_missing_pct)
        kept = stats.loc[keep, group_cols].sort_values(group_cols).reset_index(drop=True)

        ids = [f"feature_{i:06d}" for i in range(len(kept))]
        table = kept.assign(
            anonymous_feature_id=ids,
            mapping_version=mapping_version,
            first_seen_at=pd.Timestamp.utcnow().isoformat(),
            active=True,
        )[["anonymous_feature_id", "taxonomy", "concept_name", "unit",
           "mapping_version", "first_seen_at", "active"]]
        return cls(mapping_version=mapping_version, table=table)

    # ------------------------------------------------------------------

    @property
    def feature_ids(self) -> list[str]:
        return self.table["anonymous_feature_id"].tolist()

    def join(self, facts: pd.DataFrame) -> pd.DataFrame:
        """Attach `anonymous_feature_id` to `facts`; rows that don't map are dropped."""
        merged = facts.merge(
            self.table[["taxonomy", "concept_name", "unit", "anonymous_feature_id"]],
            on=["taxonomy", "concept_name", "unit"],
            how="inner",
        )
        return merged

    def to_parquet(self, path: str) -> None:
        self.table.to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str, mapping_version: str = "v1") -> "AnonymousFeatureMap":
        return cls(mapping_version=mapping_version, table=pd.read_parquet(path))
