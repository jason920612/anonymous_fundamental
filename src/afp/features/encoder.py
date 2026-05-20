"""High-level feature encoder bundling mapping + matrix + scaling (RFC-03)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from afp.features.anonymous_mapping import AnonymousFeatureMap
from afp.features.derived import compute_derived
from afp.features.normalization import NormalizerConfig, RobustScaler
from afp.features.raw_fact_matrix import FactMatrixConfig, build_sample_matrices

FORBIDDEN_SUBSTRINGS = (
    "Revenue", "Income", "Asset", "Liability", "Cash", "Debt", "Equity",
    "EPS", "Margin", "ROE", "ROA", "Earnings",
)


@dataclass
class EncoderArtifact:
    feature_map: AnonymousFeatureMap
    scaler: RobustScaler
    periods_back: int
    derived_scaler: RobustScaler | None = None   # Phase 14
    derived_enabled: bool = False
    derived_yoy_lookback: int = 4

    def num_features(self) -> int:
        n = len(self.feature_map.feature_ids)
        if self.derived_enabled:
            n += 2 * len(self.feature_map.feature_ids)   # yoy + z per base feature
        return n


class AnonymousFeatureEncoder:
    """Fit on training samples; transform train/val/test using same statistics."""

    def __init__(self,
                 periods_back: int = 8,
                 min_company_count: int = 100,
                 min_sample_coverage_pct: float = 1.0,
                 max_missing_pct: float = 99.0,
                 mapping_version: str = "v1",
                 normalizer_config: NormalizerConfig | None = None,
                 derived_enabled: bool = False,
                 derived_yoy_lookback: int = 4):
        self.cfg = FactMatrixConfig(periods_back=periods_back)
        self.min_company_count = min_company_count
        self.min_sample_coverage_pct = min_sample_coverage_pct
        self.max_missing_pct = max_missing_pct
        self.mapping_version = mapping_version
        self.normalizer_config = normalizer_config or NormalizerConfig()
        self.derived_enabled = derived_enabled
        self.derived_yoy_lookback = derived_yoy_lookback
        self.artifact: EncoderArtifact | None = None

    # ------------------------------------------------------------------

    def fit(self, train_samples: pd.DataFrame, facts: pd.DataFrame) -> EncoderArtifact:
        # Mapping uses only facts visible at-or-before the training samples' accepted_datetime.
        max_accepted = train_samples["accepted_datetime"].max()
        train_facts = facts[facts["accepted_datetime"] <= max_accepted]
        feature_map = AnonymousFeatureMap.fit(
            train_facts,
            mapping_version=self.mapping_version,
            min_company_count=self.min_company_count,
            min_sample_coverage_pct=self.min_sample_coverage_pct,
            max_missing_pct=self.max_missing_pct,
            num_samples=len(train_samples),
        )
        values, missing, _, _ = build_sample_matrices(train_samples, facts, feature_map, self.cfg)
        scaler = RobustScaler(self.normalizer_config).fit(values, missing)

        derived_scaler = None
        if self.derived_enabled:
            d_vals, d_missing = compute_derived(values, missing, yoy_lookback=self.derived_yoy_lookback)
            derived_scaler = RobustScaler(self.normalizer_config).fit(d_vals, d_missing)

        self.artifact = EncoderArtifact(feature_map=feature_map, scaler=scaler,
                                        periods_back=self.cfg.periods_back,
                                        derived_scaler=derived_scaler,
                                        derived_enabled=self.derived_enabled,
                                        derived_yoy_lookback=self.derived_yoy_lookback)
        return self.artifact

    def transform(self, samples: pd.DataFrame, facts: pd.DataFrame
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if self.artifact is None:
            raise RuntimeError("encoder not fit")
        values, missing, metadata, sample_ids = build_sample_matrices(
            samples, facts, self.artifact.feature_map, self.cfg
        )
        scaled = self.artifact.scaler.transform(values, missing)
        if self.artifact.derived_enabled and self.artifact.derived_scaler is not None:
            d_vals, d_missing = compute_derived(values, missing,
                                                yoy_lookback=self.artifact.derived_yoy_lookback)
            d_scaled = self.artifact.derived_scaler.transform(d_vals, d_missing)
            # Pad to periods_back depth so downstream code treats derived as offset=0 only.
            P = self.cfg.periods_back
            N, _, D = d_scaled.shape
            d_scaled_padded = np.zeros((N, P, D), dtype=d_scaled.dtype)
            d_missing_padded = np.ones((N, P, D), dtype=d_missing.dtype)
            d_scaled_padded[:, 0, :] = d_scaled[:, 0, :]
            d_missing_padded[:, 0, :] = d_missing[:, 0, :]
            scaled = np.concatenate([scaled, d_scaled_padded], axis=2)
            missing = np.concatenate([missing.astype(np.int8),
                                      d_missing_padded.astype(np.int8)], axis=2)
        return scaled, missing.astype(np.float32), metadata.astype(np.float32), sample_ids

    # ------------------------------------------------------------------

    def to_dense_frame(self, scaled: np.ndarray, missing: np.ndarray,
                       metadata: np.ndarray, sample_ids: list[str]) -> pd.DataFrame:
        """Flatten to one row per sample with anonymized column names."""
        N, P, F = scaled.shape
        feature_ids = list(self.artifact.feature_map.feature_ids)
        base_F = len(feature_ids)
        if self.artifact.derived_enabled and F > base_F:
            # Derived block follows base block: first base_F YoY, then base_F z-score.
            feature_ids += [f"derived_yoy_{fid}" for fid in feature_ids[:base_F]]
            feature_ids += [f"derived_z_{fid}" for fid in feature_ids[:base_F]]
        data: dict[str, np.ndarray] = {"sample_id": np.array(sample_ids)}
        for f, fid in enumerate(feature_ids):
            for p in range(P):
                data[f"{fid}_o{p}_value"] = scaled[:, p, f]
                data[f"{fid}_o{p}_missing"] = missing[:, p, f]
        meta_names = ["meta_filing_delay_days", "meta_fiscal_period_index",
                      "meta_months_since_period_end"]
        for m, name in enumerate(meta_names):
            for p in range(P):
                data[f"{name}_o{p}"] = metadata[:, p, m]
        return pd.DataFrame(data)

    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.artifact.feature_map.to_parquet(directory / "anonymous_feature_map.parquet")
        np.savez(directory / "scaling_stats.npz",
                 median=self.artifact.scaler.median, iqr=self.artifact.scaler.iqr)
        meta = {
            "periods_back": self.artifact.periods_back,
            "num_features": self.artifact.num_features(),
            "mapping_version": self.artifact.feature_map.mapping_version,
            "derived_enabled": self.artifact.derived_enabled,
            "derived_yoy_lookback": self.artifact.derived_yoy_lookback,
        }
        (directory / "metadata.json").write_text(
            __import__("json").dumps(meta, indent=2))
        if self.artifact.derived_enabled and self.artifact.derived_scaler is not None:
            np.savez(directory / "derived_scaling_stats.npz",
                     median=self.artifact.derived_scaler.median,
                     iqr=self.artifact.derived_scaler.iqr)

    @classmethod
    def load(cls, directory: str | Path) -> "AnonymousFeatureEncoder":
        import json
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        encoder = cls(periods_back=meta["periods_back"],
                      mapping_version=meta["mapping_version"],
                      derived_enabled=meta.get("derived_enabled", False),
                      derived_yoy_lookback=meta.get("derived_yoy_lookback", 4))
        feature_map = AnonymousFeatureMap.from_parquet(
            directory / "anonymous_feature_map.parquet", mapping_version=meta["mapping_version"]
        )
        scaler = RobustScaler()
        npz = np.load(directory / "scaling_stats.npz")
        scaler.median = npz["median"]
        scaler.iqr = npz["iqr"]
        derived_scaler = None
        derived_npz = directory / "derived_scaling_stats.npz"
        if derived_npz.exists():
            d = np.load(derived_npz)
            derived_scaler = RobustScaler()
            derived_scaler.median = d["median"]
            derived_scaler.iqr = d["iqr"]
        encoder.artifact = EncoderArtifact(
            feature_map=feature_map, scaler=scaler,
            periods_back=meta["periods_back"],
            derived_scaler=derived_scaler,
            derived_enabled=meta.get("derived_enabled", False),
            derived_yoy_lookback=meta.get("derived_yoy_lookback", 4),
        )
        return encoder


# ---------------------------------------------------------------------------

def check_no_human_field_names(columns: list[str],
                               forbidden: tuple[str, ...] = FORBIDDEN_SUBSTRINGS) -> list[str]:
    """Return list of offending column names. Used by leakage gates and tests."""
    offenders: list[str] = []
    for col in columns:
        for token in forbidden:
            if token.lower() in col.lower() and not col.startswith("meta_"):
                offenders.append(col)
                break
    return offenders
