"""Phase 21: anonymous price-derived features.

Per sample we compute trailing log returns at multiple horizons from the
company's own price history strictly before `entry_date`. The horizons are
chosen but the model sees only opaque IDs (`price_feature_001..N`) — no
human label like "momentum_3m" enters the column space.

Horizons (configurable):
  - 21 trading days (~1 month)
  - 63 (~3 months)
  - 126 (~6 months)
  - 252 (~12 months)
  - 21..63 (relative momentum: 1m return minus 3m return)
  - rolling volatility 63d  (as a vol feature, anonymized)

Each is a single scalar per sample. They are appended to the encoder's
output as extra anonymous columns; the encoder fits its own scaler on
training data only.

RFC compliance: this passes the "model can't know meaning" test because
each feature is exposed as an opaque ID. The user has lifted the
V1 §11 restriction on price momentum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PriceFeatureConfig:
    enabled: bool = False
    horizons_days: tuple[int, ...] = (21, 63, 126, 252)
    include_relative_momentum: bool = True
    include_rolling_vol: bool = True
    vol_window_days: int = 63


def compute_price_features(
    samples: pd.DataFrame,
    prices: pd.DataFrame,
    config: PriceFeatureConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Return `(features_df, feature_ids)`.

    `features_df` has columns `sample_id` + `price_feature_NNN` (one per signal).
    All lookups use prices strictly before each sample's `entry_date`.
    """
    if not config.enabled or samples.empty:
        return pd.DataFrame({"sample_id": samples.get("sample_id", [])}), []

    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["internal_company_id", "date"])
    # Per company: log returns + rolling vol precomputed
    cache: dict[str, dict] = {}
    for icid, grp in px.groupby("internal_company_id", sort=False):
        grp = grp.copy()
        grp["log_close"] = np.log(grp["adjusted_close"].astype(float))
        grp["log_return"] = grp["log_close"].diff()
        cache[icid] = {
            "dates_ns": grp["date"].astype("datetime64[ns]").astype("int64").to_numpy(),
            "log_close": grp["log_close"].to_numpy(),
            "log_return": grp["log_return"].to_numpy(),
        }

    n_signals = len(config.horizons_days) + (1 if config.include_relative_momentum else 0) \
        + (1 if config.include_rolling_vol else 0)
    feature_ids = [f"price_feature_{i:03d}" for i in range(n_signals)]

    out_rows: list[dict] = []
    for row in samples.itertuples():
        entry_ts = pd.Timestamp(row.entry_date)
        cache_entry = cache.get(row.internal_company_id)
        if cache_entry is None:
            out_rows.append({"sample_id": row.sample_id,
                             **{fid: np.nan for fid in feature_ids}})
            continue
        entry_ns = np.datetime64(pd.Timestamp(entry_ts), "ns").astype("int64")
        idx = int(np.searchsorted(cache_entry["dates_ns"], entry_ns, side="left"))
        feats: dict[str, float] = {}
        for h in config.horizons_days:
            if idx - 1 - h >= 0:
                feats[f"price_feature_{len(feats):03d}"] = float(
                    cache_entry["log_close"][idx - 1] - cache_entry["log_close"][idx - 1 - h]
                )
            else:
                feats[f"price_feature_{len(feats):03d}"] = np.nan
        if config.include_relative_momentum:
            # 1m return minus 3m return
            if idx - 1 - 21 >= 0 and idx - 1 - 63 >= 0:
                r1 = cache_entry["log_close"][idx - 1] - cache_entry["log_close"][idx - 1 - 21]
                r3 = cache_entry["log_close"][idx - 1] - cache_entry["log_close"][idx - 1 - 63]
                feats[f"price_feature_{len(feats):03d}"] = float(r1 - r3)
            else:
                feats[f"price_feature_{len(feats):03d}"] = np.nan
        if config.include_rolling_vol:
            window = config.vol_window_days
            if idx - 1 - window >= 0:
                window_returns = cache_entry["log_return"][idx - 1 - window: idx - 1]
                feats[f"price_feature_{len(feats):03d}"] = float(np.nanstd(window_returns, ddof=1))
            else:
                feats[f"price_feature_{len(feats):03d}"] = np.nan
        feats["sample_id"] = row.sample_id
        out_rows.append(feats)

    return pd.DataFrame(out_rows), feature_ids


def fit_scaler(df: pd.DataFrame, feature_ids: list[str]) -> dict:
    """Return per-feature `{median, iqr}` from non-null values."""
    stats: dict[str, tuple[float, float]] = {}
    for fid in feature_ids:
        col = df[fid].dropna()
        if col.empty:
            stats[fid] = (0.0, 1.0)
            continue
        med = float(np.median(col))
        iqr = float(np.percentile(col, 75) - np.percentile(col, 25)) or 1e-6
        stats[fid] = (med, iqr)
    return stats


def transform_with_scaler(df: pd.DataFrame, feature_ids: list[str],
                          stats: dict, clip: tuple[float, float] = (-10.0, 10.0)
                          ) -> pd.DataFrame:
    out = df.copy()
    for fid in feature_ids:
        med, iqr = stats[fid]
        scaled = (out[fid] - med) / max(iqr, 1e-6)
        out[fid] = scaled.clip(*clip).fillna(0.0)
        out[f"{fid}_missing"] = df[fid].isna().astype(np.int8)
    return out
