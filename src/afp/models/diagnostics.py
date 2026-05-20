"""Phase 11: signal diagnostics for the LightGBM (or any) prediction parquet.

Answers the key validation questions:
  - Does predicted_signal sort future realized returns?
  - Is the rank ordering stable across quarters?
  - Is the top minus bottom decile spread positive after costs?
  - Does excess vs SPY / vs equal-risk universe come from real signal or from sampling?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class DiagnosticsConfig:
    n_buckets: int = 10
    benchmark_column_spy: str = "spy_log_return"


# ---------------------------------------------------------------------------

def decile_table(joined: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """Bin `predicted_signal` into deciles; report realized stats per bin.

    `joined` must contain: predicted_signal, raw_log_return,
    target_normalized_signal, spy_log_return (optional). One row per sample.
    """
    df = joined.dropna(subset=["predicted_signal", "raw_log_return"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(df["predicted_signal"], q=n_buckets,
                           labels=False, duplicates="drop")
    agg_cols = {
        "predicted_signal": "mean",
        "raw_log_return": ["mean", "median", "count"],
    }
    if "target_normalized_signal" in df.columns:
        agg_cols["target_normalized_signal"] = "mean"
    if "spy_log_return" in df.columns:
        df["excess_vs_spy"] = df["raw_log_return"] - df["spy_log_return"]
        agg_cols["excess_vs_spy"] = "mean"
    out = df.groupby("bucket").agg(agg_cols)
    out.columns = ["_".join(c).strip("_") for c in out.columns.to_flat_index()]
    out = out.reset_index().rename(columns={"raw_log_return_count": "count"})
    return out


def top_minus_bottom_spread(joined: pd.DataFrame, n_buckets: int = 10) -> float:
    """Mean realized log return of top decile minus mean of bottom decile."""
    df = joined.dropna(subset=["predicted_signal", "raw_log_return"]).copy()
    if df.empty:
        return float("nan")
    df["bucket"] = pd.qcut(df["predicted_signal"], q=n_buckets,
                           labels=False, duplicates="drop")
    grouped = df.groupby("bucket")["raw_log_return"].mean()
    if grouped.empty:
        return float("nan")
    return float(grouped.iloc[-1] - grouped.iloc[0])


def quarterly_rank_correlation(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-quarter Spearman correlation between predicted_signal and raw_log_return."""
    df = joined.dropna(subset=["predicted_signal", "raw_log_return", "entry_date"]).copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["quarter"] = df["entry_date"].dt.to_period("Q")
    rows = []
    for q, grp in df.groupby("quarter"):
        if len(grp) < 5:
            rho = float("nan")
        else:
            rho, _ = spearmanr(grp["predicted_signal"], grp["raw_log_return"])
        rows.append({"quarter": str(q), "n": int(len(grp)), "spearman": float(rho)})
    return pd.DataFrame(rows)


def quarterly_direction_accuracy(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.dropna(subset=["predicted_signal", "raw_log_return", "entry_date"]).copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["quarter"] = df["entry_date"].dt.to_period("Q")
    df["correct"] = np.sign(df["predicted_signal"]) == np.sign(df["raw_log_return"])
    g = df.groupby("quarter").agg(n=("correct", "size"),
                                  direction_accuracy=("correct", "mean")).reset_index()
    g["quarter"] = g["quarter"].astype(str)
    return g


def positive_signal_vs_random(joined: pd.DataFrame, n_holdings: int = 50,
                              seed: int = 42, n_trials: int = 25) -> dict[str, float]:
    """Compare mean realized return of top-N by predicted_signal vs N random positive picks.

    For each event date (using `entry_date` as proxy), pick the N largest
    `positive_signal` (replacement with random pick if fewer than N).
    """
    df = joined.dropna(subset=["predicted_signal", "raw_log_return", "entry_date"]).copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    rng = np.random.default_rng(seed)
    model_returns: list[float] = []
    random_returns_per_trial: list[list[float]] = [[] for _ in range(n_trials)]
    for d, grp in df.groupby("entry_date"):
        positives = grp[grp["predicted_signal"] > 0]
        if positives.empty:
            continue
        top = positives.nlargest(min(n_holdings, len(positives)), "predicted_signal")
        model_returns.append(float(top["raw_log_return"].mean()))
        for t in range(n_trials):
            pick = positives.sample(min(n_holdings, len(positives)), random_state=int(rng.integers(0, 1_000_000)))
            random_returns_per_trial[t].append(float(pick["raw_log_return"].mean()))
    if not model_returns:
        return {"model_mean": float("nan"), "random_mean": float("nan"),
                "spread": float("nan"), "model_t_stat": float("nan")}
    model_mean = float(np.mean(model_returns))
    random_means = [float(np.mean(r)) for r in random_returns_per_trial if r]
    random_mean = float(np.mean(random_means))
    random_std = float(np.std(random_means, ddof=1)) if len(random_means) > 1 else 0.0
    t_stat = (model_mean - random_mean) / max(random_std / np.sqrt(len(random_means)), 1e-12) \
        if random_std > 0 else float("nan")
    return {"model_mean": model_mean, "random_mean": random_mean,
            "spread": model_mean - random_mean, "model_t_stat": t_stat,
            "n_trials": int(len(random_means))}


def build_diagnostic_report(joined: pd.DataFrame,
                            config: DiagnosticsConfig | None = None,
                            split: str | None = "test") -> dict:
    config = config or DiagnosticsConfig()
    df = joined.copy()
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    return {
        "n_samples": int(len(df)),
        "decile_table": decile_table(df, config.n_buckets).to_dict(orient="records"),
        "top_minus_bottom_log_return": top_minus_bottom_spread(df, config.n_buckets),
        "quarterly_spearman": quarterly_rank_correlation(df).to_dict(orient="records"),
        "quarterly_direction_accuracy": quarterly_direction_accuracy(df).to_dict(orient="records"),
        "positive_signal_vs_random": positive_signal_vs_random(df),
    }
