"""HPO grid search for LightGBM on the static v1000 encoder.

Runs only on train + validation rows (test is structurally excluded).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json

import pandas as pd

from afp.features.encoder import AnonymousFeatureEncoder
from afp.models.hpo import grid_search


def main():
    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")
    train_val = samples[samples["split"].isin(("train", "validation"))].dropna(
        subset=["target_normalized_signal"]
    )
    print(f"HPO over {len(train_val)} train+val samples")

    grid = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.03, 0.05],
        "min_data_in_leaf": [50, 100, 200],
        "feature_fraction": [0.5, 0.8],
        "n_estimators": [300],
    }

    result = grid_search(
        model_kind="gbt",
        param_grid=grid,
        encoder=encoder,
        train_val_samples=train_val,
        facts=facts,
        metric="spearman",
        n_folds=4,
        sample_weight_kwargs={"by_company": True, "by_quarter": True},
    )

    out = Path("artifacts/hpo/v1000_static.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "best_params": result.best_params,
        "best_metric": result.best_metric,
        "n_trials": len(result.trials),
        "trials": [{"params": t["params"], "spearman": t["spearman"]} for t in result.trials],
    }, indent=2, default=str))
    print(f"best params: {result.best_params}")
    print(f"best spearman: {result.best_metric:.4f}")


if __name__ == "__main__":
    main()
