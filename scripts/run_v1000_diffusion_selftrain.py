"""Phase 25: self-training loop with backtest reward reweighting.

Algorithm
---------
1. Use diffusion v2 predictions on the TRAIN split (computed via leave-one-quarter-out
   to avoid trivial leakage of the sample's own target through the model).
2. For each train sample, compute a reward:

       agreement = sign(predicted_signal) == sign(raw_log_return)
       confidence = abs(probability_positive - 0.5) * 2     # in [0,1]
       reward = (1 if agreement else -1) * confidence * |raw_log_return|

3. Convert reward to a sample weight:

       weight = exp(reward_scale * reward)
       weights renormalized to mean 1

   Successful, high-confidence picks get up-weighted; confidently-wrong picks get
   down-weighted; low-confidence picks stay near 1.

4. Retrain diffusion v3 with these weights.

The "leave-one-quarter-out" predictions on train are produced by training four
sub-models, each excluding one of train's calendar quartiles, then scoring the
held-out quartile. This prevents the reward signal from rewarding overfit
predictions of the sample's own target.

Test data is *never* touched in this loop.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afp.features.encoder import AnonymousFeatureEncoder
from afp.features.price_features import (
    PriceFeatureConfig,
    compute_price_features,
    fit_scaler,
    transform_with_scaler,
)
from afp.models.datasets import build_tabular_dataset
from afp.models.diffusion import ConditionalDiffusion, DiffusionConfig
from afp.models.metrics import regression_metrics
from afp.models.train import attach_prediction_context


def _dataset(samples_df, price_df, encoder, facts, feature_ids, price_ids, enable_price):
    scaled, missing, meta, sids = encoder.transform(samples_df, facts)
    targets = samples_df.set_index("sample_id")["target_normalized_signal"]
    base = build_tabular_dataset(scaled, missing, meta, sids, targets, feature_ids)
    if enable_price:
        sub = price_df.set_index("sample_id").reindex(base.sample_ids)
        extra = sub[price_ids].to_numpy(dtype=np.float64)
        extra_miss = sub[[f"{fid}_missing" for fid in price_ids]].to_numpy(dtype=np.float64)
        base.X = np.concatenate([base.X, extra, extra_miss], axis=1)
    return base


def _train_diffusion(X, y, epochs, sample_weight=None):
    cfg = DiffusionConfig(
        epochs=epochs,
        batch_size=256,
        timesteps=200,
        cond_hidden=1024, cond_latent=192,
        denoise_hidden=512, time_embed_dim=64,
        inference_samples=64,
        lr=2e-4,
        dropout=0.3,
    )
    model = ConditionalDiffusion(cfg)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def main(model_id: str = "diffusion_v3_self", epochs_per_round: int = 50,
         reward_scale: float = 8.0):
    samples = pd.read_parquet("data/processed/event_samples.parquet")
    facts = pd.read_parquet("data/processed/financial_facts_long.parquet")
    prices = pd.read_parquet("data/processed/prices_daily.parquet")
    encoder = AnonymousFeatureEncoder.load("artifacts/feature_encoder/v1")

    train = samples[samples["split"] == "train"].dropna(subset=["target_normalized_signal"]).copy()
    val = samples[samples["split"] == "validation"].dropna(subset=["target_normalized_signal"])
    test = samples[samples["split"] == "test"].dropna(subset=["target_normalized_signal"])
    train["entry_ts"] = pd.to_datetime(train["entry_date"])
    train["quartile"] = pd.qcut(train["entry_ts"].rank(method="first"), 4, labels=False)

    feature_ids = encoder.artifact.feature_map.feature_ids
    pf_cfg = PriceFeatureConfig(enabled=True)
    print("computing price features...")
    p_train, price_ids = compute_price_features(train, prices, pf_cfg)
    p_val, _ = compute_price_features(val, prices, pf_cfg)
    p_test, _ = compute_price_features(test, prices, pf_cfg)
    stats = fit_scaler(p_train, price_ids)
    p_train = transform_with_scaler(p_train, price_ids, stats)
    p_val = transform_with_scaler(p_val, price_ids, stats)
    p_test = transform_with_scaler(p_test, price_ids, stats)

    train_ds = _dataset(train, p_train, encoder, facts, feature_ids, price_ids, True)
    val_ds = _dataset(val, p_val, encoder, facts, feature_ids, price_ids, True)
    test_ds = _dataset(test, p_test, encoder, facts, feature_ids, price_ids, True)
    print(f"train X: {train_ds.X.shape}")

    # ----- Step 1: leave-one-quartile-out predictions on train -----
    sid_to_train_idx = {sid: i for i, sid in enumerate(train_ds.sample_ids)}
    sid_to_train_row = train.set_index("sample_id")
    train_pred = np.full(len(train_ds.sample_ids), np.nan, dtype=np.float64)
    train_prob_pos = np.full(len(train_ds.sample_ids), np.nan, dtype=np.float64)
    for q in range(4):
        out_mask = sid_to_train_row.loc[train_ds.sample_ids, "quartile"].to_numpy() == q
        in_mask = ~out_mask
        print(f"  fold q={q}: train {in_mask.sum()}, oos {out_mask.sum()}")
        sub_model = _train_diffusion(train_ds.X[in_mask], train_ds.y[in_mask], epochs_per_round)
        sub_out = sub_model.predict_distribution(train_ds.X[out_mask])
        train_pred[out_mask] = sub_out["mean"]
        train_prob_pos[out_mask] = sub_out["probability_positive"]

    # ----- Step 2: compute rewards -----
    realized = train["raw_log_return"].to_numpy()
    confidence = np.clip(np.abs(train_prob_pos - 0.5) * 2.0, 0.0, 1.0)
    agreement = (np.sign(train_pred) == np.sign(realized)).astype(float) * 2.0 - 1.0
    reward = agreement * confidence * np.abs(realized)
    weight = np.exp(reward_scale * reward)
    weight = weight / weight.mean()
    print(f"reward stats: min {reward.min():.4f}, max {reward.max():.4f}, mean {reward.mean():.4f}")
    print(f"weight stats: min {weight.min():.3f}, max {weight.max():.3f}, "
          f"std {weight.std():.3f}")

    # ----- Step 3: retrain diffusion v3 on full train with reward weights -----
    print(f"retraining diffusion v3 ({epochs_per_round * 2} epochs, reward-weighted)...")
    final_model = _train_diffusion(train_ds.X, train_ds.y, epochs_per_round * 2,
                                   sample_weight=weight)

    val_out = final_model.predict_distribution(val_ds.X)
    val_metrics = regression_metrics(val_ds.y, val_out["mean"])
    print("val metrics:", {k: round(v, 4) for k, v in val_metrics.items()})

    test_out = final_model.predict_distribution(test_ds.X)
    print(f"test mean range [{test_out['mean'].min():.3f}, {test_out['mean'].max():.3f}], "
          f"avg std {test_out['std'].mean():.3f}")

    rows = []
    for ds, out, name in ((val_ds, val_out, "validation"),
                          (test_ds, test_out, "test")):
        for i, sid in enumerate(ds.sample_ids):
            rows.append({
                "sample_id": sid,
                "predicted_signal": float(out["mean"][i]),
                "predicted_signal_std": float(out["std"][i]),
                "probability_positive": float(out["probability_positive"][i]),
                "q05": float(out["q05"][i]),
                "q95": float(out["q95"][i]),
                "split": name,
            })
    preds = pd.DataFrame(rows)
    preds["model_id"] = model_id
    preds["model_type"] = "diffusion_v3_selftrain"
    preds["prediction_created_at"] = datetime.now(tz=timezone.utc).isoformat()
    joined = attach_prediction_context(preds, samples)
    out_path = Path(f"artifacts/predictions/{model_id}.parquet")
    joined.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
