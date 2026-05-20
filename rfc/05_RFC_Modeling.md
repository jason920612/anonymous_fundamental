# RFC-05: Modeling

Status: Draft  
Phase: 5  
Purpose: Define model training, baselines, representation learning, optional diffusion model, prediction outputs, validation, and model acceptance criteria.

---

## 1. Objective

Train models that map anonymous raw financial representations to a bounded directional movement signal:

```text
input: anonymous financial feature representation at event t
output: predicted_signal in [-1, 1]
```

The model must not output portfolio weights.

---

## 2. Model Input

Allowed inputs:

```text
anonymous feature values
missing flags
period offsets
numeric timing metadata
optional anonymous company ID
optional anonymous sector ID
```

Forbidden inputs:

```text
human-readable financial names
ticker
company name
raw concept_name
sector name
future prices
future filings
future target scale
manually engineered financial ratios in Version 1
```

---

## 3. Model Target

Primary target:

```text
target_normalized_signal = tanh(raw_log_return / (k * ex_ante_scale))
```

Range:

```text
[-1, 1]
```

---

## 4. Required Baseline Models

Advanced models are not valid until these baselines exist.

### 4.1 Constant Zero Model

Prediction:

```text
predicted_signal = 0
```

Purpose:

Checks whether portfolio construction accidentally creates performance without model signal.

---

### 4.2 Historical Mean Model

Prediction:

```text
predicted_signal = mean(target_normalized_signal in training data)
```

---

### 4.3 Ridge Regression

Input:

```text
dense anonymous feature matrix
```

Output:

```text
continuous prediction clipped to [-1, 1]
```

Config:

```yaml
ridge:
  alpha_grid: [0.1, 1.0, 10.0, 100.0]
```

---

### 4.4 Random Forest

Config:

```yaml
random_forest:
  n_estimators: 500
  max_depth: [5, 10, 20, null]
  min_samples_leaf: [20, 50, 100]
```

---

### 4.5 Gradient Boosted Trees

Preferred:

```text
LightGBM or XGBoost
```

Config:

```yaml
gbt:
  objective: regression
  n_estimators: 2000
  learning_rate: 0.02
  max_depth: [-1, 4, 6, 8]
  num_leaves: [31, 63, 127]
  min_data_in_leaf: [50, 100, 200]
  subsample: [0.7, 0.9, 1.0]
  colsample_bytree: [0.5, 0.8, 1.0]
  early_stopping_rounds: 100
```

---

## 5. Neural Baseline

### 5.1 MLP

Input:

```text
flattened tensor [periods_back * num_features * value/missing]
```

Architecture:

```text
Linear -> LayerNorm -> GELU -> Dropout
Linear -> LayerNorm -> GELU -> Dropout
Linear -> output
tanh output activation
```

Default:

```yaml
mlp:
  hidden_dims: [1024, 512, 128]
  dropout: 0.2
  batch_size: 512
  learning_rate: 0.0003
  weight_decay: 0.0001
  epochs: 100
  early_stopping_patience: 10
```

Loss:

```text
MSE(predicted_signal, target_normalized_signal)
```

Optional loss:

```text
Huber loss
```

---

## 6. Representation Learning Models

These models are closer to the intended research idea: the model learns structure in anonymous financial values.

---

### 6.1 Denoising Autoencoder

Input:

```text
anonymous financial tensor
```

Training objective:

```text
reconstruct masked or corrupted financial values
```

Then:

```text
latent representation -> prediction head -> predicted_signal
```

Architecture:

```text
encoder: MLP or transformer
latent_dim: 64 to 512
decoder: reconstruct values
prediction_head: predicts normalized signal
```

Loss:

```text
total_loss = reconstruction_loss + lambda_pred * prediction_loss
```

Default:

```yaml
autoencoder:
  latent_dim: 256
  mask_probability: 0.15
  reconstruction_loss_weight: 1.0
  prediction_loss_weight: 1.0
```

---

### 6.2 Masked Tabular Modeling

Randomly mask some anonymous financial fields and train model to reconstruct them.

Masking:

```text
mask 15% of present values
do not mask missing flags unless explicitly testing
```

Pretraining objective:

```text
predict masked scaled values
```

Fine-tuning objective:

```text
predict target_normalized_signal
```

---

### 6.3 Tabular Transformer

Input tokens:

```text
token = anonymous_feature_id embedding + period_offset embedding + value projection + missing flag embedding
```

Output:

```text
pooled representation -> prediction head
```

Important:

The feature ID embedding uses anonymous IDs only.

Architecture default:

```yaml
tabular_transformer:
  embedding_dim: 128
  num_layers: 4
  num_heads: 8
  dropout: 0.1
  max_tokens_per_sample: 8000
  pooling: attention_pooling
```

Because financial facts are sparse, long-format token representation may be more efficient than dense matrix.

---

## 7. Optional Conditional Diffusion Model

Diffusion is not required in Version 1. It should be tested only after baselines exist.

### 7.1 Purpose

The diffusion model generates possible future normalized movement outcomes conditioned on anonymous financial representation.

Input:

```text
condition = latent financial representation
noise = random Gaussian
```

Output:

```text
samples of target_normalized_signal or restored return distribution
```

---

### 7.2 Recommended Approach

Do not use diffusion directly on raw financial table in first attempt.

Use:

```text
anonymous features -> encoder -> latent condition
latent condition + diffusion noise -> target distribution samples
```

---

### 7.3 Diffusion Target

Option A:

```text
diffuse normalized target signal
```

Option B:

```text
diffuse standardized movement before tanh
```

Recommended Version 2:

```text
diffuse standardized_movement clipped to [-5, 5]
```

Then convert to signal:

```text
predicted_signal_sample = tanh(sampled_standardized_movement / k)
```

---

### 7.4 Distribution Output Metrics

For each company-event, generate `N` samples.

Default:

```yaml
num_generated_samples: 100
```

Compute:

```text
mean_signal
median_signal
probability_positive
probability_above_threshold
downside_quantile_05
upside_quantile_95
```

Portfolio module may use:

```text
mean_signal
probability_positive
downside risk
```

---

### 7.5 Diffusion Acceptance Rule

The diffusion model is only useful if it improves at least one of:

1. event-level rank correlation,
2. top-decile realized return,
3. portfolio Sharpe after costs,
4. max drawdown reduction,
5. calibration of predicted distribution,

without materially increasing turnover or instability.

If it does not beat LightGBM/XGBoost and autoencoder models, it should not be used.

---

## 8. Training Splits

Default:

```yaml
train:
  end_date: "2015-12-31"
validation:
  start_date: "2016-01-01"
  end_date: "2019-12-31"
test:
  start_date: "2020-01-01"
```

Split key:

```text
entry_date
```

Do not split randomly for final evaluation.

---

## 9. Walk-Forward Training

Final backtest should use walk-forward retraining.

Example:

```text
train from start to 2015-12-31 -> predict 2016Q1
train from start to 2016-03-31 -> predict 2016Q2
...
```

Default retrain frequency:

```yaml
walk_forward_retrain_frequency: quarterly
```

To reduce compute, Version 1 may retrain annually and disclose this.

---

## 10. Loss Functions

### 10.1 MSE

```text
loss = mean((predicted_signal - target)^2)
```

---

### 10.2 Huber

Recommended if target noise is high:

```yaml
huber_delta: 0.2
```

---

### 10.3 Rank-Aware Auxiliary Loss

Optional:

```text
loss_total = mse_loss + lambda_rank * pairwise_ranking_loss
```

Pairwise ranking uses same-date or same-month samples.

Default:

```yaml
use_rank_loss: false
```

Enable only after baseline works.

---

## 11. Prediction Output Schema

Each trained model must output:

```text
prediction_id
model_id
sample_id
internal_company_id
entry_date
exit_date
predicted_signal
prediction_created_at
feature_encoder_version
model_version
```

Optional fields:

```text
predicted_signal_std
probability_positive
mean_generated_signal
downside_quantile_05
upside_quantile_95
```

---

## 12. Prediction Clipping

Model outputs must be clipped:

```text
predicted_signal = clip(predicted_signal, -1, 1)
```

Portfolio restoration uses stricter clipping:

```text
clip to [-0.99, 0.99]
```

---

## 13. Evaluation Metrics at Model Level

Report on validation and test:

```text
MSE normalized target
MAE normalized target
R2 normalized target
direction accuracy
Spearman correlation predicted_signal vs target
Pearson correlation predicted_signal vs target
top decile average realized raw return
bottom decile average realized raw return
long-short spread diagnostic
calibration by predicted bucket
```

Even though portfolio is long-only, long-short spread is useful as diagnostic.

---

## 14. Prediction Bucket Report

Bucket by `predicted_signal`:

```text
[-1.0, -0.8)
[-0.8, -0.6)
[-0.6, -0.4)
[-0.4, -0.2)
[-0.2, 0.0)
[0.0, 0.2)
[0.2, 0.4)
[0.4, 0.6)
[0.6, 0.8)
[0.8, 1.0]
```

For each bucket report:

```text
count
mean predicted_signal
mean target
mean raw_log_return
median raw_log_return
positive raw return rate
mean benchmark excess return
```

---

## 15. Model Registry

Each model must have:

```text
model_id
model_type
training_start_date
training_end_date
validation_date_range
feature_encoder_version
target_config_hash
hyperparameters
random_seed
code_commit_hash
artifact_path
metrics
created_at
```

---

## 16. Reproducibility

Required seeds:

```yaml
random_seed: 42
numpy_seed: 42
torch_seed: 42
data_split_seed: 42
```

For GPU training, record:

```text
CUDA version
PyTorch version
GPU model
deterministic flag
```

---

## 17. Overfitting Controls

Required:

1. Time-based validation.
2. Early stopping.
3. Hyperparameter search only on validation.
4. Test set used only once for final reporting.
5. Compare to simple baselines.
6. Report turnover and transaction-cost performance.

---

## 18. Acceptance Criteria

Phase 5 is complete when:

1. Constant, Ridge, and tree baseline models train successfully.
2. Model inputs contain no human-readable financial names.
3. Predictions are in `[-1, 1]`.
4. Validation metrics are reported.
5. Prediction files are saved.
6. Feature encoder and target config are linked to model artifact.
7. At least one nontrivial model beats constant zero on validation MSE or rank correlation.
8. Portfolio backtest can consume prediction output.

---

## 19. Example Training Command

```bash
python -m src.models.train \
  --config configs/model_lgbm.yaml \
  --features data/model_inputs/features_dense.parquet \
  --targets data/processed/event_targets.parquet \
  --output artifacts/models/lgbm_v001
```

---

## 20. Example Prediction Row

```json
{
  "prediction_id": "PRED_abc123",
  "model_id": "lgbm_v001",
  "sample_id": "SAMPLE_7f3a2c",
  "internal_company_id": "COMP000001",
  "entry_date": "2020-05-04",
  "exit_date": "2020-07-30",
  "predicted_signal": 0.214,
  "prediction_created_at": "2026-05-20T00:00:00Z",
  "feature_encoder_version": "feature_encoder_v001",
  "model_version": "lgbm_v001"
}
```
