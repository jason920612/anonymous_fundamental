# RFC-08: Experiment Management

Status: Draft  
Phase: 8  
Purpose: Define reproducibility, configuration, experiment registry, logging, validation, and governance for the research system.

---

## 1. Objective

Every experiment must be reproducible.

Given:

```text
data version
feature encoder version
target config
model config
portfolio config
backtest config
code commit
random seed
```

The system should be able to reproduce the same outputs or explain why exact reproduction is not possible.

---

## 2. Experiment ID

Experiment ID format:

```text
{date}_{model_type}_{feature_version}_{target_version}_{portfolio_method}_{short_hash}
```

Example:

```text
20260520_lgbm_featv001_tgtv001_invvol_a1b2c3
```

---

## 3. Config Files

Required config files:

```text
configs/data.yaml
configs/features.yaml
configs/target.yaml
configs/model.yaml
configs/portfolio.yaml
configs/backtest.yaml
configs/experiment.yaml
```

A full experiment config should merge all of them into:

```text
artifacts/experiments/{experiment_id}/resolved_config.yaml
```

---

## 4. Config Hash

Compute:

```text
config_hash = sha256(canonical_json(resolved_config))
```

Store in every major output artifact.

---

## 5. Code Version

Store:

```text
git_commit_hash
git_branch
is_dirty_worktree
```

If worktree is dirty, experiment may run but must be marked:

```text
reproducibility_status = dirty_code
```

---

## 6. Data Version

Each processed dataset must have:

```text
data_version
raw_data_manifest_hash
created_at
source_date_range
row_counts
quality_report_path
```

Example:

```json
{
  "data_version": "data_v001",
  "raw_data_manifest_hash": "sha256...",
  "companies": 4210,
  "event_samples": 84522,
  "valid_training_samples": 71230,
  "created_at": "2026-05-20T00:00:00Z"
}
```

---

## 7. Artifact Directory

For each experiment:

```text
artifacts/experiments/{experiment_id}/
  resolved_config.yaml
  metadata.json
  feature_encoder/
  model/
  predictions/
  portfolio_weights/
  backtest/
  reports/
  logs/
```

---

## 8. Metadata File

Required:

```json
{
  "experiment_id": "20260520_lgbm_featv001_tgtv001_invvol_a1b2c3",
  "created_at": "2026-05-20T00:00:00Z",
  "owner": "jason",
  "purpose": "baseline lgbm on anonymous raw financial features",
  "data_version": "data_v001",
  "feature_encoder_version": "feature_encoder_v001",
  "target_version": "target_v001",
  "model_id": "lgbm_v001",
  "portfolio_id": "lgbm_v001_invvol",
  "config_hash": "sha256...",
  "git_commit_hash": "abc123",
  "random_seed": 42
}
```

---

## 9. Logging Requirements

Every pipeline step logs:

```text
timestamp
level
module
run_id
message
key_value_context
```

Example:

```json
{
  "timestamp": "2026-05-20T12:01:02Z",
  "level": "INFO",
  "module": "features.encoder",
  "run_id": "feature_encoder_v001",
  "message": "finished fitting encoder",
  "num_features": 1843,
  "num_samples": 50321
}
```

---

## 10. Validation Gates

The pipeline must stop if any required validation fails.

Required gates:

1. raw data completeness gate,
2. event date ordering gate,
3. no future filing gate,
4. no future price in scale gate,
5. anonymous feature name gate,
6. target range gate,
7. model prediction range gate,
8. portfolio weight constraint gate,
9. backtest accounting gate.

---

## 11. Data Leakage Checklist

Before any result is accepted, answer:

```text
Did the model see any future financial filing? no
Did the model see any future price? no
Was target scale computed using future returns? no
Were scaler statistics fit only on training data? yes
Were hyperparameters selected on test data? no
Was universe membership point-in-time or disclosed as imperfect? yes/disclosed
Were restatements controlled or disclosed? yes/disclosed
```

The report must include this checklist.

---

## 12. Experiment Registry Table

```sql
CREATE TABLE experiment_registry (
    experiment_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    owner TEXT,
    purpose TEXT,
    data_version TEXT,
    feature_encoder_version TEXT,
    target_version TEXT,
    model_id TEXT,
    portfolio_id TEXT,
    backtest_id TEXT,
    config_hash TEXT,
    git_commit_hash TEXT,
    random_seed INTEGER,
    status TEXT,
    summary_metric_json TEXT,
    artifact_path TEXT
);
```

Allowed statuses:

```text
created
running
failed
completed
rejected
promoted
archived
```

---

## 13. Model Promotion Rule

A model can be promoted from research candidate to deeper study only if:

1. It beats required baselines on validation.
2. It does not fail leakage checks.
3. It has positive out-of-sample evidence.
4. It does not rely on unrealistic transaction assumptions.
5. It has acceptable turnover.
6. Its improvement is not explained solely by sector concentration.
7. Its report is reproducible.

---

## 14. Required Baseline Comparison Report

Each experiment report must compare against:

```text
constant zero model
ridge model
tree model
equal-weight universe
equal-risk universe
SPY
QQQ
```

Do not report only the best model.

---

## 15. Hyperparameter Search

Allowed search methods:

```text
grid search
random search
Bayesian search
```

Rules:

1. Use validation set only.
2. Do not tune on test set.
3. Save all trials.
4. Report best trial and median trial.
5. Keep failed trials.

Trial table:

```sql
CREATE TABLE hyperparameter_trials (
    experiment_id TEXT,
    trial_id TEXT,
    model_type TEXT,
    parameters_json TEXT,
    validation_metrics_json TEXT,
    status TEXT,
    artifact_path TEXT,
    created_at TIMESTAMP
);
```

---

## 16. Random Seeds

Default:

```yaml
random_seed: 42
numpy_seed: 42
torch_seed: 42
lightgbm_seed: 42
xgboost_seed: 42
```

Each trial may vary seed, but must store it.

---

## 17. Report Template

Each experiment report must include:

```text
1. Purpose
2. Data version
3. Feature version
4. Target definition
5. Model details
6. Portfolio construction
7. Backtest period
8. Performance metrics
9. Benchmark comparison
10. Drawdown analysis
11. Turnover and cost analysis
12. Prediction diagnostics
13. Robustness tests
14. Leakage checklist
15. Known limitations
16. Decision: reject / keep studying / promote
```

---

## 18. Known Limitations Section

Every report must explicitly state:

1. Whether delisted stocks are fully included.
2. Whether historical constituents are point-in-time.
3. Whether SEC filing date or earnings announcement date is used.
4. Whether restatements are controlled.
5. Whether price source is adjusted correctly.
6. Whether transaction costs are realistic.
7. Whether liquidity constraints are sufficient.

---

## 19. Acceptance Criteria

Phase 8 is complete when:

1. Experiment registry exists.
2. Every run saves resolved config.
3. Every artifact links to data/model/config versions.
4. Validation gates run.
5. Leakage checklist is included in reports.
6. Failed runs are logged.
7. Hyperparameter trials are stored.
8. Reports are reproducible from saved configs.

---

## 20. Example Experiment Command

```bash
python -m src.experiments.run \
  --experiment-config configs/experiment_lgbm_invvol.yaml \
  --output artifacts/experiments
```

---

## 21. Example Resolved Experiment Config

```yaml
experiment:
  owner: jason
  purpose: "baseline anonymous financial feature model"
  random_seed: 42

data:
  version: data_v001

features:
  version: feature_encoder_v001
  periods_back: 8
  anonymize: true
  use_human_ratios: false

target:
  version: target_v001
  transform: tanh
  k: 2.5
  scale_method: hybrid_daily_event

model:
  type: lightgbm
  objective: regression
  n_estimators: 2000
  learning_rate: 0.02

portfolio:
  method: signal_weighted_inverse_vol
  max_single_stock_weight: 0.05
  target_positions: 50
  transaction_cost_bps_per_trade: 10

backtest:
  start_date: "2020-01-01"
  end_date: "2025-12-31"
  benchmarks: ["SPY", "QQQ"]
```
