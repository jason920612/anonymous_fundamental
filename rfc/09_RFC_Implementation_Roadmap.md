# RFC-09: Implementation Roadmap

Status: Draft  
Phase: 9  
Purpose: Convert the RFC system into concrete implementation milestones, deliverables, tests, and Claude Code task prompts.

---

## 1. Implementation Principle

Implement from simplest reliable version to advanced models.

Do not start with diffusion.

Recommended order:

1. Data ingestion.
2. Event sample construction.
3. Target normalization.
4. Anonymous feature encoding.
5. Constant/Ridge/LightGBM baseline.
6. Simple inverse-vol portfolio.
7. Backtest and benchmark report.
8. Representation learning.
9. Full risk parity.
10. Optional diffusion model.

---

## 2. Milestone 1: Repository and Config Skeleton

### Deliverables

```text
pyproject.toml
README.md
configs/*.yaml
src/ package structure
tests/ directory
docs/rfc/ copied RFC files
```

### Acceptance Tests

```text
pytest runs
configs load successfully
project imports successfully
```

### Claude Code Prompt

```text
Create the repository skeleton for the anonymous fundamental portfolio research project.

Implement:
- pyproject.toml
- src package directories
- configs/default YAML files
- tests directory
- basic config loader
- README with project purpose

Do not implement model logic yet.
```

---

## 3. Milestone 2: Data Ingestion

### Deliverables

```text
src/data/sec_client.py
src/data/price_client.py
src/data/identifiers.py
src/data/trading_calendar.py
src/data/universe.py
data/raw/*
data/processed/companies.parquet
data/processed/filings.parquet
data/processed/financial_facts_long.parquet
data/processed/prices_daily.parquet
data/processed/benchmarks_daily.parquet
```

### Acceptance Tests

1. SEC company list downloads.
2. SEC submissions parse.
3. SEC company facts parse.
4. Price data is saved with adjusted close.
5. Trading calendar next/previous trading day works.

### Claude Code Prompt

```text
Implement Phase 1 data ingestion according to RFC-01.

Focus on:
- SEC company ticker ingestion
- SEC submissions parsing for 10-Q and 10-K
- SEC company facts normalization into long format
- price client interface
- trading calendar next/previous trading day functions
- parquet output
- data quality report

Add tests for CIK formatting, filing parsing, and trading calendar lookup.
```

---

## 4. Milestone 3: Event Sample Builder

### Deliverables

```text
src/features/sample_builder.py
src/targets/event_returns.py
data/processed/event_samples.parquet
data/processed/event_samples_excluded.parquet
```

### Acceptance Tests

1. `entry_date > report_event_date`.
2. `exit_date < next_report_event_date`.
3. `entry_date <= exit_date`.
4. adjusted prices exist.
5. excluded samples have reasons.

### Claude Code Prompt

```text
Implement event sample construction according to RFC-02.

For each company:
- sort 10-Q and 10-K filings by accepted datetime
- create one sample per filing with a next filing
- define entry_date as first trading day after report_event_date
- define exit_date as trading day before next report_event_date
- calculate entry_price, exit_price, raw_log_return, raw_simple_return
- align SPY and QQQ benchmark returns
- write valid and excluded sample parquet files

Add tests for date ordering and missing price exclusion.
```

---

## 5. Milestone 4: Target Normalization

### Deliverables

```text
src/targets/volatility_scale.py
src/targets/target_transform.py
data/processed/event_targets.parquet
```

### Acceptance Tests

1. ex-ante scale uses only dates before entry date.
2. target is always in `[-1, 1]`.
3. atanh restoration works.
4. missing scale samples are excluded or fallback is used.

### Claude Code Prompt

```text
Implement target normalization according to RFC-04.

Implement:
- trailing daily volatility scaled to holding period
- prior event return volatility
- hybrid scale
- tanh target transform
- atanh inverse restoration
- leakage tests ensuring no future prices are used

Default:
k = 2.5
daily lookback = 252
event lookback = 8
scale method = hybrid_daily_event
```

---

## 6. Milestone 5: Anonymous Feature Encoder

### Deliverables

```text
src/features/anonymous_mapping.py
src/features/raw_fact_matrix.py
src/features/normalization.py
artifacts/feature_encoder/feature_encoder_v001/*
data/model_inputs/features_dense.parquet
data/model_inputs/train.pt optional
```

### Acceptance Tests

1. No model input column contains human financial words.
2. Anonymous feature IDs are stable.
3. scaler fit uses train only.
4. missing flags exist.
5. unseen validation/test features do not alter mapping.

### Claude Code Prompt

```text
Implement anonymous feature encoding according to RFC-03.

Requirements:
- map taxonomy/concept/unit to stable anonymous_feature_id
- do not expose human-readable concept names in model inputs
- encode current and prior 8 periods
- signed log transform values
- robust scale using training data only
- include missing flags
- include numeric timing metadata
- write dense parquet model input

Add tests:
- forbidden financial words not present in model input columns
- one feature ID maps to one raw concept internally
- train-only scaler statistics
```

---

## 7. Milestone 6: Baseline Models

### Deliverables

```text
src/models/datasets.py
src/models/baselines.py
src/models/train.py
src/models/predict.py
artifacts/models/*
artifacts/predictions/*
```

### Required Models

```text
constant zero
historical mean
ridge regression
random forest
LightGBM or XGBoost
simple MLP optional
```

### Acceptance Tests

1. Models train.
2. Predictions are clipped to `[-1, 1]`.
3. Prediction files match schema.
4. Validation metrics are saved.
5. Baseline comparison exists.

### Claude Code Prompt

```text
Implement baseline model training according to RFC-05.

Start with:
- constant zero model
- historical mean model
- Ridge regression
- LightGBM if available, otherwise XGBoost or sklearn HistGradientBoosting

Use time-based train/validation/test split.
Target is target_normalized_signal.
Save predictions with sample_id, model_id, predicted_signal, entry_date, exit_date.
Clip predictions to [-1, 1].
Generate validation metrics including MSE, MAE, direction accuracy, and Spearman correlation.
```

---

## 8. Milestone 7: Simple Portfolio Construction

### Deliverables

```text
src/portfolio/signal_restore.py
src/portfolio/candidate_selection.py
src/portfolio/risk_models.py
src/portfolio/allocation.py
data/backtest_outputs/portfolio_weights.parquet
```

### First Version Allocation

Use signal-weighted inverse volatility:

```text
positive_signal = max(predicted_signal, 0)
risk_budget_raw = positive_signal ^ p
raw_weight = risk_budget / volatility
normalize
apply max weight cap
allow cash
```

### Acceptance Tests

1. Negative signals are not bought.
2. weights sum to <= 1.
3. single stock cap enforced.
4. cash row exists if not fully invested.
5. transaction cost inputs are prepared.

### Claude Code Prompt

```text
Implement simple portfolio construction according to RFC-06.

Use:
- predicted_signal
- ex_ante_scale
- atanh restoration
- long-only positive signals
- top N candidate selection
- signal-weighted inverse-vol allocation
- max single stock weight
- optional cash

Do not implement full covariance risk parity yet.
Add tests for weight constraints, negative signal exclusion, and cash handling.
```

---

## 9. Milestone 8: Backtest Engine

### Deliverables

```text
src/backtest/engine.py
src/backtest/ledger.py
src/backtest/metrics.py
src/backtest/benchmarks.py
reports/backtest/*
```

### Acceptance Tests

1. Daily returns generated.
2. transaction costs deducted.
3. equity curve generated.
4. SPY and QQQ benchmark comparison generated.
5. max drawdown calculation correct.
6. turnover calculation correct.

### Claude Code Prompt

```text
Implement the event-driven backtest engine according to RFC-07.

Requirements:
- simulate daily portfolio returns
- handle weight drift
- rebalance when signals become valid or expire
- apply transaction costs
- track holdings, trades, and daily returns
- compare against SPY, QQQ, equal-weight universe, and equal-risk universe
- output metrics, CSVs, and charts

Add tests for portfolio accounting, drawdown, and turnover calculations.
```

---

## 10. Milestone 9: Experiment Registry

### Deliverables

```text
src/experiments/registry.py
src/experiments/logging.py
src/experiments/validation.py
artifacts/experiments/*
```

### Acceptance Tests

1. resolved config saved.
2. experiment metadata saved.
3. config hash stored.
4. validation gates run.
5. leakage checklist appears in report.

### Claude Code Prompt

```text
Implement experiment management according to RFC-08.

Requirements:
- experiment ID generation
- resolved config saving
- metadata.json
- config hash
- experiment registry table or parquet
- validation gates
- leakage checklist
- report template

Every pipeline run must be reproducible from saved config and metadata.
```

---

## 11. Milestone 10: Representation Learning

### Deliverables

```text
src/models/autoencoder.py
src/models/tabular_transformer.py
artifacts/models/autoencoder_v001
artifacts/models/transformer_v001
```

### Acceptance Tests

1. Autoencoder pretraining runs.
2. latent representation is saved.
3. prediction head trains.
4. validation metrics compare against LightGBM.
5. no human field names enter model.

### Claude Code Prompt

```text
Implement representation learning models according to RFC-05.

Start with denoising autoencoder:
- input anonymous feature tensor
- randomly mask present values
- reconstruct masked values
- use latent representation to predict target_normalized_signal

Save latent representations and predictions.
Compare against Ridge and LightGBM baselines.
```

---

## 12. Milestone 11: Full Risk Parity

### Deliverables

```text
src/portfolio/risk_parity_optimizer.py
```

### Acceptance Tests

1. covariance matrix uses only past returns.
2. optimizer solves for weights.
3. risk contributions approximate target risk budgets.
4. fallback to inverse-vol works.
5. constraints hold.

### Claude Code Prompt

```text
Implement full signal-weighted risk parity according to RFC-06.

Requirements:
- estimate trailing covariance using only past returns
- shrink covariance toward diagonal
- convert positive signals to risk budgets
- solve for long-only weights whose risk contributions match risk budgets
- enforce max single-stock weight and optional sector caps
- fallback to inverse-vol if optimization fails

Add tests for risk contribution accuracy and fallback behavior.
```

---

## 13. Milestone 12: Optional Diffusion Research

### Deliverables

```text
src/models/diffusion.py
artifacts/models/diffusion_v001
artifacts/predictions/diffusion_distribution_v001.parquet
```

### Acceptance Tests

1. diffusion model generates distribution samples.
2. generated samples are converted to predicted signals.
3. distribution metrics are saved.
4. performance is compared against LightGBM and autoencoder.
5. diffusion is rejected if it does not improve evidence.

### Claude Code Prompt

```text
Implement optional conditional diffusion research according to RFC-05.

Do not replace the baseline model.
Use anonymous financial latent representation as condition.
Generate a distribution of future standardized movements or normalized signals.
Save mean signal, probability positive, downside quantile, and upside quantile.
Compare portfolio results against LightGBM and autoencoder.
```

---

## 14. Recommended Development Order

```text
Week 1:
  repo skeleton
  data ingestion prototype
  trading calendar
  sample builder

Week 2:
  target normalization
  anonymous feature encoder
  leakage tests

Week 3:
  baseline models
  simple portfolio construction
  first backtest

Week 4:
  reports
  experiment registry
  robustness tests

Week 5+:
  representation learning
  full risk parity
  optional diffusion
```

Do not treat this timeline as a promise. It is an implementation planning estimate.

---

## 15. Minimum Useful Version

The minimum useful version must include:

1. 1,000+ companies if data availability permits.
2. 10-Q/10-K event samples.
3. anonymous feature encoding.
4. normalized target.
5. Ridge and LightGBM or equivalent.
6. signal-weighted inverse-vol portfolio.
7. SPY/QQQ comparison.
8. transaction costs.
9. leakage tests.
10. written report.

---

## 16. Final Research Decision Tree

After first full backtest:

### Case A: No model beats simple baselines

Decision:

```text
Stop advanced modeling.
Improve data quality or target definition first.
```

### Case B: Tree model beats baselines but neural models do not

Decision:

```text
Use tree model as main research baseline.
Do not use diffusion yet.
```

### Case C: Representation learning improves event ranking

Decision:

```text
Study autoencoder or transformer further.
Try full risk parity.
```

### Case D: Diffusion improves distribution calibration and portfolio risk

Decision:

```text
Keep diffusion as research candidate.
Do not deploy without more robustness testing.
```

---

## 17. Done Definition

The whole RFC implementation is done when:

1. Data pipeline is reproducible.
2. Feature encoding is anonymous.
3. Target construction is leak-free.
4. Baseline models train and predict.
5. Portfolio module constructs valid long-only weights.
6. Backtest compares against benchmarks.
7. Reports include risk and drawdown.
8. Experiment registry tracks all runs.
9. Tests cover major leakage and accounting errors.
10. Results are honest about limitations.
