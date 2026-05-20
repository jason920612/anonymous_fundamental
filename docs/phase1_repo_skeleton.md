# Phase 1 — Repository Skeleton

Maps to **RFC-00 §8** (Repository Structure) and **RFC-09 Milestone 1**.

## Scope

Phase 1 establishes the project shape so all later phases plug in deterministically:

- `pyproject.toml` (PEP 621) with pinned-floor dependencies and dev extras.
- `src/afp/` package laid out by phase (`data`, `features`, `targets`, `models`,
  `portfolio`, `backtest`, `experiments`, `utils`).
- `configs/` with one YAML per phase plus a `default.yaml` that lists `includes:`.
- `data/`, `artifacts/`, `reports/` directories (kept empty in git but pre-created
  so ingestion and backtest writers never need to mkdir from inside business code).
- `tests/` for the pytest suite.

## Config Loader (`afp.utils.config`)

```python
from afp.utils.config import load_config, config_hash
cfg = load_config("configs/default.yaml")    # merges includes recursively
h   = config_hash(cfg)                        # deterministic sha256 over canonical JSON
```

Rules:

- `includes:` is processed first (left-to-right) so later includes override earlier ones.
- The root file's keys override all includes.
- `_deep_merge` walks nested dicts; lists are replaced (not concatenated) to keep
  override semantics predictable.
- `config_hash` is the canonical fingerprint stored on every artifact — see RFC-08 §4.

## Logging (`afp.utils.logging`)

`get_logger(name)` returns a stdlib `Logger` whose handler emits **one JSON line
per record**: `timestamp / level / module / message / …extras`. All ingestion,
training, backtest, and experiment-registry components route through this so logs
are grep-friendly and trivially shippable.

## IO & dates

- `afp.utils.io.{read,write}_parquet` standardize on `pandas.to_parquet` (pyarrow
  backend). Output paths are auto-created.
- `afp.utils.dates.to_eastern_date` converts SEC `acceptedDate` values to a
  US/Eastern calendar date — RFC-00 §5.2 requires Eastern day boundaries.

## Acceptance

- `pytest tests/test_skeleton.py` passes (config loads + package importable).
- `python -c "import afp"` exits 0.
- `python -c "from afp.utils.config import load_config; load_config('configs/default.yaml')"` works.

## Outputs to Downstream Phases

| Consumer | Reads |
|---|---|
| Phase 2 ingestion | `data.*` (sec, prices, benchmarks, calendar, universe) |
| Phase 3 sample builder | `data.forms`, `data.universe` |
| Phase 4 targets | `target.*` |
| Phase 5 features | `features.*` |
| Phase 6 models | `model.*` + `features.forbidden_model_input_substrings` |
| Phase 7 portfolio | `portfolio.*`, `target.k` |
| Phase 8 backtest | `backtest.*`, `portfolio.costs` |
| Phase 9 experiments | the whole resolved dict + its hash |
