"""Phase 1 smoke checks: package import, config load, deep merge semantics."""

from pathlib import Path

import afp
from afp.utils.config import config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_package_importable():
    assert afp.__version__


def test_default_config_loads_and_merges_includes():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    # Each included phase config contributes a top-level key.
    for key in ("project", "data", "features", "target", "model", "portfolio", "backtest"):
        assert key in cfg, f"missing top-level key {key} in merged config"
    assert cfg["project"]["random_seed"] == 42
    assert cfg["target"]["k"] == 2.5
    assert cfg["portfolio"]["positions"]["max_single_stock_weight"] == 0.05


def test_config_hash_is_stable():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert config_hash(cfg) == config_hash(cfg)
    assert len(config_hash(cfg)) == 64
