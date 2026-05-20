"""Shared CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from afp.utils.config import load_config
from afp.utils.io import read_parquet, write_parquet
from afp.utils.logging import get_logger

log = get_logger("afp.cli")


def resolve_config(config_path: str | None) -> dict:
    path = Path(config_path) if config_path else Path("configs/default.yaml")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return load_config(path)


def require_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"required input missing: {p}")
    return read_parquet(p)


__all__ = ["resolve_config", "require_parquet", "read_parquet", "write_parquet", "log"]
