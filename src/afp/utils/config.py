"""Config loading: merges YAML files with an `includes:` directive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, resolving an optional `includes:` list relative to its directory."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    includes = cfg.pop("includes", []) or []
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = path.parent / inc
        merged = _deep_merge(merged, load_config(inc_path))
    merged = _deep_merge(merged, cfg)
    return merged


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable sha256 over canonical JSON of a config dict."""
    canonical = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
