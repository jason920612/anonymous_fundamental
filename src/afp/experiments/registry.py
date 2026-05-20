"""Experiment registry (RFC-08 §2-§7, §12)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from afp.utils.config import config_hash


def _git_state(repo_root: Path) -> dict[str, str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                      stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                         cwd=repo_root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=repo_root, stderr=subprocess.DEVNULL).strip())
        return {"git_commit_hash": sha, "git_branch": branch, "is_dirty_worktree": str(dirty)}
    except Exception:
        return {"git_commit_hash": "unknown", "git_branch": "unknown", "is_dirty_worktree": "unknown"}


def make_experiment_id(*, owner: str, model_type: str, feature_version: str,
                       target_version: str, portfolio_method: str, config_h: str) -> str:
    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = config_h[:8]
    return f"{date_str}_{owner}_{model_type}_{feature_version}_{target_version}_{portfolio_method}_{short}"


class ExperimentRun:
    def __init__(self, base_dir: Path, experiment_id: str):
        self.dir = Path(base_dir) / experiment_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id

    # ---- save artifacts ----

    def save_resolved_config(self, cfg: dict[str, Any]) -> Path:
        path = self.dir / "resolved_config.yaml"
        path.write_text(__import__("yaml").safe_dump(cfg, sort_keys=True))
        return path

    def save_metadata(self, **fields) -> Path:
        path = self.dir / "metadata.json"
        path.write_text(json.dumps(fields, indent=2, default=str))
        return path

    def save_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.dir / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2, default=str))
        return path

    def save_table(self, df: pd.DataFrame, name: str) -> Path:
        path = self.dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def save_report(self, body_md: str) -> Path:
        path = self.dir / "report.md"
        path.write_text(body_md)
        return path


def append_registry(registry_csv: Path, row: dict[str, Any]) -> None:
    """Append an experiment row to a CSV-backed registry (RFC-08 §12)."""
    registry_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if registry_csv.exists():
        df_old = pd.read_csv(registry_csv)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(registry_csv, index=False)


def build_experiment_metadata(
    cfg: dict[str, Any],
    *,
    owner: str,
    purpose: str,
    model_type: str,
    feature_version: str,
    target_version: str,
    portfolio_method: str,
    repo_root: Path,
) -> tuple[str, dict[str, Any]]:
    h = config_hash(cfg)
    exp_id = make_experiment_id(owner=owner, model_type=model_type,
                                feature_version=feature_version,
                                target_version=target_version,
                                portfolio_method=portfolio_method,
                                config_h=h)
    meta = {
        "experiment_id": exp_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "owner": owner,
        "purpose": purpose,
        "model_type": model_type,
        "feature_encoder_version": feature_version,
        "target_version": target_version,
        "portfolio_method": portfolio_method,
        "config_hash": h,
        "random_seed": cfg.get("project", {}).get("random_seed"),
        **_git_state(repo_root),
    }
    return exp_id, meta
