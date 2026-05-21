"""Phase 68c: daemon helper tests (no network, no long loop)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from afp.cli import daemon


def test_load_state_returns_dict_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "STATE_PATH", tmp_path / "no.json")
    assert daemon._load_state() == {}


def test_save_and_load_state_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    monkeypatch.setattr(daemon, "STATE_PATH", p)
    daemon._save_state({"a": 1, "b": "two"})
    out = daemon._load_state()
    assert out == {"a": 1, "b": "two"}


def test_latest_filing_dates_no_cache_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False, raising=False)
    # Simpler: just call with CIKs that have no cache (we don't actually patch Path)
    out = daemon._latest_filing_dates(["9999999999"])
    assert out["9999999999"] is None


def test_detect_new_filings_returns_empty_with_no_data():
    sp500 = pd.DataFrame({"cik": [], "ticker": []})
    state = {}
    out = daemon._detect_new_filings(sp500, state)
    assert out == []


def test_build_digest_message_includes_keys():
    cycle_info = {
        "cycle_started": "2026-05-21T00:00:00Z",
        "sp500_size": 503,
        "new_filings": [{"ticker": "AAPL", "cik": "320193"}],
        "retrained": True,
        "model_meta": {"best_iteration": 250},
        "top_picks": [{"ticker": "AAPL", "percentile": 92, "expected_return": 0.05}],
        "next_cycle_hours": 24.0,
    }
    msg = daemon._build_digest_message(cycle_info)
    assert "afp-daemon digest" in msg
    assert "503" in msg
    assert "AAPL" in msg
    assert "retrained" in msg.lower()
    assert "best_iter=250" in msg


def test_build_digest_no_retrain():
    cycle_info = {
        "cycle_started": "2026-05-21T00:00:00Z",
        "sp500_size": 503,
        "new_filings": [],
        "retrained": False,
        "model_meta": {},
        "top_picks": [],
        "next_cycle_hours": 24.0,
    }
    msg = daemon._build_digest_message(cycle_info)
    assert "not retrained" in msg.lower()
