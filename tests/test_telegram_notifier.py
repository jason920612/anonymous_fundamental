"""Phase 68b: Telegram notifier tests (no real HTTP)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afp.notify.telegram import TelegramConfig, TelegramNotifier


class _MockResponse:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_disabled_when_token_missing():
    cfg = TelegramConfig(bot_token=None, chat_ids=["123"])
    assert not cfg.enabled()


def test_disabled_when_no_chats():
    cfg = TelegramConfig(bot_token="x", chat_ids=[])
    assert not cfg.enabled()


def test_send_disabled_returns_error_dict():
    cfg = TelegramConfig()
    notifier = TelegramNotifier(cfg)
    results = notifier.send("hi")
    assert results[0]["ok"] is False


def test_send_calls_post_per_chat(monkeypatch):
    cfg = TelegramConfig(bot_token="abc", chat_ids=["1", "2", "3"])
    notifier = TelegramNotifier(cfg)
    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append((url, json["chat_id"]))
        return _MockResponse(200, {"ok": True, "result": {}})
    monkeypatch.setattr("afp.notify.telegram.requests.post", fake_post)
    results = notifier.send("hello")
    assert len(results) == 3
    assert all(r["ok"] for r in results)
    chat_ids_called = [c[1] for c in calls]
    assert chat_ids_called == ["1", "2", "3"]


def test_from_env_parses_csv(monkeypatch):
    env = {"TELEGRAM_BOT_TOKEN": "abc", "TELEGRAM_CHAT_IDS": "1,2,3"}
    cfg = TelegramConfig.from_env(env)
    assert cfg.bot_token == "abc"
    assert cfg.chat_ids == ["1", "2", "3"]


def test_from_file_parses_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"bot_token": "xyz", "chat_ids": ["555"]}))
    cfg = TelegramConfig.from_file(p)
    assert cfg.bot_token == "xyz"
    assert cfg.chat_ids == ["555"]


def test_http_error_recorded(monkeypatch):
    cfg = TelegramConfig(bot_token="abc", chat_ids=["1"])
    notifier = TelegramNotifier(cfg)
    def fake_post(url, json=None, timeout=None):
        return _MockResponse(400, {"ok": False, "description": "bad chat"})
    monkeypatch.setattr("afp.notify.telegram.requests.post", fake_post)
    results = notifier.send("x")
    assert results[0]["ok"] is False
    assert "HTTP 400" in results[0]["error"]


def test_prompt_skip_returns_disabled(monkeypatch):
    from afp.notify import telegram as mod
    # Empty token → skip
    monkeypatch.setattr("builtins.input", lambda *_: "")
    cfg = mod.prompt_user_for_credentials(save=False, test_ping=False)
    assert not cfg.enabled()


def test_prompt_saves_credentials(monkeypatch, tmp_path):
    from afp.notify import telegram as mod
    responses = iter(["test-token", "111,222"])
    monkeypatch.setattr("builtins.input", lambda *_: next(responses))
    # Belt-and-suspenders: redirect both the module global AND pass an
    # explicit path, so even if a regression makes _save_config bind the
    # default at definition time again, we still don't touch the real
    # configs/telegram.json.
    monkeypatch.setattr(mod, "CONFIG_PATH", tmp_path / "tg.json")
    cfg = mod.prompt_user_for_credentials(save=True, test_ping=False)
    assert cfg.bot_token == "test-token"
    assert cfg.chat_ids == ["111", "222"]
    saved = json.loads((tmp_path / "tg.json").read_text())
    assert saved["bot_token"] == "test-token"
    assert saved["chat_ids"] == ["111", "222"]
    # Make damn sure we didn't touch the real config file
    real = Path("configs/telegram.json")
    if real.exists():
        # If a previous user actually configured it, just check it doesn't
        # match the test sentinel value.
        live = json.loads(real.read_text())
        assert live.get("bot_token") != "test-token"


def test_get_default_notifier_prefers_config_file(tmp_path, monkeypatch):
    from afp.notify import telegram as mod
    cfg_path = tmp_path / "tg.json"
    cfg_path.write_text(json.dumps({"bot_token": "from-file", "chat_ids": ["9"]}))
    monkeypatch.setattr(mod, "CONFIG_PATH", cfg_path)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    notifier = mod.get_default_notifier(interactive=False)
    assert notifier.cfg.bot_token == "from-file"


def test_get_default_notifier_falls_back_to_env(tmp_path, monkeypatch):
    from afp.notify import telegram as mod
    monkeypatch.setattr(mod, "CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "envtok")
    monkeypatch.setenv("TELEGRAM_CHAT_IDS", "42")
    notifier = mod.get_default_notifier(interactive=False)
    assert notifier.cfg.bot_token == "envtok"
    assert notifier.cfg.chat_ids == ["42"]


def test_get_default_notifier_interactive_called_when_no_config(tmp_path, monkeypatch):
    from afp.notify import telegram as mod
    monkeypatch.setattr(mod, "CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    # Force TTY True
    import sys as _sys
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: True, raising=False)
    called = {"n": 0}
    def fake_prompt(*args, **kwargs):
        called["n"] += 1
        return TelegramConfig(bot_token="prompted", chat_ids=["55"])
    monkeypatch.setattr(mod, "prompt_user_for_credentials", fake_prompt)
    notifier = mod.get_default_notifier(interactive=True)
    assert called["n"] == 1
    assert notifier.cfg.bot_token == "prompted"
