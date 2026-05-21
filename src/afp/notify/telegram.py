"""Telegram bot notifier.

Sends notifications to Telegram chats via the public Bot API. Uses
only `requests`; no heavy `python-telegram-bot` dependency.

Setup steps for users:

  1. Create a bot via @BotFather in Telegram, get a token of the form
     `1234567890:AAH...`.
  2. Subscribe by sending any message to the bot from your account.
  3. Get your chat_id: open
     `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy the
     numeric `chat.id` field from the latest message.
  4. Export environment variables:
        TELEGRAM_BOT_TOKEN=<your-token>
        TELEGRAM_CHAT_IDS=<id1>,<id2>,...
     Or set them in a config file the daemon reads.

A single bot can broadcast to multiple subscribed chats. The daemon
reads `TELEGRAM_CHAT_IDS` (comma-separated) and posts to each.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests


@dataclass
class TelegramConfig:
    bot_token: str | None = None      # env: TELEGRAM_BOT_TOKEN
    chat_ids: list[str] = field(default_factory=list)  # env: TELEGRAM_CHAT_IDS (csv)
    timeout_sec: float = 10.0
    api_base: str = "https://api.telegram.org"

    @classmethod
    def from_env(cls, env: dict | None = None) -> "TelegramConfig":
        env = env if env is not None else os.environ
        chat_csv = env.get("TELEGRAM_CHAT_IDS", "")
        chat_ids = [s.strip() for s in chat_csv.split(",") if s.strip()]
        return cls(bot_token=env.get("TELEGRAM_BOT_TOKEN"), chat_ids=chat_ids)

    @classmethod
    def from_file(cls, path: str | Path) -> "TelegramConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(
            bot_token=data.get("bot_token"),
            chat_ids=list(data.get("chat_ids") or []),
        )

    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_ids)


class TelegramNotifier:
    """Thin client that posts messages to one or more Telegram chats."""

    def __init__(self, cfg: TelegramConfig):
        self.cfg = cfg

    def send(self, message: str, parse_mode: str = "Markdown") -> list[dict]:
        """Send `message` to every configured chat_id. Returns per-chat
        response dicts (`ok=True` or `ok=False, error=...`)."""
        if not self.cfg.enabled():
            return [{"ok": False, "error": "telegram not configured (bot_token/chat_ids missing)"}]
        url = f"{self.cfg.api_base}/bot{self.cfg.bot_token}/sendMessage"
        results = []
        for chat_id in self.cfg.chat_ids:
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            try:
                resp = requests.post(url, json=payload, timeout=self.cfg.timeout_sec)
                if resp.status_code == 200 and resp.json().get("ok"):
                    results.append({"ok": True, "chat_id": chat_id})
                else:
                    results.append({"ok": False, "chat_id": chat_id,
                                     "error": f"HTTP {resp.status_code}: {resp.text[:200]}"})
            except requests.RequestException as exc:
                results.append({"ok": False, "chat_id": chat_id, "error": str(exc)})
            # be polite — Telegram rate-limits at ~30 msg/sec across all chats
            time.sleep(0.05)
        return results

    def test_ping(self) -> bool:
        """Send a one-line ping; useful for verifying setup."""
        results = self.send("✅ afp-daemon ping — telegram bot is wired up.")
        return all(r.get("ok") for r in results)


CONFIG_PATH = Path("configs/telegram.json")


def _save_config(cfg: TelegramConfig, path: Path | None = None) -> None:
    # Resolve `CONFIG_PATH` dynamically so monkeypatching in tests works.
    path = path if path is not None else CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "bot_token": cfg.bot_token,
        "chat_ids": cfg.chat_ids,
    }, indent=2))


def prompt_user_for_credentials(save: bool = True,
                                 test_ping: bool = True) -> TelegramConfig:
    """Interactive setup — ask the user for bot token + chat IDs.

    Returns an empty (disabled) config if the user presses Enter to skip.
    Otherwise (optionally) writes the credentials to `configs/telegram.json`
    so subsequent runs don't re-prompt.
    """
    print("\n" + "=" * 60)
    print("  Telegram bot setup")
    print("=" * 60)
    print("Press Enter at any prompt to skip Telegram notifications.")
    print()
    print("To create a bot:  message @BotFather on Telegram, follow the prompts,")
    print("                  and copy the token of the form 1234567890:AAH...")
    print("To get a chat_id: message your bot from your personal Telegram account,")
    print("                  then visit")
    print("                  https://api.telegram.org/bot<TOKEN>/getUpdates")
    print("                  and copy the numeric `chat.id` field.")
    print()
    try:
        token = input("Bot token: ").strip()
    except EOFError:
        return TelegramConfig()
    if not token:
        print("  → skipped, Telegram disabled.")
        return TelegramConfig()
    try:
        chats_csv = input("Chat IDs (comma-separated, e.g. 123456789,987654321): ").strip()
    except EOFError:
        return TelegramConfig()
    chat_ids = [s.strip() for s in chats_csv.split(",") if s.strip()]
    if not chat_ids:
        print("  → no chat IDs provided, Telegram disabled.")
        return TelegramConfig()

    cfg = TelegramConfig(bot_token=token, chat_ids=chat_ids)
    if save:
        _save_config(cfg)
        print(f"  Saved credentials to {CONFIG_PATH}")
    if test_ping:
        notifier = TelegramNotifier(cfg)
        if notifier.test_ping():
            print("  ✅ Ping sent successfully — check your Telegram.")
        else:
            print("  ⚠️  Ping failed — credentials may be wrong. "
                  "Edit configs/telegram.json or rerun with --telegram-setup.")
    return cfg


def get_default_notifier(interactive: bool = False) -> TelegramNotifier:
    """Build a notifier.

    Resolution order:
      1. `configs/telegram.json` if present + valid.
      2. Environment variables `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_IDS`.
      3. If `interactive=True` and stdin is a TTY, prompt the user
         and persist the answers to configs/telegram.json.
      4. Otherwise return a disabled notifier.
    """
    cfg = TelegramConfig.from_file(CONFIG_PATH)
    if cfg.enabled():
        return TelegramNotifier(cfg)
    cfg = TelegramConfig.from_env()
    if cfg.enabled():
        return TelegramNotifier(cfg)
    if interactive:
        import sys as _sys
        if _sys.stdin.isatty():
            cfg = prompt_user_for_credentials()
    return TelegramNotifier(cfg)
