"""`afp daemon` — long-running service that watches the S&P 500 for new
filings, retrains the model when needed, and pushes Telegram notifications.

Each cycle:
  1. Fetch the current S&P 500 constituent list (cached 24h).
  2. For each constituent CIK, check SEC submissions cache for any
     filing acceptance date newer than the last-known watermark.
  3. If new filings detected → run `afp refresh-all` (full pipeline).
  4. Predict on all S&P 500 tickers using the latest trained model.
  5. Push a digest to Telegram (top long-only ideas + new filings).
  6. Sleep for the configured interval (default 24h).

Designed to run as a systemd service or a long-running background
process. Persists watermarks in `data/processed/daemon_state.json`.

Setup:
  Environment variables (or `configs/telegram.json`):
    TELEGRAM_BOT_TOKEN=<your-bot-token>
    TELEGRAM_CHAT_IDS=<csv-of-chat-ids>

Launch:
  afp daemon                          # foreground, 24h cycle
  afp daemon --interval-hours 6       # poll 4×/day
  afp daemon --once                   # run one cycle and exit
  afp daemon --dry-run                # no retrain, just send a status ping
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from afp.cli._common import log

STATE_PATH = Path("data/processed/daemon_state.json")


# ---------------------------------------------------------------------------

def add_subparser(sub):
    p = sub.add_parser("daemon",
                       help="24h watcher: SP500 + filings + refresh-all + Telegram")
    p.add_argument("--interval-hours", type=float, default=24.0,
                   help="Sleep duration between cycles (default 24h)")
    p.add_argument("--once", action="store_true",
                   help="Run one cycle then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip refresh + prediction; only send a status ping to Telegram")
    p.add_argument("--no-telegram", action="store_true",
                   help="Don't send Telegram notifications even if configured")
    p.add_argument("--telegram-setup", action="store_true",
                   help="Force interactive Telegram setup at startup, overwriting "
                        "any existing configs/telegram.json")
    p.add_argument("--no-prompt", action="store_true",
                   help="Never prompt; just use existing config or env vars")
    p.add_argument("--config", default="configs/deployment_v1000.yaml")
    p.add_argument("--limit-ciks", type=int, default=1000,
                   help="Forwarded to refresh-all when a retrain is triggered")
    p.add_argument("--top-n", type=int, default=20,
                   help="Number of top long ideas to include in the digest")
    p.set_defaults(func=run)


# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _latest_filing_dates(sp500_ciks: Iterable[str]) -> dict[str, str | None]:
    """For each CIK, return the most recent `acceptanceDateTime` in the
    cached SEC submissions JSON. Returns None if no cache exists.

    This is intentionally CACHE-ONLY — fresh data comes in via
    refresh-all on a scheduled retrain trigger.
    """
    sub_dir = Path("data/raw/sec/submissions")
    out: dict[str, str | None] = {}
    for cik in sp500_ciks:
        path = sub_dir / f"CIK{cik}.json"
        if not path.exists():
            out[cik] = None
            continue
        try:
            payload = json.loads(path.read_text())
            recent = (payload.get("filings") or {}).get("recent") or {}
            accept_dates = recent.get("acceptanceDateTime") or []
            forms = recent.get("form") or []
            quarterly = [d for d, f in zip(accept_dates, forms)
                         if f in ("10-Q", "10-K", "10-Q/A", "10-K/A")]
            out[cik] = max(quarterly) if quarterly else None
        except Exception:
            out[cik] = None
    return out


def _detect_new_filings(sp500: pd.DataFrame, state: dict) -> list[dict]:
    """Compare current cached filings to last watermark; return list of new ones."""
    cik_to_ticker = dict(zip(sp500["cik"].astype(str), sp500["ticker"]))
    last_watermarks = state.get("last_filing_watermarks", {})
    current = _latest_filing_dates(sp500["cik"].astype(str).tolist())
    new = []
    for cik, latest in current.items():
        if latest is None:
            continue
        prev = last_watermarks.get(cik)
        if prev is None or latest > prev:
            new.append({"cik": cik,
                        "ticker": cik_to_ticker.get(cik, "?"),
                        "previous": prev,
                        "current": latest})
    return new


def _build_digest_message(cycle_info: dict) -> str:
    """Format a Markdown digest string for Telegram."""
    lines = []
    lines.append("📊 *afp-daemon digest*")
    lines.append(f"_cycle started at_ `{cycle_info['cycle_started']}`")
    lines.append("")
    sp500_n = cycle_info.get("sp500_size", 0)
    new_filings = cycle_info.get("new_filings", [])
    lines.append(f"• S&P 500 watched: {sp500_n} tickers")
    lines.append(f"• New 10-Q/10-K filings this cycle: *{len(new_filings)}*")
    if new_filings:
        sample = ", ".join(f["ticker"] for f in new_filings[:10])
        more = f" (and {len(new_filings)-10} more)" if len(new_filings) > 10 else ""
        lines.append(f"  ↳ {sample}{more}")

    if cycle_info.get("retrained"):
        meta = cycle_info.get("model_meta", {})
        lines.append(f"• Model *retrained* (best_iter={meta.get('best_iteration', '?')})")
    else:
        lines.append("• Model *not retrained* — no new filings or refresh disabled")

    top = cycle_info.get("top_picks") or []
    if top:
        lines.append("")
        lines.append("*Top long-side ideas*")
        for i, t in enumerate(top, 1):
            lines.append(f"  {i}. `{t['ticker']}` — rank {t['percentile']:.0f}%, exp.ret {t['expected_return']:+.1%}")
    lines.append("")
    lines.append(f"_next cycle in {cycle_info.get('next_cycle_hours', '?')}h_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def _run_refresh_all(limit_ciks: int) -> dict:
    """Invoke afp refresh-all in-process; returns the resulting model metadata."""
    from afp.cli import refresh_all
    ns = argparse.Namespace(
        config="configs/deployment_v1000.yaml",
        limit_ciks=limit_ciks,
        skip_ingest_sec=False,
        skip_ingest_prices=False,
        skip_build=False,
        skip_train=False,
        force_retrain=False,
        model_out="artifacts/models/lambdarank_v3_cross",
    )
    refresh_all.run(ns)
    meta_path = Path("artifacts/models/lambdarank_v3_cross/metadata.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _run_predict_on_sp500(sp500: pd.DataFrame, top_n: int) -> list[dict]:
    """Score every S&P 500 ticker via the programmatic predict API.

    Returns up to `top_n` highest-percentile names with their
    direction, expected return, vol, and rank.
    """
    if sp500.empty:
        return []
    tickers = sp500["ticker_yf"].astype(str).tolist()
    try:
        from afp.cli.predict import predict_tickers
        df, missing = predict_tickers(tickers, anchor="today")
    except Exception as exc:
        log.exception("daemon_predict_failed", extra={"err": str(exc)})
        return []
    if df is None or df.empty:
        return []
    top = df.head(top_n)
    return [{
        "ticker": str(r["ticker"]),
        "percentile": float(r["rank_pct"]) * 100,
        "expected_return": float(r["expected_pct_return"]),
        "annual_vol": float(r["annualized_vol_pct"]),
        "direction": str(r["direction"]),
    } for _, r in top.iterrows()]


# ---------------------------------------------------------------------------

def _one_cycle(args: argparse.Namespace, notifier) -> dict:
    cycle_started = datetime.now(tz=timezone.utc).isoformat()
    log.info("daemon_cycle_start", extra={"started": cycle_started})

    # 1. SP500
    from afp.data.sp500_universe import load_sp500_constituents
    try:
        sp500 = load_sp500_constituents()
    except Exception as exc:
        log.error("daemon_sp500_fetch_failed", extra={"err": str(exc)})
        sp500 = pd.DataFrame(columns=["ticker", "name", "sector", "sub_industry",
                                       "cik", "ticker_yf"])

    state = _load_state()
    new_filings = _detect_new_filings(sp500, state) if not sp500.empty else []

    retrained = False
    model_meta = {}
    if args.dry_run:
        log.info("daemon_dry_run", extra={"sp500": len(sp500), "new_filings": len(new_filings)})
    elif new_filings:
        log.info("daemon_new_filings_triggering_refresh", extra={"n": len(new_filings)})
        model_meta = _run_refresh_all(args.limit_ciks)
        retrained = True
    else:
        log.info("daemon_no_new_filings_skip_refresh")

    # 2. Persist watermarks
    if not sp500.empty:
        current = _latest_filing_dates(sp500["cik"].astype(str).tolist())
        state["last_filing_watermarks"] = current
    state["last_cycle"] = cycle_started
    _save_state(state)

    # 3. Build top picks: score the full S&P 500 via predict_tickers
    # (only worth running when we have a model AND samples ready).
    top_picks = []
    if not args.dry_run and not sp500.empty:
        try:
            top_picks = _run_predict_on_sp500(sp500, args.top_n)
        except Exception as exc:
            log.exception("daemon_top_picks_failed", extra={"err": str(exc)})
            # Fall back to just the new-filing tickers as a minimum signal
            top_picks = [{"ticker": f["ticker"], "percentile": 50.0,
                          "expected_return": 0.0,
                          "annual_vol": 0.0, "direction": "?"}
                         for f in new_filings[: args.top_n]]

    cycle_info = {
        "cycle_started": cycle_started,
        "sp500_size": len(sp500),
        "new_filings": new_filings,
        "retrained": retrained,
        "model_meta": model_meta,
        "top_picks": top_picks,
        "next_cycle_hours": args.interval_hours,
    }

    # 4. Telegram digest
    if notifier and not args.no_telegram:
        msg = _build_digest_message(cycle_info)
        results = notifier.send(msg)
        log.info("daemon_telegram_sent",
                 extra={"results": [r.get("ok") for r in results]})

    log.info("daemon_cycle_done",
             extra={"retrained": retrained, "new_filings": len(new_filings)})
    return cycle_info


# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    from afp.notify.telegram import (
        get_default_notifier,
        prompt_user_for_credentials,
        TelegramNotifier,
    )
    if args.telegram_setup:
        # User explicitly asked to (re)configure
        cfg = prompt_user_for_credentials(save=True, test_ping=True)
        notifier = TelegramNotifier(cfg)
    else:
        # Try config file → env → interactive prompt (unless --no-prompt or
        # --no-telegram or non-TTY)
        prompt_ok = not args.no_prompt and not args.no_telegram
        notifier = get_default_notifier(interactive=prompt_ok)
    if notifier.cfg.enabled():
        log.info("daemon_telegram_configured",
                 extra={"chat_ids": len(notifier.cfg.chat_ids)})
    else:
        log.warning("daemon_telegram_not_configured",
                    extra={"hint": "use --telegram-setup to configure, or set "
                                    "TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_IDS, or edit "
                                    "configs/telegram.json"})

    if args.once:
        _one_cycle(args, notifier)
        return 0

    log.info("daemon_loop_start", extra={"interval_hours": args.interval_hours})
    while True:
        try:
            _one_cycle(args, notifier)
        except Exception as exc:
            log.exception("daemon_cycle_failed", extra={"err": str(exc)})
            if notifier.cfg.enabled() and not args.no_telegram:
                notifier.send(f"⚠️ afp-daemon cycle failed: `{exc}`")
        time.sleep(args.interval_hours * 3600)
