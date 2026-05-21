# Phase 68 — `afp daemon` 24h watcher + Telegram notifications

> Status: ✓ shipped. Long-running service that monitors the S&P 500
> for new earnings filings, retrains the Phase 62 model when needed,
> and pushes a digest to Telegram subscribers.

## What it does

```
┌─────────────────────────────────────────────────────────────────┐
│  Every N hours (default 24):                                    │
│                                                                 │
│  1. Pull current S&P 500 constituent list from Wikipedia        │
│     (cached 24h to data/processed/sp500_constituents.parquet)   │
│                                                                 │
│  2. For each constituent CIK, read cached SEC submissions JSON  │
│     and find the latest 10-Q/10-K acceptanceDateTime.           │
│                                                                 │
│  3. Compare to last-cycle watermarks (data/processed/           │
│     daemon_state.json). Detect any new filings since last run.  │
│                                                                 │
│  4. IF new filings detected → run `afp refresh-all` (full       │
│     pipeline: ingest + features + retrain).                     │
│                                                                 │
│  5. Predict on all S&P 500 tickers using the freshly trained    │
│     model (Phase 62 cross-features stack).                      │
│                                                                 │
│  6. Send a Markdown digest to every configured Telegram chat.   │
│                                                                 │
│  7. Sleep --interval-hours, repeat.                             │
└─────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# Just run — first launch interactively prompts for the bot token
# and chat IDs, then saves them to configs/telegram.json. Press
# Enter to skip Telegram entirely.
afp daemon
```

The first start prints:

```
============================================================
  Telegram bot setup
============================================================
Press Enter at any prompt to skip Telegram notifications.

To create a bot:  message @BotFather on Telegram, follow the prompts,
                  and copy the token of the form 1234567890:AAH...
To get a chat_id: message your bot from your personal Telegram account,
                  then visit
                  https://api.telegram.org/bot<TOKEN>/getUpdates
                  and copy the numeric `chat.id` field.

Bot token: <paste>
Chat IDs (comma-separated, e.g. 123456789,987654321): <paste>
  Saved credentials to configs/telegram.json
  ✅ Ping sent successfully — check your Telegram.
```

Subsequent launches read from `configs/telegram.json` and skip the
prompt automatically.

### Non-interactive setup

Pre-populate credentials without prompting via env vars OR JSON:

```bash
# env vars (then pass --no-prompt to bypass the interactive setup)
export TELEGRAM_BOT_TOKEN='1234567890:AAH...'
export TELEGRAM_CHAT_IDS='123456789,987654321'
afp daemon --no-prompt

# OR write the JSON directly
cat > configs/telegram.json <<'EOF'
{"bot_token": "1234567890:AAH...", "chat_ids": ["123456789"]}
EOF
afp daemon --no-prompt
```

### Reconfigure / change credentials later

```bash
afp daemon --telegram-setup
```

forces the interactive setup again, overwriting `configs/telegram.json`.

### Background / systemd

```bash
nohup afp daemon --no-prompt > daemon.log 2>&1 &   # background
# systemd unit example: see deploy/systemd/afp-daemon.service
```

Use `--no-prompt` in systemd / Docker / CI where stdin is not a TTY.

## Notification format

The digest is a Markdown message:

```
📊 *afp-daemon digest*
_cycle started at_ `2026-05-21T08:00:00Z`

• S&P 500 watched: 503 tickers
• New 10-Q/10-K filings this cycle: *7*
  ↳ AAPL, MSFT, GOOG, AMZN, NVDA, META, TSLA
• Model *retrained* (best_iter=287)

*Top long-side ideas*
  1. `NVDA` — rank 94%, exp.ret +6.3%
  2. `AAPL` — rank 91%, exp.ret +4.8%
  ...

_next cycle in 24.0h_
```

When no new filings exist, the daemon **does not retrain** — it sends
a lightweight digest indicating "no new filings" and resumes
waiting. This keeps daily compute cost low.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--interval-hours N` | 24 | Sleep duration between cycles |
| `--once` | off | Run one cycle and exit (testing) |
| `--dry-run` | off | Skip retrain + prediction; only ping Telegram |
| `--no-telegram` | off | Disable notifications even if configured |
| `--limit-ciks N` | 1000 | Forwarded to refresh-all on retrain trigger |
| `--top-n N` | 20 | Number of top picks in the digest |
| `--config PATH` | `configs/deployment_v1000.yaml` | Pipeline config |

## State persistence

The daemon keeps a small JSON state file at
`data/processed/daemon_state.json` containing:

- `last_cycle`: ISO timestamp of last cycle
- `last_filing_watermarks`: dict of `{cik: latest_acceptanceDateTime}`
  per S&P 500 company

State is read on startup so the daemon survives restarts cleanly.

## Failure handling

- If `refresh-all` fails, the daemon catches the exception and posts
  an error message to Telegram (`⚠️ afp-daemon cycle failed: ...`),
  then continues to the next cycle.
- If Telegram is unreachable, the cycle still completes — Telegram
  failures are logged but not fatal.
- If the SP500 fetch fails (Wikipedia unavailable), the cycle
  proceeds without filing-watch but still emits a status update.

## Phase 62 is now the default model

`refresh-all` (Phase 67) was modified in Phase 68 to **mirror the
trained Phase 62 artifacts** to `artifacts/models/lambdarank_v3/`
(the legacy default path) in addition to
`artifacts/models/lambdarank_v3_cross/`. This means:

- Any tooling that hard-codes the legacy `lambdarank_v3/` path now
  picks up the Phase 62 cross-features model.
- The predict CLI's auto-discovery still resolves
  `lambdarank_v3_cross` first; the legacy mirror is a safety net.

## Implementation

- `src/afp/data/sp500_universe.py` — Wikipedia fetcher (cached 24h).
- `src/afp/notify/__init__.py` + `notify/telegram.py` — HTTP-only
  Telegram client.
- `src/afp/cli/daemon.py` — the long-running service.
- `src/afp/cli/refresh_all.py` — mirrors trained model to legacy path.
- `src/afp/cli/predict.py` — exposes `predict_tickers()` programmatic
  helper so the daemon can score the full S&P 500 in-process for the
  digest's "Top long-side ideas" section.

## Tests

`tests/test_sp500_universe.py`, `tests/test_telegram_notifier.py`,
`tests/test_daemon_helpers.py` — 16 new unit tests, all offline (no
real HTTP calls).

Total suite now 240 tests, all passing.

## Limitations

- The daemon relies on `refresh-all`'s SEC ingest to keep the
  submissions cache fresh. A first cold-cache cycle will be slow
  (~30 min) because all SP500 companyfacts need to be fetched.
- Wikipedia is not a 100%-reliable source. Constituent changes are
  rare so the 24h cache is fine, but a managed data feed (e.g.,
  iShares IVV holdings CSV) would be more robust for capital
  deployment.
