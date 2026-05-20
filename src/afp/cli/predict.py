"""`afp predict TICKER [TICKER ...]` — user-facing prediction tool.

For each ticker:
  1. Resolve ticker → CIK via SEC company_tickers.json (cached after first call).
  2. Ensure submissions + companyfacts cached at data/raw/sec/.
  3. Ensure yfinance prices cached at data/raw/prices_cache/.
  4. Build event samples; keep the most recent filing event.
  5. Encode anonymous fundamental + price features.
  6. Run the trained LambdaRank booster.
  7. Output ranked predictions with restored expected return.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

# The OPEN samples deliberately carry NaN return / vol columns; the pandas
# warning about all-NA columns affecting dtype inference is benign here.
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="The behavior of DataFrame concatenation")
import pandas as pd

from afp.cli._common import log
from afp.data.identifiers import internal_company_id, normalize_cik
from afp.data.parse_companies import parse_company_tickers
from afp.data.parse_companyfacts import parse_company_facts
from afp.data.parse_submissions import parse_submissions
from afp.data.cache_refresh import refresh_prices_for_ticker, refresh_sec_for_cik
from afp.data.price_client_yfinance import YFinancePriceClient
from afp.data.sec_client import PermanentSecError, SecClient
from afp.data.trading_calendar import TradingCalendar
from afp.features.encoder import AnonymousFeatureEncoder
from afp.features.price_features import (
    PriceFeatureConfig,
    compute_price_features,
    transform_with_scaler,
)
from afp.features.sample_builder import build_event_samples, build_live_samples, build_open_samples
from afp.targets.target_transform import TargetConfig
from afp.targets.target_transform import apply as apply_target
from afp.targets.target_transform import restore as restore_signal
from afp.targets.volatility_scale import ScaleConfig, compute_ex_ante_scale_for_samples


SEC_USER_AGENT = "anonymous-fundamental-portfolio research jasonya2206@gmail.com"
MODEL_DIR = Path("artifacts/models/lambdarank_v3")
ENCODER_DIR = Path("artifacts/feature_encoder/v1")
RAW_DIR = Path("data/raw/sec")
PRICES_CACHE = Path("data/raw/prices_cache")
CALENDAR_PATH = Path("data/processed/trading_calendar.parquet")


def add_subparser(sub):
    p = sub.add_parser("predict", help="Predict on user-supplied US tickers")
    p.add_argument("tickers", nargs="*", help="Ticker symbols (e.g. AAPL MSFT NVDA)")
    p.add_argument("--tickers-file", default=None,
                   help="Path to text file with one ticker per line")
    p.add_argument("--start-date", default="2010-01-01",
                   help="Earliest price data to fetch (default: 2010-01-01)")
    p.add_argument("--end-date", default=None,
                   help="Latest price date (default: today)")
    p.add_argument("--top-n", type=int, default=None,
                   help="If set, also print a suggested top-N long-only portfolio")
    p.add_argument("--json", action="store_true",
                   help="Output JSON instead of human-readable table")
    p.add_argument("--anchor", choices=["today", "filing"], default="today",
                   help="'today' (default) anchors prediction at the latest available "
                        "trading day using today's prices/vol; 'filing' anchors at the "
                        "entry day right after the most recent filing")
    p.add_argument("--no-refresh", action="store_true",
                   help="Skip the SEC + price cache freshness check. Faster but may use "
                        "stale data. Default is to verify cache is up-to-date.")
    p.add_argument("--refresh-cohort", action="store_true",
                   help="Force rebuild of the reference cohort even if its quarter matches "
                        "the latest available training-data quarter. Use after `afp ingest-sec` "
                        "or `afp build-dataset` re-runs that added new filings.")
    p.set_defaults(func=run)


# ---------------------------------------------------------------------------

def _load_ticker_map() -> pd.DataFrame:
    cache = RAW_DIR / "company_tickers" / "company_tickers.json"
    if cache.exists():
        return parse_company_tickers(json.loads(cache.read_text()))
    client = SecClient(user_agent=SEC_USER_AGENT)
    raw = client.company_tickers()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw))
    return parse_company_tickers(raw)


def _ensure_sec_for_cik(cik: str, client: SecClient) -> tuple[list[dict], dict]:
    sub_cache = RAW_DIR / "submissions" / f"CIK{cik}.json"
    facts_cache = RAW_DIR / "companyfacts" / f"CIK{cik}.json"
    sub_cache.parent.mkdir(parents=True, exist_ok=True)
    facts_cache.parent.mkdir(parents=True, exist_ok=True)
    if sub_cache.exists():
        primary = json.loads(sub_cache.read_text())
        pages = [primary]
        for f in (primary.get("filings") or {}).get("files") or []:
            name = f.get("name")
            if not name:
                continue
            try:
                page = client.get_json(f"https://data.sec.gov/submissions/{name}")
                pages.append({"cik": cik, "filings": {"recent": page}})
            except Exception:
                continue
    else:
        pages = client.submissions_full(cik)
        sub_cache.write_text(json.dumps(pages[0]))
    if facts_cache.exists():
        facts = json.loads(facts_cache.read_text())
    else:
        facts = client.company_facts(cik)
        facts_cache.write_text(json.dumps(facts))
    return pages, facts


def _ensure_prices_for_ticker(ticker: str, start: str, end: str,
                              client: YFinancePriceClient) -> pd.DataFrame:
    return client.get_daily_prices(ticker, start, end)


def _load_calendar(start: str, end: str) -> TradingCalendar:
    if CALENDAR_PATH.exists():
        cal_df = pd.read_parquet(CALENDAR_PATH)
        cal = TradingCalendar.from_dates(cal_df["date"].tolist())
        # Ensure it covers the requested window; if not, rebuild.
        if (pd.Timestamp(cal.trading_dates[0]).date() <= pd.Timestamp(start).date()
                and pd.Timestamp(cal.trading_dates[-1]).date() >= pd.Timestamp(end).date()):
            return cal
    return TradingCalendar.build(start, end)


def _load_model_state():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_DIR}. "
            f"Run `python3 scripts/save_lambdarank_model.py` once to create it.")
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(MODEL_DIR / "booster.txt"))
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    npz = np.load(MODEL_DIR / "price_stats.npz", allow_pickle=True)
    price_stats = {fid: (float(med), float(iqr))
                   for fid, med, iqr in zip(npz["feature_ids"], npz["medians"], npz["iqrs"])}
    return booster, meta, price_stats


def _reference_cohort_scores(booster, encoder, feature_ids, price_ids, price_stats,
                              force_rebuild: bool = False
                              ) -> tuple[np.ndarray | None, str | None]:
    """Build a reference cohort of raw LambdaRank scores for the most recent
    quarter in the training universe.

    Auto-rebuilds when stale:
      - cache's quarter is older than the latest quarter available in
        `data/processed/event_samples.parquet`
      - underlying samples were updated after the cache (mtime check)
      - `force_rebuild=True`
    """
    cache = MODEL_DIR / "reference_cohort.parquet"
    samples_path = Path("data/processed/event_samples.parquet")
    facts_path = Path("data/processed/financial_facts_long.parquet")
    prices_path = Path("data/processed/prices_daily.parquet")
    if not (samples_path.exists() and facts_path.exists() and prices_path.exists()):
        # No underlying training data — fall back to cache if present, else None.
        if cache.exists():
            df = pd.read_parquet(cache)
            log.warning("reference_cohort_no_training_data_using_stale_cache",
                        extra={"quarter": str(df["quarter"].iloc[0])})
            return df["raw_score"].to_numpy(), str(df["quarter"].iloc[0])
        return None, None

    # Determine what the freshest possible quarter is, cheaply.
    entries = pd.to_datetime(pd.read_parquet(samples_path, columns=["entry_date"])["entry_date"])
    latest_q_in_data = entries.dt.to_period("Q").max()
    latest_q_str = str(latest_q_in_data)
    # If the training pool itself hasn't seen new data in a while, warn the user.
    days_since_latest_sample = (date.today() - entries.max().date()).days
    if days_since_latest_sample > 120:
        log.warning("reference_cohort_underlying_universe_stale",
                    extra={"latest_sample_date": str(entries.max().date()),
                           "days_since": int(days_since_latest_sample),
                           "hint": "re-run `afp ingest-sec --config configs/deployment_v1000.yaml "
                                   "--limit-ciks 1000` then `afp build-dataset` to refresh the "
                                   "training universe"})

    stale = force_rebuild
    if not stale and cache.exists():
        df = pd.read_parquet(cache)
        cached_q = str(df["quarter"].iloc[0])
        if cached_q != latest_q_str:
            log.info("reference_cohort_stale_quarter",
                     extra={"cached": cached_q, "available": latest_q_str})
            stale = True
        elif cache.stat().st_mtime < samples_path.stat().st_mtime:
            log.info("reference_cohort_samples_newer_than_cache")
            stale = True
        if not stale:
            return df["raw_score"].to_numpy(), cached_q

    log.info("predict_building_reference_cohort",
             extra={"quarter": latest_q_str,
                    "reason": "force" if force_rebuild else "stale_or_missing"})
    samples = pd.read_parquet(samples_path).dropna(subset=["target_normalized_signal"])
    facts = pd.read_parquet(facts_path)
    prices = pd.read_parquet(prices_path)

    samples["entry_ts"] = pd.to_datetime(samples["entry_date"])
    last_q = samples["entry_ts"].dt.to_period("Q").max()
    cohort = samples[samples["entry_ts"].dt.to_period("Q") == last_q].copy()
    if cohort.empty:
        return None, None

    from afp.models.datasets import build_tabular_dataset

    pf_cfg = PriceFeatureConfig(enabled=True)
    pf_df, _ = compute_price_features(cohort, prices, pf_cfg)
    pf_df = transform_with_scaler(pf_df, price_ids, price_stats)
    scaled, missing_arr, meta_arr, sids = encoder.transform(cohort, facts)
    targets = cohort.set_index("sample_id")["target_normalized_signal"]
    ds = build_tabular_dataset(scaled, missing_arr, meta_arr, sids, targets, feature_ids)
    pf_sub = pf_df.set_index("sample_id").reindex(ds.sample_ids)
    extra = pf_sub[price_ids].to_numpy(dtype=np.float64)
    extra_miss = pf_sub[[f"{fid}_missing" for fid in price_ids]].to_numpy(dtype=np.float64)
    X = np.concatenate([ds.X, extra, extra_miss], axis=1)
    scores = booster.predict(X)

    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"raw_score": scores, "quarter": str(last_q)}).to_parquet(cache, index=False)
    return scores, str(last_q)


def _resolve_tickers(tickers_arg: list[str], tickers_file: str | None) -> list[str]:
    out: list[str] = []
    if tickers_file:
        for line in Path(tickers_file).read_text().splitlines():
            t = line.strip().upper()
            if t and not t.startswith("#"):
                out.append(t)
    out.extend(t.strip().upper() for t in tickers_arg if t.strip())
    seen, deduped = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    tickers = _resolve_tickers(args.tickers, args.tickers_file)
    if not tickers:
        log.warning("predict_no_tickers_given")
        return 1

    log.info("predict_start", extra={"n_tickers": len(tickers)})
    ticker_map = _load_ticker_map()
    ticker_to_row = {row["ticker"]: row for _, row in ticker_map.iterrows()}

    sec = SecClient(user_agent=SEC_USER_AGENT)
    yf = YFinancePriceClient(cache_root=str(PRICES_CACHE), pause_seconds=0.5)

    end_date = args.end_date or date.today().isoformat()
    start_date = args.start_date
    calendar = _load_calendar(start_date, end_date)

    # Step 1: per-ticker SEC + price fetch + parse, with cache freshness check
    refresh_enabled = not args.no_refresh
    filings_frames: list[pd.DataFrame] = []
    facts_frames: list[pd.DataFrame] = []
    price_frames: list[pd.DataFrame] = []
    resolved_tickers: list[str] = []
    missing: list[str] = []
    refreshed_count = 0
    for t in tickers:
        row = ticker_to_row.get(t)
        if row is None:
            missing.append(t)
            log.warning("predict_ticker_not_found_in_sec", extra={"ticker": t})
            continue
        cik = row["cik"]
        icid = row["internal_company_id"]
        try:
            if refresh_enabled:
                pages, facts_payload, sec_refreshed = refresh_sec_for_cik(cik, sec, RAW_DIR)
                if pages is None:
                    missing.append(t)
                    continue
                if sec_refreshed:
                    refreshed_count += 1
            else:
                pages, facts_payload = _ensure_sec_for_cik(cik, sec)
        except PermanentSecError as exc:
            log.warning("predict_no_sec_data", extra={"ticker": t, "cik": cik,
                                                     "status": exc.status_code})
            missing.append(t)
            continue
        for page in pages:
            df = parse_submissions(page, allowed_forms=("10-Q", "10-K"))
            if not df.empty:
                filings_frames.append(df)
        if facts_payload is not None:
            facts_frames.append(parse_company_facts(facts_payload))
        try:
            if refresh_enabled:
                px, price_refreshed = refresh_prices_for_ticker(
                    t, yf, start_date, end_date, PRICES_CACHE)
                if price_refreshed:
                    refreshed_count += 1
            else:
                px = _ensure_prices_for_ticker(t, start_date, end_date, yf)
        except Exception as exc:
            log.warning("predict_price_fetch_failed", extra={"ticker": t, "err": str(exc)})
            missing.append(t)
            continue
        if px.empty:
            missing.append(t)
            continue
        px = px.copy()
        px["ticker_at_date"] = t
        px["internal_company_id"] = icid
        price_frames.append(px)
        resolved_tickers.append(t)
    if refreshed_count:
        log.info("predict_cache_refreshed", extra={"refresh_actions": refreshed_count,
                                                    "tickers": len(tickers)})

    if not resolved_tickers:
        print("No tickers could be resolved. Check the ticker symbols.")
        if missing:
            print(f"Missing: {', '.join(missing)}")
        return 1

    filings = pd.concat(filings_frames, ignore_index=True).drop_duplicates(
        subset=["filing_event_id"]) if filings_frames else pd.DataFrame()
    facts = pd.concat(facts_frames, ignore_index=True).drop_duplicates(
        subset=["fact_id"]) if facts_frames else pd.DataFrame()
    prices = pd.concat(price_frames, ignore_index=True)

    log.info("predict_data_ready",
             extra={"n_resolved": len(resolved_tickers), "n_filings": len(filings),
                    "n_facts": len(facts), "n_price_rows": len(prices)})

    # Step 2: build LIVE samples (anchored at today, default) or OPEN samples
    # (anchored at the latest filing's entry day).
    if args.anchor == "today":
        open_samples = build_live_samples(filings, prices, calendar)
    else:
        open_samples = build_open_samples(filings, prices, calendar)
    if open_samples.empty:
        print(f"Could not build {args.anchor}-anchored samples — make sure each ticker has at "
              "least one 10-K/10-Q and a recent valid price.")
        return 1

    # The completed-position samples are still used so ex-ante scale can use
    # prior event-to-event returns as part of the hybrid estimate.
    completed, _excluded = build_event_samples(
        filings, prices, calendar, benchmarks=None,
        train_end_date="1900-01-01", validation_end_date="1900-01-02",
    )
    if not completed.empty:
        # Drop fully-empty columns from `completed` before concat to silence the
        # pandas FutureWarning about all-NA columns affecting dtype inference.
        completed = completed.dropna(axis=1, how="all")
        all_samples = pd.concat([completed, open_samples], ignore_index=True)
    else:
        all_samples = open_samples
    all_samples = compute_ex_ante_scale_for_samples(all_samples, prices, calendar, ScaleConfig())
    all_samples = apply_target(all_samples, TargetConfig(k=2.5))

    # Now select ONLY the synthetic prediction samples (live or open) for prediction.
    target_split = "live" if args.anchor == "today" else "open"
    latest = all_samples[all_samples["split"] == target_split].copy()
    latest = latest.dropna(subset=["ex_ante_scale"])
    if latest.empty:
        print("No open samples have a valid ex-ante scale. Need more price history.")
        return 1
    latest["entry_date_ts"] = pd.to_datetime(latest["entry_date"])
    # Open samples don't have realized targets — give them a placeholder so the
    # downstream `build_tabular_dataset` (which drops NaN-target rows) keeps them.
    # The placeholder is never used because predict() only consumes X.
    latest["target_normalized_signal"] = 0.0

    # Step 3: encode features
    encoder = AnonymousFeatureEncoder.load(str(ENCODER_DIR))
    booster, meta, price_stats = _load_model_state()
    feature_ids = encoder.artifact.feature_map.feature_ids

    scaled, missing_arr, meta_arr, sample_ids = encoder.transform(latest, facts)

    pf_cfg = PriceFeatureConfig(enabled=True)
    pf_df, price_ids = compute_price_features(latest, prices, pf_cfg)
    pf_df = transform_with_scaler(pf_df, price_ids, price_stats)

    from afp.models.datasets import build_tabular_dataset
    targets = latest.set_index("sample_id")["target_normalized_signal"]
    ds = build_tabular_dataset(scaled, missing_arr, meta_arr, sample_ids, targets, feature_ids)
    pf_sub = pf_df.set_index("sample_id").reindex(ds.sample_ids)
    extra = pf_sub[price_ids].to_numpy(dtype=np.float64)
    extra_miss = pf_sub[[f"{fid}_missing" for fid in price_ids]].to_numpy(dtype=np.float64)
    X = np.concatenate([ds.X, extra, extra_miss], axis=1)

    # Step 4: predict — score user tickers
    raw_score = booster.predict(X)

    # Build a reference cohort from the existing 1000-CIK universe's most recent quarter,
    # so even a single user ticker has 1000+ peers to be ranked against.
    ref_scores, ref_quarter = _reference_cohort_scores(booster, encoder, feature_ids,
                                                       price_ids, price_stats,
                                                       force_rebuild=args.refresh_cohort)
    if ref_scores is not None and len(ref_scores) > 0:
        log.info("predict_reference_cohort_loaded",
                 extra={"size": len(ref_scores), "quarter": ref_quarter})
        # Combined ranking — user scores vs reference pool
        combined = np.concatenate([raw_score, ref_scores])
        order = np.argsort(np.argsort(combined))
        pct_rank_all = (order + 1) / (len(combined) + 1)
        pct_rank = pct_rank_all[: len(raw_score)]
        signal = 2.0 * pct_rank - 1.0
    else:
        # Fallback: within the user's own cohort only.
        order = np.argsort(np.argsort(raw_score))
        pct_rank = (order + 1) / (len(order) + 1)
        signal = 2.0 * pct_rank - 1.0

    # The signal here is a cohort percentile, NOT a tanh-of-magnitude. Restoring it
    # with atanh produces wildly inflated magnitudes (atanh(0.9) ≈ 1.47 →  ~150% return).
    # We instead apply heavy shrinkage so the displayed "expected return" is a
    # conservative directional estimate, not a calibrated forecast.
    shrunk_signal = np.clip(signal * 0.25, -0.4, 0.4)
    cfg = TargetConfig(k=2.5, restore_method="atanh", restore_clip_abs=0.99)
    latest_by_sid = latest.set_index("sample_id").loc[ds.sample_ids].reset_index()
    scales = latest_by_sid["ex_ante_scale"].to_numpy()
    restored = restore_signal(shrunk_signal, scales, cfg)

    today = date.today()
    # Pre-compute each company's empirical filing cadence (median gap between
    # consecutive accepted_datetimes). Far more accurate than a fixed offset.
    cadence_days: dict[str, int] = {}
    if not filings.empty:
        for icid, grp in filings.groupby("internal_company_id"):
            dates_ = pd.to_datetime(grp["accepted_datetime"]).dt.date.dropna().sort_values()
            if len(dates_) >= 2:
                gaps = np.diff(dates_.to_numpy()).astype("timedelta64[D]").astype(int)
                cadence_days[icid] = int(np.median(gaps))
            else:
                cadence_days[icid] = 91   # one-quarter fallback
    rows = []
    for i, sid in enumerate(ds.sample_ids):
        rec = latest_by_sid.iloc[i]
        period_end_ts = pd.Timestamp(rec.get("period_end_date") or rec["report_event_date"]).date()
        accepted_ts = pd.Timestamp(rec.get("report_event_date")).date()
        days_since_period_end = (today - period_end_ts).days
        days_since_filing = (today - accepted_ts).days
        form = rec["form_type"]
        gap = cadence_days.get(rec["internal_company_id"], 91)
        next_filing_est = (pd.Timestamp(accepted_ts) + pd.Timedelta(days=gap)).date()
        days_until_next = (next_filing_est - today).days
        # Magnitude
        exp_log_ret = float(restored[i])
        exp_pct_ret = float(math.exp(exp_log_ret) - 1)
        # ex_ante_scale already encodes expected vol over the holding period.
        daily_vol = float(scales[i]) / max(1, math.sqrt(int(rec["holding_days"]) or 60))
        annual_vol = daily_vol * math.sqrt(252)
        rows.append({
            "ticker": rec.get("ticker_at_event") or _icid_to_ticker(rec["internal_company_id"], ticker_map),
            "cik": rec["cik"],
            "predicted_signal": float(signal[i]),
            "raw_score": float(raw_score[i]),
            "rank_pct": float(pct_rank[i]),
            "direction": "↑ Up" if signal[i] > 0.05 else ("↓ Down" if signal[i] < -0.05 else "~ Flat"),
            "expected_pct_return": exp_pct_ret,
            "expected_log_return": exp_log_ret,
            "annualized_vol_pct": annual_vol,
            "ex_ante_scale": float(scales[i]),
            "last_filing_date": str(accepted_ts),
            "last_filing_form": form,
            "last_period_end": str(period_end_ts),
            "days_since_last_filing": int(days_since_filing),
            "next_filing_est": str(next_filing_est),
            "days_until_next_filing": int(days_until_next),
        })
    df_out = pd.DataFrame(rows).sort_values("predicted_signal", ascending=False).reset_index(drop=True)

    if args.json:
        print(json.dumps({"missing": missing, "predictions": df_out.to_dict(orient="records")},
                         indent=2, default=str))
    else:
        _print_table(df_out, missing)
        if args.top_n:
            _print_portfolio(df_out, args.top_n)
    return 0


def _icid_to_ticker(icid: str, ticker_map: pd.DataFrame) -> str:
    row = ticker_map[ticker_map["internal_company_id"] == icid]
    return row["ticker"].iloc[0] if not row.empty else icid


def _print_table(df: pd.DataFrame, missing: list[str]):
    """Single ticker → narrative; multiple → compact table."""
    if len(df) == 1:
        _print_one(df.iloc[0])
    else:
        _print_multi(df)
    if missing:
        print(f"\nSkipped (no SEC/price data): {', '.join(missing)}")


def _print_one(r: pd.Series):
    print()
    print(f"=== {r['ticker']} ===")
    print(f"  Last filing            {r['last_filing_form']} on {r['last_filing_date']} "
          f"(period ending {r['last_period_end']})")
    print(f"  Days since             {r['days_since_last_filing']} days ago")
    print(f"  Next filing (ESTIMATE) ~{r['next_filing_est']}  "
          f"({r['days_until_next_filing']} days from today)")
    print(f"                         ↑ estimated from this company's median filing cadence "
          f"— SEC has not confirmed yet")
    print()
    print(f"  Direction              {r['direction']}")
    print(f"  Expected return        {r['expected_pct_return']*100:+.2f}%  "
          f"(over the estimated window)")
    print(f"  Expected vol           ±{r['annualized_vol_pct']*100:.1f}% annualized  "
          f"(daily ≈ ±{r['ex_ante_scale']*100/max(1, math.sqrt(60)):.2f}%)")
    pct = int(round(r['rank_pct'] * 100))
    suffix = "th" if 10 <= pct % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")
    print(f"  Rank vs peers          {pct}{suffix} percentile "
          f"(within same-quarter reporting cohort)")
    print(f"  Confidence             {_confidence_label(r['rank_pct'])}")
    print()


def _confidence_label(pct: float) -> str:
    """Five-tier asymmetric label based on cohort ranking position.

    All tiers reflect rank position — never a "high-confidence short"
    interpretation, since the strategy is long-only.

      ≥85th pct  →  high          (top 15%, strong long candidate)
      70-85th    →  moderate      (top 30%, plausible long)
      30-70th    →  neutral       (middle 40%, no clear thesis)
      15-30th    →  moderate-avoid (bottom 30%, weak fundamentals vs peers)
       <15th    →  avoid          (bottom 15%, clear under-performer in cohort)
    """
    if pct >= 0.85:
        return "high"
    if pct >= 0.70:
        return "moderate"
    if pct >= 0.30:
        return "neutral"
    if pct >= 0.15:
        return "moderate-avoid"
    return "avoid"


def _print_multi(df: pd.DataFrame):
    header = (f"{'Rank':>4}  {'Ticker':<6}  {'Dir':<7}  {'Exp.Ret':>8}  "
              f"{'Vol':>5}  {'Last filing':<12}  {'Next~':<12}  {'Days':>4}")
    print()
    print(header)
    print("-" * len(header))
    for i, r in df.iterrows():
        print(f"{i+1:>4}  {r['ticker']:<6}  {r['direction']:<7}  "
              f"{r['expected_pct_return']*100:>+7.2f}%  {r['annualized_vol_pct']*100:>4.0f}%  "
              f"{r['last_filing_date']:<12}  ~{r['next_filing_est']:<11}  "
              f"{r['days_since_last_filing']:>3}d")
    print()
    print("  Dir       = predicted direction over the next filing window")
    print("  Exp.Ret   = expected pct return between now and next filing (conservative)")
    print("  Vol       = annualized vol estimate (1σ)")
    print("  Next~     = ESTIMATED next filing date (median historical cadence)")
    print("  Days      = days since latest 10-Q/10-K filing")


def _print_portfolio(df: pd.DataFrame, top_n: int):
    longs = df[df["predicted_signal"] > 0].head(top_n).copy()
    if longs.empty:
        print("No positive-signal names to construct a long-only portfolio.")
        return
    inv_vol = 1.0 / longs["ex_ante_scale"].clip(lower=1e-3)
    weights = inv_vol / inv_vol.sum()
    longs["suggested_weight"] = weights.values
    print(f"\nSuggested long-only portfolio (top {len(longs)} positive signals, inverse-vol weighted):")
    print(f"  {'Ticker':<6}  {'Signal':>7}  {'Weight':>7}  {'Exp.Ret':>8}")
    for _, r in longs.iterrows():
        print(f"  {r['ticker']:<6}  {r['predicted_signal']:+7.3f}  "
              f"{r['suggested_weight']*100:>6.1f}%  {r['expected_pct_return']*100:>+7.1f}%")
