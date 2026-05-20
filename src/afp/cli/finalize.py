"""`afp finalize` — build a one-shot markdown summary applying the decision rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from afp.cli._common import log


def add_subparser(sub):
    p = sub.add_parser("finalize", help="Synthesize metrics + diagnostics + attribution into "
                                        "a single decision-rule report")
    p.add_argument("--portfolio-id", required=True)
    p.add_argument("--reports-dir", default="reports/backtest")
    p.add_argument("--diagnostics-dir", default="reports/diagnostics")
    p.add_argument("--out", default=None,
                   help="Output markdown path; defaults to reports/backtest/<id>/summary.md")
    p.set_defaults(func=run)


def _decision(strategy: dict, eq_risk_active: dict, eq_risk: dict) -> str:
    """Apply Phase 11 decision rule A/B/C."""
    s_sh = strategy["sharpe"]
    s_mdd = strategy["max_drawdown"]
    s_ret = strategy["annualized_return"]
    r_sh = eq_risk_active.get("sharpe", eq_risk["sharpe"])
    r_mdd = eq_risk_active.get("max_drawdown", eq_risk["max_drawdown"])
    r_ret = eq_risk_active.get("annualized_return", eq_risk["annualized_return"])

    beats_sharpe = s_sh > r_sh
    beats_mdd = abs(s_mdd) <= abs(r_mdd)
    beats_return = s_ret > r_ret

    if not beats_sharpe and not beats_return:
        return ("**A — Strategy does NOT beat equal-risk-active baseline.**\n\n"
                "Per the Phase 11 decision rule: do not tune advanced models, do not add "
                "autoencoder, do not add diffusion. Improve data, universe, target, or "
                "portfolio design first.")
    if beats_sharpe and not beats_mdd:
        return ("**B — Strategy beats Sharpe but worse drawdown.**\n\n"
                "Investigate risk concentration (check `attribution.top_contributors` and "
                "`concentration.top10_weight`). Likely a few names dominate. Tighten "
                "single-stock caps or sector caps before adding model complexity.")
    if beats_sharpe and beats_mdd and beats_return:
        return ("**C — Strategy beats equal-risk-active on Sharpe, MDD, and return.**\n\n"
                "The right to try representation learning has been earned (Phase 10 of RFC-09: "
                "autoencoder / tabular transformer). Diffusion remains gated behind autoencoder "
                "showing further incremental alpha.")
    return ("**Borderline.** Some criteria met, others not. Recommend extending the test "
            "window (run pre-COVID and COVID-era sub-windows) before deciding.")


def run(args: argparse.Namespace) -> int:
    portfolio_id = args.portfolio_id
    bt_dir = Path(args.reports_dir) / portfolio_id
    diag_dir = Path(args.diagnostics_dir) / portfolio_id

    if not (bt_dir / "metrics.csv").exists():
        raise FileNotFoundError(bt_dir / "metrics.csv")
    metrics = pd.read_csv(bt_dir / "metrics.csv").set_index("portfolio")
    attribution = json.loads((bt_dir / "attribution.json").read_text())
    diagnostics = json.loads((diag_dir / "diagnostics.json").read_text())

    def row(name):
        return metrics.loc[name].to_dict() if name in metrics.index else {}

    strategy = row("strategy")
    spy = row("benchmark_spy")
    qqq = row("benchmark_qqq")
    eq_w = row("benchmark_equal_weight_universe")
    eq_r = row("benchmark_equal_risk_universe")
    eq_w_a = row("benchmark_equal_weight_active_universe")
    eq_r_a = row("benchmark_equal_risk_active_universe")
    rand = row("benchmark_random_positive_signal")
    shuf = row("benchmark_shuffled_signal")
    dv = row("benchmark_dollar_volume_weighted")

    decision = _decision(strategy, eq_r_a or eq_r, eq_r)

    decile_table = diagnostics.get("decile_table", [])
    spread = diagnostics.get("top_minus_bottom_log_return", float("nan"))
    pvr = diagnostics.get("positive_signal_vs_random", {})
    qspearman = diagnostics.get("quarterly_spearman", [])
    qdir = diagnostics.get("quarterly_direction_accuracy", [])

    def metrics_block(label, m):
        if not m:
            return f"- {label}: _not available_\n"
        return (f"- {label}: Sharpe {m['sharpe']:.2f} | Ann.Ret {m['annualized_return']*100:.1f}% "
                f"| Ann.Vol {m['annualized_volatility']*100:.1f}% | MDD {m['max_drawdown']*100:.1f}%\n")

    body = []
    body.append(f"# Phase 11 — {portfolio_id} Summary\n")
    body.append("## Strategy vs Benchmarks\n")
    body.append(metrics_block("**Strategy**", strategy))
    body.append(metrics_block("Equal-risk **active universe** (strictest)", eq_r_a))
    body.append(metrics_block("Equal-weight active universe", eq_w_a))
    body.append(metrics_block("Equal-risk full universe", eq_r))
    body.append(metrics_block("Equal-weight full universe", eq_w))
    body.append(metrics_block("Dollar-volume weighted", dv))
    body.append(metrics_block("Random positive signal", rand))
    body.append(metrics_block("Shuffled signal", shuf))
    body.append(metrics_block("SPY", spy))
    body.append(metrics_block("QQQ", qqq))

    body.append("\n## Decision\n")
    body.append(decision + "\n")

    body.append("\n## Signal Diagnostics (test split)\n")
    body.append(f"- top minus bottom decile mean realized log return: **{spread:.4f}**\n")
    body.append(f"- positive-signal vs random-positive baseline spread: "
                f"**{pvr.get('spread', float('nan')):.4f}** "
                f"(t≈{pvr.get('model_t_stat', float('nan')):.2f}, "
                f"n_trials={pvr.get('n_trials', 0)})\n")
    if decile_table:
        body.append("\n### Decile table\n")
        body.append("| Decile | mean pred | mean realized log ret | count |\n|---|---:|---:|---:|\n")
        for d in decile_table:
            body.append(f"| D{int(d['bucket'])+1} | {d['predicted_signal_mean']:+.3f} | "
                       f"{d['raw_log_return_mean']:+.4f} | {int(d['count'])} |\n")

    if qspearman:
        sp_vals = [q["spearman"] for q in qspearman if not pd.isna(q["spearman"])]
        if sp_vals:
            mean_s = sum(sp_vals) / len(sp_vals)
            pos_q = sum(1 for v in sp_vals if v > 0)
            body.append(f"\n### Quarterly Spearman: mean **{mean_s:+.3f}** | "
                       f"{pos_q}/{len(sp_vals)} quarters positive\n")
    if qdir:
        d_vals = [q["direction_accuracy"] for q in qdir]
        if d_vals:
            mean_d = sum(d_vals) / len(d_vals)
            body.append(f"\n### Quarterly direction accuracy: mean **{mean_d:.1%}**\n")

    cts = attribution.get("cash_turnover_summary", {})
    if cts:
        body.append("\n## Portfolio Attribution\n")
        body.append(f"- avg cash weight: **{cts.get('avg_cash_weight', 0)*100:.1f}%**\n")
        body.append(f"- max cash weight: **{cts.get('max_cash_weight', 0)*100:.1f}%**\n")
        body.append(f"- annualized turnover: **{cts.get('annualized_turnover', 0):.1f}×**\n")
        body.append(f"- cost drag/year: **{cts.get('cost_drag_per_year_pct', 0):.2f}%**\n")
    top = attribution.get("top_contributors", [])[:10]
    if top:
        body.append("\n### Top 10 Contributors\n")
        body.append("| internal_company_id | total_contribution |\n|---|---:|\n")
        for t in top:
            body.append(f"| {t['internal_company_id']} | {t['total_contribution']:+.4f} |\n")

    conc = attribution.get("concentration", [])
    if conc:
        hhi_vals = [c["herfindahl"] for c in conc]
        top10_vals = [c["top10_weight"] for c in conc]
        if hhi_vals:
            body.append(f"\n### Concentration\n"
                       f"- mean Herfindahl: **{sum(hhi_vals)/len(hhi_vals):.4f}** "
                       f"(>0.10 means top-heavy)\n"
                       f"- mean top-10 weight share: **{sum(top10_vals)/len(top10_vals)*100:.1f}%**\n")

    body.append("\n## Caveats\n"
                "- survivorship handling: imperfect / best-effort (universe drawn from current SEC ticker file)\n"
                "- price source: yfinance — research prototype quality, not production grade\n"
                "- test window: 2020-01 → 2026-05; longer/sub-period decomposition pending\n"
                "- restatements: as-of-download snapshots only\n")

    out_path = Path(args.out) if args.out else bt_dir / "summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(body))
    log.info("finalize_done", extra={"portfolio_id": portfolio_id, "out": str(out_path)})
    print(f"Wrote {out_path}")
    return 0
