# Phase 16 — SIC Sector Parsing + Portfolio Sector Caps

## Why

Phase 11 attribution showed Herfindahl 0.012 / top-10 weight 20% — diversified
at the *name* level — but the 1000-CIK universe is heavily tilted toward
tech, finance, and healthcare. The portfolio almost certainly inherits the
same sector imbalance, which Phase 11's "equal-risk-active" beat the strategy
by 0.76 Sharpe partly because risk parity *implicitly* spreads across sectors.

Sector caps in the portfolio module let us neutralize that bias without
giving the model any sector knowledge.

## RFC Compliance

- The prediction model still sees **only anonymous feature IDs**. Sector
  data is loaded into the portfolio module only.
- RFC-01 §2.5 / §3.3 explicitly authorize sector usage in the portfolio
  module while forbidding it in model input.
- Sector codes come from SEC submissions JSON (`sicCode` / `sicDescription`)
  — public point-in-time data, no look-ahead.

## SIC Source

`afp.data.sectors.parse_sector_from_submissions` extracts the `sic` field
from each cached `submissions/CIK*.json` and maps the 4-digit SIC code down
to its **1-digit SIC division**. Coarse on purpose: enough to neutralize
mega-sector tilt, not enough to encode subtle industry signals into the
portfolio. Ten divisions, deterministic mapping.

`afp.cli.build_sectors` walks the cache and writes
`data/processed/sectors.parquet`:

```
internal_company_id, cik, sic, sic_description, sic_division
```

## Sector Cap Algorithm

`afp.portfolio.allocation._apply_sector_cap(weights, sectors, cap)`:

```
loop up to 20 times:
    for each sector s:
        if sum(weights[mask_s]) / total > cap:
            scale weights[mask_s] down to exactly cap × total
            redistribute excess to non-capped sectors proportional to their weights
    break when no sector is over cap
```

Then the single-stock cap is re-applied (an iteratively-cap → sector-cap →
iteratively-cap pattern handles the rare case where a sector cap pushes
a stock above its single-name cap).

## Wiring

`PortfolioConfig.max_sector_weight: float | None`. If `None`, sector cap is
skipped entirely — preserves backward compatibility.

```yaml
portfolio:
  sector:
    use_sector_caps: true
    max_sector_weight: 0.30
```

The backtest CLI auto-loads `data/processed/sectors.parquet` if it exists
and `use_sector_caps: true`. If the file is missing or the config flag is
off, sector caps are silently skipped.

`build_target_portfolio(sector_map=...)` plumbs the per-`internal_company_id`
sector lookup through to `select_candidates` (which preserves the new column)
and on to `inverse_vol_allocation`.

## Acceptance — `tests/test_sectors.py`

1. `sic_to_division` handles ints, strings, None.
2. `parse_sector_from_submissions` produces the canonical CIK + SIC fields.
3. With 4 names in sector 3 and 1 in sector 5, `max_sector_weight=0.5`
   correctly caps sector 3 at 50% and grows sector 5 to absorb the excess.
4. With `max_sector_weight=None`, sector caps are a no-op (weights sum to 1).
