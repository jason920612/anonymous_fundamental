"""Time-decay + RMT-barbell allocator (Phase 63).

Wraps Phase 54 winner by pre-applying exponential signal-age decay
before the allocation. First-principles: signal informativeness
decays after a filing (τ = 63d), so positions held late in the
window should get less weight.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from afp.portfolio.allocation import AllocationResult, PortfolioConfig
from afp.portfolio.risk_models import VolEstimator
from afp.portfolio.rmt_barbell_allocator import build_rmt_barbell_target_portfolio
from afp.portfolio.signal_decay import TimeDecayConfig, apply_time_decay


def build_decay_rmt_barbell_target_portfolio(
    as_of: date,
    predictions_with_context: pd.DataFrame,
    vol_estimator: VolEstimator,
    cfg: PortfolioConfig,
    k: float = 2.5,
    sector_map: dict[str, str] | None = None,
    decay_cfg: TimeDecayConfig | None = None,
    safe_fraction: float = 0.80,
    concentrated_top_k: int = 5,
    risk_aversion: float = 5.0,
) -> AllocationResult:
    decay_cfg = decay_cfg or TimeDecayConfig(enabled=True, tau_days=63.0,
                                              decay_mode="exponential",
                                              min_floor=0.10)
    decayed = apply_time_decay(predictions_with_context, as_of, decay_cfg)
    return build_rmt_barbell_target_portfolio(
        as_of, decayed, vol_estimator, cfg, k=k, sector_map=sector_map,
        safe_fraction=safe_fraction, concentrated_top_k=concentrated_top_k,
        risk_aversion=risk_aversion,
    )
