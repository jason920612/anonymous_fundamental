"""Risk estimation: trailing daily vol + (optional) shrunk covariance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    lookback_days: int = 252
    min_daily_vol: float = 0.005
    max_daily_vol: float = 0.10
    covariance_shrinkage_lambda: float = 0.5


class VolEstimator:
    """Per-company trailing daily vol lookup keyed by `(internal_company_id, asof)`."""

    def __init__(self, prices: pd.DataFrame, config: RiskConfig):
        self.cfg = config
        self._by_company: dict[str, pd.Series] = {}
        for icid, grp in prices.sort_values(["internal_company_id", "date"]).groupby(
                "internal_company_id", sort=False):
            p = grp["adjusted_close"].astype(float).to_numpy()
            if len(p) < 2:
                continue
            log_ret = np.log(p[1:] / p[:-1])
            self._by_company[icid] = pd.Series(log_ret,
                                               index=pd.to_datetime(grp["date"].iloc[1:].to_numpy()))

    def vol_at(self, internal_company_id: str, asof: date) -> float | None:
        s = self._by_company.get(internal_company_id)
        if s is None:
            return None
        past = s.loc[s.index < pd.Timestamp(asof)]
        if len(past) < max(20, self.cfg.lookback_days // 4):
            return None
        window = past.tail(self.cfg.lookback_days)
        vol = float(window.std(ddof=1))
        if not np.isfinite(vol) or vol <= 0:
            return None
        return float(np.clip(vol, self.cfg.min_daily_vol, self.cfg.max_daily_vol))

    def covariance(self, ids: list[str], asof: date) -> np.ndarray | None:
        series: list[pd.Series] = []
        for icid in ids:
            s = self._by_company.get(icid)
            if s is None:
                return None
            past = s.loc[s.index < pd.Timestamp(asof)].tail(self.cfg.lookback_days)
            if len(past) < max(30, self.cfg.lookback_days // 4):
                return None
            series.append(past)
        df = pd.concat(series, axis=1, keys=ids).dropna()
        if len(df) < 30:
            return None
        cov = df.cov().to_numpy()
        diag = np.diag(np.diag(cov))
        lam = self.cfg.covariance_shrinkage_lambda
        return (1 - lam) * cov + lam * diag
