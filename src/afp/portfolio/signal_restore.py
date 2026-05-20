"""Signal restoration helpers — atanh wrapper that broadcasts across DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afp.targets.target_transform import TargetConfig, restore


def add_restored_returns(predictions: pd.DataFrame, k: float = 2.5,
                         restore_clip_abs: float = 0.99) -> pd.DataFrame:
    """Add `restored_expected_return` column. Requires `predicted_signal` and `ex_ante_scale`."""
    cfg = TargetConfig(k=k, restore_method="atanh", restore_clip_abs=restore_clip_abs)
    out = predictions.copy()
    valid = out["ex_ante_scale"].notna() & (out["ex_ante_scale"] > 0)
    out["restored_expected_return"] = np.nan
    if valid.any():
        out.loc[valid, "restored_expected_return"] = restore(
            out.loc[valid, "predicted_signal"].to_numpy(),
            out.loc[valid, "ex_ante_scale"].to_numpy(),
            cfg,
        )
    return out
