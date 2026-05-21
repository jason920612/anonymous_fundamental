"""Michaelis-Menten signal saturation (Phase 59).

Michaelis-Menten kinetics (1913): the rate of an enzymatic reaction
as a function of substrate concentration [S] is

    v = V_max · [S] / (K_m + [S])

- For [S] ≪ K_m: rate is linear in concentration (rate ≈ V_max/K_m · S).
- For [S] ≫ K_m: rate saturates to V_max.
- K_m is the substrate concentration at half-maximum rate.

Adapted to portfolio signals: treat each positive_signal as a
"substrate concentration." Pass it through the M-M transform:

    μ_sat = signal / (K_m + signal)

with K_m = median of |signal| across the candidate universe (an
adaptive, parameter-free choice). The transformation:

- Preserves rank order (monotone).
- Damps extreme signals to ≤ 1.
- Linear regime for typical signals (below K_m).
- Saturating regime for outliers.

This is a NON-FINANCIAL physics/biochemistry idea applied to
allocator preprocessing. Single-pass evaluation; nothing tuned.
"""

from __future__ import annotations

import numpy as np


def michaelis_menten(signal: np.ndarray, Km: float | None = None,
                     Vmax: float = 1.0) -> np.ndarray:
    """Apply M-M saturation. Non-negative inputs only (clips at 0)."""
    sig = np.clip(signal.astype(float), 0.0, None)
    if Km is None:
        # adaptive: Km = median of positive (non-zero) signals
        nz = sig[sig > 1e-9]
        Km = float(np.median(nz)) if len(nz) > 0 else 1e-6
    Km = max(Km, 1e-9)
    return Vmax * sig / (Km + sig)
