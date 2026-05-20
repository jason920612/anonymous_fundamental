"""Identifier helpers — CIK normalization and internal company ID generation."""

from __future__ import annotations

import re

_CIK_RE = re.compile(r"\d+")


def normalize_cik(raw: str | int) -> str:
    """Return a 10-digit zero-padded CIK string. RFC-01 §2.1."""
    if raw is None:
        raise ValueError("cik is None")
    if isinstance(raw, int):
        digits = str(raw)
    else:
        match = _CIK_RE.search(str(raw))
        if not match:
            raise ValueError(f"no digits in cik {raw!r}")
        digits = match.group(0)
    return digits.zfill(10)


def internal_company_id(cik: str | int) -> str:
    """Deterministic internal company id derived from CIK.

    Independent of ticker so ticker changes do not break joins. RFC-01 §2.1.
    """
    return f"COMP{normalize_cik(cik)}"
