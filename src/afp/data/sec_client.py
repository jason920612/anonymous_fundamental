"""Minimal SEC HTTP client with rate-limiting and retry."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from afp.utils.logging import get_logger

log = get_logger(__name__)


class PermanentSecError(RuntimeError):
    """Raised on 4xx (except 429) so callers can skip the CIK without retrying."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class SecClient:
    """Tiny client over `data.sec.gov` and `www.sec.gov`. RFC-01 §3.1."""

    def __init__(
        self,
        user_agent: str,
        max_requests_per_second: int = 8,
        retry_attempts: int = 5,
        retry_backoff_seconds: tuple[float, ...] = (1, 2, 4, 8, 16),
        timeout: float = 30.0,
    ) -> None:
        if not user_agent or "example" in user_agent.lower() and "@" not in user_agent:
            log.warning("SEC user agent looks like a placeholder; SEC blocks anonymous calls.")
        self.user_agent = user_agent
        self._min_interval = 1.0 / max(max_requests_per_second, 1)
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff_seconds
        self.timeout = timeout
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        delta = time.monotonic() - self._last_request
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        self._last_request = time.monotonic()

    def get(self, url: str) -> bytes:
        """GET with retry on transient failures only.

        4xx (except 429) are permanent and short-circuit immediately — no point
        burning ~30s of exponential backoff on a 404 that will always be 404.
        """
        last_exc: Exception | None = None
        for attempt in range(self.retry_attempts):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code in (429, 502, 503, 504):
                    raise requests.HTTPError(f"transient {resp.status_code}", response=resp)
                if 400 <= resp.status_code < 500:
                    raise PermanentSecError(
                        f"permanent {resp.status_code} from SEC: {url}",
                        status_code=resp.status_code,
                    )
                resp.raise_for_status()
            except PermanentSecError:
                raise
            except Exception as exc:
                last_exc = exc
                backoff = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                log.warning("sec_get_retry", extra={"url": url, "attempt": attempt + 1,
                                                    "backoff_s": backoff, "err": str(exc)})
                time.sleep(backoff)
        raise RuntimeError(f"SEC request failed after {self.retry_attempts} attempts: {url}") from last_exc

    def get_json(self, url: str) -> Any:
        return json.loads(self.get(url).decode("utf-8"))

    # ---- High-level helpers (URL templates from RFC-01 §3.1) ----

    def company_tickers(self) -> Any:
        return self.get_json("https://www.sec.gov/files/company_tickers.json")

    def submissions(self, cik10: str) -> Any:
        return self.get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json")

    def submissions_full(self, cik10: str) -> list[dict]:
        """Fetch the primary submissions JSON plus every paginated archive file.

        SEC's ``submissions/CIK<cik>.json`` only contains the most recent ~1000
        filings under ``filings.recent``; older filings are listed in
        ``filings.files[]`` and must be fetched separately. Returns a list of
        per-page dicts each containing a ``filings.recent`` block. RFC-01 §3.1.
        """
        primary = self.submissions(cik10)
        pages = [primary]
        files = (primary.get("filings") or {}).get("files") or []
        for f in files:
            name = f.get("name")
            if not name:
                continue
            try:
                page = self.get_json(f"https://data.sec.gov/submissions/{name}")
            except Exception:
                continue
            pages.append({"cik": cik10, "filings": {"recent": page}})
        return pages

    def company_facts(self, cik10: str) -> Any:
        return self.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

    # ------------------------------------------------------------------

    def download_to(self, url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.get(url))
        return destination
