"""SEC client behavior tests — 404 fast-fail, 429 retry."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from afp.data.sec_client import PermanentSecError, SecClient


def _resp(status: int, body: bytes = b""):
    r = requests.Response()
    r.status_code = status
    r._content = body
    return r


def test_404_short_circuits_without_retry():
    cli = SecClient("test ua test@example.com", max_requests_per_second=1000,
                    retry_attempts=5, retry_backoff_seconds=(0, 0, 0, 0, 0))
    with patch.object(cli._session, "get", return_value=_resp(404)) as mock_get:
        with pytest.raises(PermanentSecError) as exc:
            cli.get("https://data.sec.gov/x")
        assert exc.value.status_code == 404
        assert mock_get.call_count == 1   # no retries


def test_429_triggers_retry():
    cli = SecClient("test ua test@example.com", max_requests_per_second=1000,
                    retry_attempts=3, retry_backoff_seconds=(0, 0, 0))
    with patch.object(cli._session, "get", return_value=_resp(429)) as mock_get:
        with pytest.raises(RuntimeError):
            cli.get("https://data.sec.gov/x")
        assert mock_get.call_count == 3


def test_200_returns_body():
    cli = SecClient("test ua test@example.com", max_requests_per_second=1000)
    with patch.object(cli._session, "get", return_value=_resp(200, b"ok")):
        assert cli.get("https://data.sec.gov/x") == b"ok"
