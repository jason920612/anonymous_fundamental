"""Date helpers used across phases."""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def to_eastern_date(value: datetime | str | None) -> date | None:
    """Convert UTC or naive datetime/ISO string to an America/New_York calendar date."""
    if value is None:
        return None
    if isinstance(value, str):
        # SEC accepted datetimes are typically `YYYY-MM-DDTHH:MM:SS` in Eastern.
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return date.fromisoformat(value[:10])
    elif isinstance(value, datetime):
        dt = value
    else:
        raise TypeError(f"unsupported type for date conversion: {type(value)}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=EASTERN)
    return dt.astimezone(EASTERN).date()


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
