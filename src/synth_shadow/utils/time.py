"""Timezone and timestamp helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def floor_to_interval(ts: pd.Timestamp, interval_seconds: int) -> pd.Timestamp:
    """Floor a pandas timestamp to an interval in seconds."""
    return ts.floor(f"{interval_seconds}s")
