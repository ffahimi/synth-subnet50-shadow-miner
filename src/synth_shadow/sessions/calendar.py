"""Session classification for 24/7 BTC data.

BTC never closes, so these sessions are liquidity/regime buckets rather than
exchange trading sessions:

- weekend
- eu
- eu_us_overlap
- us
- outside_market_hours
"""

from __future__ import annotations

from datetime import time

import pandas as pd


def add_session_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add a `session` column to canonical BTC bars."""
    sessions = config["sessions"]
    out = df.copy()
    timestamps = out["timestamp"].dt.tz_convert(sessions.get("timezone", "UTC"))
    out["session"] = [classify_timestamp(ts, sessions) for ts in timestamps]
    return out


def classify_timestamp(ts: pd.Timestamp, sessions: dict) -> str:
    """Classify one timestamp into a BTC liquidity session."""
    if ts.weekday() >= 5:
        return "weekend"

    current = ts.time()
    for name in ["eu", "eu_us_overlap", "us"]:
        if _in_range(current, _parse_time(sessions[name]["start"]), _parse_time(sessions[name]["end"])):
            return name
    return "outside_market_hours"


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _in_range(current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current < end
    return current >= start or current < end
