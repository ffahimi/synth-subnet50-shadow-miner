"""Polygon REST adapter for BTC 1-minute aggregate bars."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd

BASE_URL = "https://api.polygon.io"
PLACEHOLDER_API_KEYS = {
    "your_polygon_key_here",
    "your_valid_polygon_key",
    "your_polygon_key",
}
CANONICAL_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
]


@dataclass(frozen=True)
class PolygonFetchResult:
    bars: pd.DataFrame
    data_source: str = "polygon_1m_rest"


class Polygon1mClient:
    """Small Polygon aggregate-bar client.

    This is intentionally narrow so it can later be replaced with a local DB
    implementation returning the same `PolygonFetchResult`.
    """

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 30.0) -> None:
        self.api_key = resolve_api_key(api_key)
        self.timeout_seconds = timeout_seconds

    async def fetch_recent(
        self,
        ticker: str,
        end: pd.Timestamp,
        lookback_days: int = 14,
    ) -> PolygonFetchResult:
        start = _as_utc(end) - timedelta(days=lookback_days)
        bars = await self.fetch_range(ticker=ticker, start=start, end=end)
        return PolygonFetchResult(bars=bars)

    async def fetch_range(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        url = (
            f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/"
            f"1/minute/{start_utc.strftime('%Y-%m-%d')}/{end_utc.strftime('%Y-%m-%d')}"
        )
        params: dict[str, Any] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        results: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            while url:
                response = await client.get(url, params=params)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise httpx.HTTPStatusError(
                        sanitized_http_error_message(response),
                        request=exc.request,
                        response=response,
                    ) from exc
                payload = response.json()
                status = payload.get("status")
                if status not in {"OK", "DELAYED"}:
                    raise RuntimeError(f"Polygon request failed with status={status}: {payload}")
                results.extend(payload.get("results") or [])
                next_url = payload.get("next_url")
                if next_url:
                    url = next_url if str(next_url).startswith("http") else f"{BASE_URL}{next_url}"
                    params = {"apiKey": self.api_key}
                else:
                    url = ""

        bars = polygon_results_to_bars(results)
        bars = bars[(bars["timestamp"] >= start_utc) & (bars["timestamp"] <= end_utc)]
        return repair_missing_1m_bars(bars).reset_index(drop=True)


def resolve_api_key(api_key: str | None = None) -> str:
    key = (api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if key and key not in PLACEHOLDER_API_KEYS:
        return key
    raise ValueError("POLYGON_API_KEY is required in the environment or .env; do not commit it.")


def polygon_results_to_bars(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        raise ValueError("Polygon returned no aggregate bars.")
    df = pd.DataFrame(results).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for column in CANONICAL_COLUMNS:
        if column not in df:
            df[column] = 0.0
    df = df[CANONICAL_COLUMNS].sort_values("timestamp").drop_duplicates("timestamp")
    return df.reset_index(drop=True)


def repair_missing_1m_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        raise ValueError("Cannot repair an empty bar dataframe.")
    indexed = bars.set_index("timestamp").sort_index()
    full_index = pd.date_range(indexed.index.min(), indexed.index.max(), freq="60s", tz="UTC")
    repaired = indexed.reindex(full_index)
    repaired["close"] = repaired["close"].ffill()
    for column in ["open", "high", "low", "vwap"]:
        repaired[column] = repaired[column].fillna(repaired["close"])
    repaired["volume"] = repaired["volume"].fillna(0.0)
    repaired["transactions"] = repaired["transactions"].fillna(0.0)
    return repaired.reset_index(names="timestamp")[CANONICAL_COLUMNS]


def sanitized_http_error_message(response: httpx.Response) -> str:
    parsed = urlparse(str(response.url))
    safe_url = parsed._replace(query="").geturl()
    return f"{response.status_code} error from Polygon for url: {safe_url}"


def _as_utc(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

