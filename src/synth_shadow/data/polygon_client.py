"""Polygon aggregate bar fetcher.

This is the only live market-data adapter in the first prototype. Later, this
module can be replaced by a database-backed Polygon mirror while preserving the
same output schema.
"""

from __future__ import annotations

import getpass
import logging
import os
import sys
from datetime import timedelta
from urllib.parse import urlparse

import pandas as pd
import requests

from synth_shadow.data.schema import polygon_results_to_bars
from synth_shadow.utils.time import utc_now

LOG = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"
PLACEHOLDER_API_KEYS = {
    "your_polygon_key_here",
    "your_valid_polygon_key",
    "your_polygon_key",
}


class PolygonClient:
    """Small Polygon REST client for aggregate bars."""

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 30) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.timeout_seconds = timeout_seconds

    def fetch_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjusted: bool = True,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """Fetch aggregate bars from Polygon and return the canonical dataframe."""
        start_date = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(end).strftime("%Y-%m-%d")
        url = (
            f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{start_date}/{end_date}"
        )
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit,
            "apiKey": self.api_key,
        }
        results: list[dict] = []

        while url:
            safe_url = self._without_key(url)
            LOG.debug("Requesting Polygon aggregates: %s", safe_url)
            response = requests.get(url, params=params, timeout=self.timeout_seconds)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise requests.HTTPError(
                    _sanitized_http_error_message(response),
                    response=response,
                ) from exc
            payload = response.json()
            status = payload.get("status")
            if status not in {"OK", "DELAYED"}:
                raise RuntimeError(f"Polygon request failed: {payload}")

            page_results = payload.get("results", [])
            results.extend(page_results)
            LOG.debug(
                "Polygon page status=%s results=%s accumulated=%s",
                status,
                len(page_results),
                len(results),
            )

            next_url = payload.get("next_url")
            if next_url:
                url = next_url if next_url.startswith("http") else f"{BASE_URL}{next_url}"
                params = {"apiKey": self.api_key}
            else:
                url = ""

        df = polygon_results_to_bars(results)
        end_ts = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
        start_ts = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        LOG.debug("Polygon fetch returned %s bars after timestamp filtering.", len(df))
        return df.reset_index(drop=True)

    def fetch_recent(self, config: dict) -> pd.DataFrame:
        """Fetch recent bars using the active asset config."""
        now = pd.Timestamp(utc_now())
        lookback_days = int(config["history"]["lookback_days"])
        return self.fetch_aggregates(
            ticker=config["polygon_ticker"],
            multiplier=int(config["history"]["bar_multiplier"]),
            timespan=config["history"]["bar_timespan"],
            start=now - timedelta(days=lookback_days),
            end=now,
            adjusted=bool(config["history"].get("adjusted", True)),
        )

    def fetch_recent_btc(self, config: dict) -> pd.DataFrame:
        """Backward-compatible alias for the original BTC-only pipeline."""
        return self.fetch_recent(config)

    @staticmethod
    def _without_key(url: str) -> str:
        parsed = urlparse(url)
        return parsed._replace(query="").geturl()


def _resolve_api_key(api_key: str | None = None) -> str:
    key = (api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if key and key not in PLACEHOLDER_API_KEYS:
        return key

    if key in PLACEHOLDER_API_KEYS:
        LOG.warning("POLYGON_API_KEY is set to a placeholder value; prompting for a real key.")

    if sys.stdin.isatty():
        prompted = getpass.getpass("Enter POLYGON_API_KEY: ").strip()
        if prompted and prompted not in PLACEHOLDER_API_KEYS:
            os.environ["POLYGON_API_KEY"] = prompted
            return prompted

    raise ValueError(
        "POLYGON_API_KEY is required. Set it in the environment/.env, or run from an "
        "interactive terminal to be prompted. Do not commit the key to git."
    )


def _sanitized_http_error_message(response: requests.Response) -> str:
    reason = response.reason.decode("utf-8", "replace") if isinstance(response.reason, bytes) else response.reason
    safe_url = PolygonClient._without_key(response.url)
    if 400 <= response.status_code < 500:
        return f"{response.status_code} Client Error: {reason} for url: {safe_url}"
    if 500 <= response.status_code < 600:
        return f"{response.status_code} Server Error: {reason} for url: {safe_url}"
    return f"HTTP error {response.status_code}: {reason} for url: {safe_url}"
