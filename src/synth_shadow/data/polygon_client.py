"""Polygon BTC bar fetcher.

This is the only live market-data adapter in the first prototype. Later, this
module can be replaced by a database-backed Polygon mirror while preserving the
same output schema.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from urllib.parse import urlparse

import pandas as pd
import requests

from synth_shadow.data.schema import polygon_results_to_bars
from synth_shadow.utils.time import utc_now

LOG = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"


class PolygonClient:
    """Small Polygon REST client for aggregate bars."""

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 30) -> None:
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY is required in the environment or .env file.")

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
            response.raise_for_status()
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

    def fetch_recent_btc(self, config: dict) -> pd.DataFrame:
        """Fetch recent BTC bars using the project config."""
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

    @staticmethod
    def _without_key(url: str) -> str:
        parsed = urlparse(url)
        return parsed._replace(query="").geturl()
