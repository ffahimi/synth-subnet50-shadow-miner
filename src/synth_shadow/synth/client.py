"""Public Synth API client."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pandas as pd
import requests

from synth_shadow.utils.time import utc_now

LOG = logging.getLogger(__name__)


class SynthClient:
    """Small client for Synth validation, score, reward, and leaderboard APIs."""

    def __init__(self, config: dict, timeout_seconds: int = 30) -> None:
        self.config = config
        self.base_url = config["synth"]["base_url"].rstrip("/")
        self.timeout_seconds = timeout_seconds

    def prompts(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> list[str]:
        synth_cfg = self.config["synth"]
        now = pd.Timestamp(utc_now())
        start = start or now - timedelta(days=float(synth_cfg["prompt_lookback_days"]))
        end = end or now + timedelta(hours=float(synth_cfg["prompt_forward_hours"]))
        payload = self._get(
            "/validation/prompts",
            {
                "from": _iso(start),
                "to": _iso(end),
                "asset": synth_cfg["asset"],
                "time_increment": synth_cfg["time_increment"],
                "time_length": synth_cfg["time_length"],
            },
        )
        start_times = payload.get("start_times", [])
        LOG.debug("Synth prompts fetched count=%s", len(start_times))
        return sorted(str(value) for value in start_times)

    def realized_path(self, start_time: str) -> dict[str, Any]:
        synth_cfg = self.config["synth"]
        payload = self._get(
            "/validation/realized-path",
            {
                "start_time": start_time,
                "asset": synth_cfg["asset"],
                "time_increment": synth_cfg["time_increment"],
                "time_length": synth_cfg["time_length"],
            },
        )
        LOG.debug(
            "Synth realized path fetched start_time=%s points=%s",
            start_time,
            len(payload.get("real_prices", [])),
        )
        return payload

    def latest_scores(self) -> list[dict[str, Any]]:
        synth_cfg = self.config["synth"]
        payload = self._get(
            "/validation/scores/latest",
            {
                "asset": synth_cfg["asset"],
                "time_increment": synth_cfg["time_increment"],
                "time_length": synth_cfg["time_length"],
            },
        )
        rows = _as_list(payload)
        LOG.debug("Synth latest scores fetched count=%s", len(rows))
        return rows

    def historical_scores(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        synth_cfg = self.config["synth"]
        now = pd.Timestamp(utc_now())
        start = start or now - timedelta(days=float(synth_cfg["score_lookback_days"]))
        end = end or now
        payload = self._get(
            "/validation/scores/historical",
            {
                "from": _iso(start),
                "to": _iso(end),
                "asset": synth_cfg["asset"],
                "time_increment": synth_cfg["time_increment"],
                "time_length": synth_cfg["time_length"],
            },
        )
        rows = _as_list(payload)
        LOG.debug("Synth historical scores fetched count=%s", len(rows))
        return rows

    def rewards_scores(self) -> list[dict[str, Any]]:
        synth_cfg = self.config["synth"]
        now = pd.Timestamp(utc_now())
        start = now - timedelta(days=min(7, int(synth_cfg["score_lookback_days"])))
        payload = self._get(
            "/rewards/scores",
            {
                "from": _iso(start),
                "to": _iso(now),
                "competition": synth_cfg["competition"],
            },
        )
        rows = _as_list(payload)
        LOG.debug("Synth rewards scores fetched count=%s", len(rows))
        return rows

    def latest_leaderboard(self) -> list[dict[str, Any]]:
        payload = self._get(
            "/v2/leaderboard/latest",
            {"prompt_name": self.config["synth"]["competition"]},
        )
        rows = _as_list(payload)
        LOG.debug("Synth latest leaderboard fetched count=%s", len(rows))
        return rows

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        LOG.debug("Requesting Synth API path=%s params=%s", path, params)
        response = requests.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()


def _iso(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _as_list(payload: Any) -> list[dict[str, Any]]:
    return payload if isinstance(payload, list) else []
