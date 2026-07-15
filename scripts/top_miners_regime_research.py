#!/usr/bin/env python3
"""Research persistent Synth leaders against market regimes.

This script is analysis tooling, not part of the live miner loop. It:

1. Fetches Synth historical scores in API-friendly chunks.
2. Finds miners that persistently appear in the top N by CRPS.
3. Fetches market bars for the same window when Polygon analysis is enabled.
4. Builds score-time market-state features from 1-minute bars.
5. Buckets days/sessions into volatility, vol-of-vol, direction, and trend regimes.
6. Writes repeatable CSV/JSON outputs for research and comparison.

Keep ``POLYGON_API_KEY`` in the environment. Do not put it in this file or git.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
load_dotenv(REPO_ROOT / ".env")

from synth_shadow.assets import apply_asset  # noqa: E402
from synth_shadow.config import load_config  # noqa: E402
from synth_shadow.data.polygon_client import PolygonClient  # noqa: E402
from synth_shadow.data.schema import repair_missing_bars  # noqa: E402
from synth_shadow.features.pipeline import build_feature_frame  # noqa: E402
from synth_shadow.synth.client import SynthClient  # noqa: E402
from synth_shadow.utils.time import utc_now  # noqa: E402

DEFAULT_ASSETS = ("BTC", "ETH")
DEFAULT_EQUITY_ASSETS = ("XAU",)
EQUITY_COMPETITION = "com-equ-24h"
EQUITY_TICKER_OVERRIDES = {
    "XAU": "C:XAUUSD",
}
DEFAULT_SYNTH_EQUITY_CANDIDATES = (
    "XAU",
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "NFLX",
    "AMD",
    "MSTR",
    "COIN",
)
FEATURE_WINDOWS_HOURS = (1, 3, 5, 8, 24)
SESSION_NAMES = ("eu", "eu_us_overlap", "us", "outside_market_hours", "weekend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to Synth shadow config.")
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS), help="Assets to analyze.")
    parser.add_argument(
        "--equities",
        action="store_true",
        help="Analyze configured Synth equity/commodity assets. Defaults to XAU unless --assets is set.",
    )
    parser.add_argument(
        "--discover-synth-equities",
        action="store_true",
        help="Probe Synth prompt coverage for common equity symbols and print active assets.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Exit after --discover-synth-equities instead of running miner analysis.",
    )
    parser.add_argument(
        "--synth-equity-candidates",
        nargs="+",
        default=list(DEFAULT_SYNTH_EQUITY_CANDIDATES),
        help="Candidate assets to probe when --discover-synth-equities is set.",
    )
    parser.add_argument(
        "--polygon-minute-smoke",
        action="store_true",
        help="Fetch a small 1-minute Polygon sample for each selected asset before analysis.",
    )
    parser.add_argument(
        "--polygon-smoke-lookback-hours",
        type=int,
        default=72,
        help="Lookback hours for --polygon-minute-smoke. Use enough hours to cover equity market closures.",
    )
    parser.add_argument(
        "--polygon-smoke-only",
        action="store_true",
        help="Exit after --polygon-minute-smoke instead of running miner analysis.",
    )
    parser.add_argument("--days", type=int, default=90, help="Lookback window in calendar days.")
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=47,
        help="Historical score chunk size. Keep below the Synth API window limit.",
    )
    parser.add_argument("--top-n", type=int, default=25, help="Persistent leader cohort size.")
    parser.add_argument(
        "--synth-timeout-seconds",
        type=int,
        default=90,
        help="Timeout for Synth API requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries per Synth historical score chunk for transient network errors.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=5.0,
        help="Base sleep between retries. Backoff is attempt * this value.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reports/top_miners_research",
        help="Directory for generated CSV/JSON research outputs.",
    )
    parser.add_argument(
        "--skip-polygon",
        action="store_true",
        help="Only fetch/rank Synth scores; skip Polygon regime features.",
    )
    parser.add_argument(
        "--feature-tolerance-minutes",
        type=int,
        default=10,
        help="Backward merge tolerance when matching score times to market features.",
    )
    return parser.parse_args()


def load_asset_config(config_path: str | Path, asset: str) -> dict[str, Any]:
    selected = asset.upper()
    base = load_config(str(REPO_ROOT / config_path))
    try:
        return apply_asset(base, selected)
    except ValueError:
        return dynamic_equity_config(base, selected)


def dynamic_equity_config(base_config: dict[str, Any], asset: str) -> dict[str, Any]:
    """Build a research-only config for Synth equity symbols absent from default.yaml."""
    cfg = copy.deepcopy(base_config)
    cfg["asset"] = asset
    cfg["polygon_ticker"] = EQUITY_TICKER_OVERRIDES.get(asset, asset)
    cfg.setdefault("assets", {})[asset] = {
        "polygon_ticker": cfg["polygon_ticker"],
        "synth_asset": asset,
        "competition": EQUITY_COMPETITION,
    }
    cfg.setdefault("synth", {})
    cfg["synth"]["asset"] = asset
    cfg["synth"]["competition"] = EQUITY_COMPETITION
    return cfg


def quantile_bucket(
    values: pd.Series,
    labels: tuple[str, ...] = ("low", "mid", "high"),
) -> pd.Series:
    ranked = values.rank(method="first")
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels)
    except ValueError:
        return pd.Series(["mid"] * len(values), index=values.index)


def fetch_historical_scores(
    asset: str,
    config_path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_hours: int,
    timeout_seconds: int,
    max_retries: int,
    retry_sleep_seconds: float,
) -> pd.DataFrame:
    cfg = load_asset_config(config_path, asset)
    client = SynthClient(cfg, timeout_seconds=timeout_seconds)
    rows: list[dict[str, Any]] = []
    cur = start
    chunk_index = 0

    while cur < end:
        chunk_index += 1
        chunk_end = min(cur + pd.Timedelta(hours=chunk_hours), end)
        chunk = _fetch_score_chunk_with_retries(
            client=client,
            asset=asset,
            chunk_index=chunk_index,
            start=cur,
            end=chunk_end,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        rows.extend(chunk)
        cur = chunk_end

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["miner_uid", "asset", "time_length", "scored_time"])
    df = df[df["crps"].notna()].copy()
    df["crps"] = pd.to_numeric(df["crps"], errors="coerce")
    df = df[np.isfinite(df["crps"]) & (df["crps"] >= 0)].copy()
    df["miner_uid"] = df["miner_uid"].astype(int)
    df["scored_time"] = pd.to_datetime(df["scored_time"], utc=True)
    df["date"] = df["scored_time"].dt.date.astype(str)
    df["crps_rank"] = df.groupby("scored_time")["crps"].rank(
        method="min",
        ascending=True,
    ).astype(int)
    return df


def _fetch_score_chunk_with_retries(
    client: SynthClient,
    asset: str,
    chunk_index: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int,
    retry_sleep_seconds: float,
) -> list[dict[str, Any]]:
    for attempt in range(1, max_retries + 1):
        try:
            chunk = client.historical_scores(start=start, end=end)
            print(
                f"{asset} scores chunk {chunk_index}: {start} -> {end}, rows={len(chunk)}",
                flush=True,
            )
            return chunk
        except requests.HTTPError as exc:
            print(
                f"{asset} scores chunk {chunk_index} HTTP failed "
                f"attempt={attempt}/{max_retries}: {exc}",
                flush=True,
            )
            if attempt == max_retries:
                return []
        except requests.RequestException as exc:
            print(
                f"{asset} scores chunk {chunk_index} network failed "
                f"attempt={attempt}/{max_retries}: {exc}",
                flush=True,
            )
            if attempt == max_retries:
                return []
        time.sleep(retry_sleep_seconds * attempt)
    return []


def latest_reward_weights(config_path: str | Path, asset: str) -> pd.DataFrame:
    cfg = load_asset_config(config_path, asset)
    rows = SynthClient(cfg).rewards_scores()
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        uid = int(row["miner_uid"])
        updated_at = str(row.get("updated_at") or "")
        if uid not in latest or updated_at > str(latest[uid].get("updated_at") or ""):
            latest[uid] = row

    df = pd.DataFrame(latest.values())
    if df.empty:
        return pd.DataFrame(columns=["miner_uid", "reward_weight", "reward_rank", "updated_at"])

    df["miner_uid"] = df["miner_uid"].astype(int)
    df["reward_weight"] = pd.to_numeric(df["reward_weight"], errors="coerce").fillna(0.0)
    df["reward_rank"] = df["reward_weight"].rank(method="min", ascending=False).astype(int)
    return df[["miner_uid", "reward_weight", "reward_rank", "updated_at"]]


def discover_synth_assets(
    config_path: str | Path,
    candidates: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout_seconds: int,
) -> pd.DataFrame:
    """Probe Synth prompt coverage for candidate assets."""
    rows: list[dict[str, Any]] = []
    for asset in [item.upper() for item in candidates]:
        try:
            cfg = load_asset_config(config_path, asset)
        except ValueError:
            cfg = load_config(str(REPO_ROOT / config_path))
            cfg = {**cfg, "asset": asset, "synth": {**cfg["synth"], "asset": asset}}
        try:
            starts = SynthClient(cfg, timeout_seconds=timeout_seconds).prompts(start=start, end=end)
            rows.append(
                {
                    "asset": asset,
                    "prompt_count": len(starts),
                    "first_prompt": starts[0] if starts else None,
                    "last_prompt": starts[-1] if starts else None,
                    "active": bool(starts),
                    "error": None,
                }
            )
        except requests.HTTPError as exc:
            rows.append(
                {
                    "asset": asset,
                    "prompt_count": 0,
                    "first_prompt": None,
                    "last_prompt": None,
                    "active": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def polygon_minute_smoke(
    asset: str,
    config_path: str | Path,
    end: pd.Timestamp,
    lookback_hours: int = 6,
) -> dict[str, Any]:
    """Fetch a tiny 1-minute Polygon sample to confirm subscription coverage."""
    cfg = load_asset_config(config_path, asset)
    start = end - pd.Timedelta(hours=lookback_hours)
    try:
        bars = PolygonClient().fetch_aggregates(
            ticker=cfg["polygon_ticker"],
            multiplier=1,
            timespan="minute",
            start=start,
            end=end,
            adjusted=bool(cfg["history"].get("adjusted", True)),
        )
        diffs = bars["timestamp"].diff().dt.total_seconds().dropna()
        one_minute_gap_rate = float(diffs.eq(60.0).mean()) if not diffs.empty else None
        min_spacing = float(diffs.min()) if not diffs.empty else None
        median_spacing = float(diffs.median()) if not diffs.empty else None
        error = None
    except Exception as exc:  # noqa: BLE001 - smoke test should report failures without hiding other assets.
        bars = pd.DataFrame(columns=["timestamp"])
        one_minute_gap_rate = None
        min_spacing = None
        median_spacing = None
        error = str(exc)
    return {
        "asset": asset,
        "polygon_ticker": cfg["polygon_ticker"],
        "rows": int(len(bars)),
        "first_timestamp": bars["timestamp"].min().isoformat() if not bars.empty else None,
        "last_timestamp": bars["timestamp"].max().isoformat() if not bars.empty else None,
        "min_spacing_seconds": min_spacing,
        "median_spacing_seconds": median_spacing,
        "one_minute_gap_rate": one_minute_gap_rate,
        "has_minute_resolution": bool(min_spacing == 60.0 or one_minute_gap_rate == 1.0)
        if min_spacing is not None
        else False,
        "error": error,
    }


def fetch_polygon_features(
    asset: str,
    config_path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    cfg = load_asset_config(config_path, asset)
    raw = PolygonClient().fetch_aggregates(
        ticker=cfg["polygon_ticker"],
        multiplier=int(cfg["history"]["bar_multiplier"]),
        timespan=cfg["history"]["bar_timespan"],
        start=start - pd.Timedelta(days=2),
        end=end + pd.Timedelta(days=1),
        adjusted=bool(cfg["history"].get("adjusted", True)),
    )
    bars = repair_missing_bars(raw, int(cfg["forecast"]["interval_seconds"]))
    features = build_feature_frame(bars, cfg)
    features = features[(features["timestamp"] >= start) & (features["timestamp"] <= end)].copy()
    features["date"] = features["timestamp"].dt.date.astype(str)
    features["log_return_abs"] = features["log_return"].abs()
    return features


def fetch_polygon_minute_state_features(
    asset: str,
    config_path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch 1-minute Polygon bars and build momentum/vol/vol-of-vol features."""
    cfg = load_asset_config(config_path, asset)
    raw = PolygonClient().fetch_aggregates(
        ticker=cfg["polygon_ticker"],
        multiplier=1,
        timespan="minute",
        start=start - pd.Timedelta(hours=max(FEATURE_WINDOWS_HOURS) + 2),
        end=end + pd.Timedelta(minutes=10),
        adjusted=bool(cfg["history"].get("adjusted", True)),
    )
    bars = repair_missing_bars(raw, 60)
    features = build_market_state_features(bars)
    features = features[(features["timestamp"] >= start) & (features["timestamp"] <= end)].copy()
    features["date"] = features["timestamp"].dt.date.astype(str)
    return features.reset_index(drop=True)


def build_market_state_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build requested 1/3/5/8/24h momentum, volatility, and vol-of-vol."""
    out = bars.sort_values("timestamp").reset_index(drop=True).copy()
    log_close = np.log(out["close"].astype(float))
    out["log_return_1m"] = log_close.diff()
    squared = out["log_return_1m"] ** 2
    for hours in FEATURE_WINDOWS_HOURS:
        minutes = hours * 60
        suffix = f"{hours}h"
        out[f"momentum_{suffix}"] = log_close - log_close.shift(minutes)
        out[f"realized_vol_{suffix}"] = np.sqrt(squared.rolling(minutes, min_periods=minutes).sum())
        out[f"vol_of_vol_{suffix}"] = (
            out[f"realized_vol_{suffix}"].diff().rolling(minutes, min_periods=max(30, minutes // 2)).std()
        )
    return out


def score_feature_join(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    tolerance_minutes: int,
) -> pd.DataFrame:
    """Match each score snapshot to the market state at forecast origin time."""
    left = scores.sort_values("feature_timestamp").copy()
    right = features.sort_values("timestamp").copy()
    joined = pd.merge_asof(
        left,
        right,
        left_on="feature_timestamp",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    joined["feature_match_lag_minutes"] = (
        (joined["feature_timestamp"] - joined["timestamp"]).dt.total_seconds() / 60.0
    )
    return joined


def trend_efficiency(group: pd.DataFrame) -> float:
    denominator = float(group["log_return_abs"].sum())
    if denominator == 0:
        return float("nan")
    total_log_return = float(np.log(group["close"].iloc[-1] / group["close"].iloc[0]))
    return abs(total_log_return) / denominator


def daily_market_regimes(features: pd.DataFrame) -> pd.DataFrame:
    daily = (
        features.groupby("date")
        .apply(
            lambda group: pd.Series(
                {
                    "daily_return": float(group["close"].iloc[-1] / group["close"].iloc[0] - 1),
                    "daily_abs_return": float(abs(group["close"].iloc[-1] / group["close"].iloc[0] - 1)),
                    "daily_realized_vol": float(group["log_return"].std() * np.sqrt(len(group))),
                    "daily_vol_of_vol": float(group["vol_1h"].std()),
                    "daily_trend_efficiency": trend_efficiency(group),
                    "daily_kurtosis_mean": float(group["kurtosis_4h"].mean()),
                    "daily_momentum_4h_mean": float(group["momentum_4h"].mean()),
                }
            )
        )
        .reset_index()
    )
    daily["vol_regime"] = quantile_bucket(daily["daily_realized_vol"]).astype(str)
    daily["vol_of_vol_regime"] = quantile_bucket(daily["daily_vol_of_vol"]).astype(str)
    daily["trend_regime"] = quantile_bucket(
        daily["daily_trend_efficiency"],
        labels=("mean_reverting", "mixed", "trending"),
    ).astype(str)
    daily["direction_regime"] = np.select(
        [daily["daily_return"] > 0.005, daily["daily_return"] < -0.005],
        ["bullish", "bearish"],
        default="flat",
    )
    return daily


def session_regimes(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (date, session), group in features.groupby(["date", "session"]):
        rows.append(
            {
                "date": date,
                "session": session,
                "session_return": float(group["close"].iloc[-1] / group["close"].iloc[0] - 1),
                "session_realized_vol": float(group["log_return"].std() * np.sqrt(len(group))),
                "session_vol_of_vol": float(group["vol_1h"].std()),
                "session_trend_efficiency": trend_efficiency(group),
                "session_momentum_4h_mean": float(group["momentum_4h"].mean()),
                "bars": int(len(group)),
            }
        )

    session_df = pd.DataFrame(rows)
    for session in session_df["session"].unique():
        mask = session_df["session"] == session
        session_df.loc[mask, "session_vol_regime"] = quantile_bucket(
            session_df.loc[mask, "session_realized_vol"]
        ).astype(str)
        session_df.loc[mask, "session_trend_regime"] = quantile_bucket(
            session_df.loc[mask, "session_trend_efficiency"],
            labels=("mean_reverting", "mixed", "trending"),
        ).astype(str)

    wide = session_df.pivot(
        index="date",
        columns="session",
        values=["session_realized_vol", "session_trend_efficiency"],
    )
    wide.columns = [f"{session}_{metric}" for metric, session in wide.columns]
    return session_df, wide.reset_index()


def daily_miner_performance(scores: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_baseline = (
        scores.groupby("date")
        .agg(day_median_crps=("crps", "median"), day_best_crps=("crps", "min"), day_obs=("crps", "size"))
        .reset_index()
    )
    daily_miner = (
        scores.groupby(["date", "miner_uid"])
        .agg(
            daily_mean_crps=("crps", "mean"),
            daily_median_crps=("crps", "median"),
            daily_best_rank=("crps_rank", "min"),
            daily_topn_hits=("crps_rank", lambda ranks: int((ranks <= top_n).sum())),
            daily_obs=("crps", "size"),
        )
        .reset_index()
        .merge(daily_baseline, on="date", how="left")
    )
    daily_miner["crps_advantage_vs_day_median"] = (
        daily_miner["day_median_crps"] - daily_miner["daily_mean_crps"]
    )
    daily_miner["topn_day"] = daily_miner["daily_topn_hits"] > 0
    return daily_miner, daily_baseline


def miner_consistency(
    daily_miner: pd.DataFrame,
    rewards: pd.DataFrame,
) -> pd.DataFrame:
    total_days = daily_miner["date"].nunique()
    summary = (
        daily_miner.groupby("miner_uid")
        .agg(
            days_seen=("date", "nunique"),
            topn_days=("topn_day", "sum"),
            mean_daily_crps=("daily_mean_crps", "mean"),
            median_daily_crps=("daily_mean_crps", "median"),
            median_daily_best_rank=("daily_best_rank", "median"),
            mean_advantage_vs_day_median=("crps_advantage_vs_day_median", "mean"),
        )
        .reset_index()
    )
    summary["topn_day_rate"] = summary["topn_days"] / total_days
    summary = summary.merge(rewards, on="miner_uid", how="left")
    summary["reward_weight"] = summary["reward_weight"].fillna(0.0)
    return summary.sort_values(
        ["topn_day_rate", "mean_advantage_vs_day_median"],
        ascending=[False, False],
    )


def cohort_daily_performance(
    daily_miner: pd.DataFrame,
    cohort_uids: list[int],
) -> pd.DataFrame:
    return (
        daily_miner[daily_miner["miner_uid"].isin(cohort_uids)]
        .groupby("date")
        .agg(
            cohort_mean_crps=("daily_mean_crps", "mean"),
            cohort_median_best_rank=("daily_best_rank", "median"),
            cohort_topn_miners=("topn_day", "sum"),
            cohort_advantage_vs_day_median=("crps_advantage_vs_day_median", "mean"),
        )
        .reset_index()
    )


def grouped_performance(joined: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        joined.groupby(group_col)
        .agg(
            days=("date", "nunique"),
            avg_advantage=("cohort_advantage_vs_day_median", "mean"),
            median_advantage=("cohort_advantage_vs_day_median", "median"),
            avg_topn_miners=("cohort_topn_miners", "mean"),
            avg_best_rank=("cohort_median_best_rank", "mean"),
            avg_crps=("cohort_mean_crps", "mean"),
        )
        .reset_index()
        .sort_values("avg_advantage", ascending=False)
    )


def session_vol_performance(
    session_detail: pd.DataFrame,
    cohort_daily: pd.DataFrame,
) -> pd.DataFrame:
    return (
        session_detail.merge(
            cohort_daily[["date", "cohort_advantage_vs_day_median", "cohort_topn_miners"]],
            on="date",
            how="left",
        )
        .groupby(["session", "session_vol_regime"])
        .agg(
            days=("date", "nunique"),
            avg_advantage=("cohort_advantage_vs_day_median", "mean"),
            avg_topn_miners=("cohort_topn_miners", "mean"),
        )
        .reset_index()
        .sort_values(["session", "avg_advantage"], ascending=[True, False])
    )


def composite_session_conditions(joined: pd.DataFrame) -> pd.DataFrame:
    empty = pd.Series(index=joined.index, dtype=float)
    conditions = {
        "eu_low_us_high_vol": (
            joined.get("eu_session_realized_vol", empty).rank(pct=True) <= 1 / 3
        )
        & (joined.get("us_session_realized_vol", empty).rank(pct=True) >= 2 / 3),
        "outside_low_us_high_vol": (
            joined.get("outside_market_hours_session_realized_vol", empty).rank(pct=True) <= 1 / 3
        )
        & (joined.get("us_session_realized_vol", empty).rank(pct=True) >= 2 / 3),
        "eu_high_us_low_vol": (
            joined.get("eu_session_realized_vol", empty).rank(pct=True) >= 2 / 3
        )
        & (joined.get("us_session_realized_vol", empty).rank(pct=True) <= 1 / 3),
    }

    rows: list[dict[str, Any]] = []
    for name, values in conditions.items():
        for condition_value, group in joined.assign(condition=values).groupby("condition"):
            rows.append(
                {
                    "condition": name,
                    "value": bool(condition_value),
                    "days": int(group["date"].nunique()),
                    "avg_advantage": float(group["cohort_advantage_vs_day_median"].mean()),
                    "avg_topn_miners": float(group["cohort_topn_miners"].mean()),
                    "avg_best_rank": float(group["cohort_median_best_rank"].mean()),
                }
            )
    return pd.DataFrame(rows)


def feature_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "daily_realized_vol",
        "daily_vol_of_vol",
        "daily_trend_efficiency",
        "daily_kurtosis_mean",
        "daily_momentum_4h_mean",
    ]
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        subset = joined[[col, "cohort_advantage_vs_day_median", "cohort_topn_miners"]].dropna()
        if len(subset) >= 5 and subset[col].std() > 0:
            rows.append(
                {
                    "feature": col,
                    "corr_advantage": float(subset[col].corr(subset["cohort_advantage_vs_day_median"])),
                    "corr_topn_miners": float(subset[col].corr(subset["cohort_topn_miners"])),
                }
            )
    return pd.DataFrame(rows).sort_values("corr_advantage", ascending=False)


def market_state_feature_columns() -> list[str]:
    columns: list[str] = []
    for hours in FEATURE_WINDOWS_HOURS:
        suffix = f"{hours}h"
        columns.extend([f"momentum_{suffix}", f"realized_vol_{suffix}", f"vol_of_vol_{suffix}"])
    return columns


def feature_bucket_performance(score_features: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_cols = market_state_feature_columns()
    for feature in feature_cols:
        if feature not in score_features:
            continue
        usable = score_features[["miner_uid", "scored_time", "crps", "crps_rank", feature]].dropna()
        if len(usable) < 10 or usable[feature].nunique() < 3:
            continue
        usable = usable.assign(feature_bucket=quantile_bucket(usable[feature]).astype(str))
        for bucket, group in usable.groupby("feature_bucket"):
            rows.append(
                {
                    "feature": feature,
                    "bucket": bucket,
                    "score_rows": int(len(group)),
                    "snapshots": int(group["scored_time"].nunique()),
                    "miners": int(group["miner_uid"].nunique()),
                    "mean_crps": float(group["crps"].mean()),
                    "median_crps": float(group["crps"].median()),
                    "mean_rank": float(group["crps_rank"].mean()),
                    "topn_rate": float(group["crps_rank"].le(top_n).mean()),
                }
            )
    return pd.DataFrame(rows)


def miner_feature_consistency(score_features: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_cols = market_state_feature_columns()
    for feature in feature_cols:
        if feature not in score_features:
            continue
        usable = score_features[["miner_uid", "scored_time", "crps", "crps_rank", feature]].dropna()
        if len(usable) < 10 or usable[feature].nunique() < 3:
            continue
        usable = usable.assign(feature_bucket=quantile_bucket(usable[feature]).astype(str))
        for (miner_uid, bucket), group in usable.groupby(["miner_uid", "feature_bucket"]):
            rows.append(
                {
                    "miner_uid": int(miner_uid),
                    "feature": feature,
                    "bucket": str(bucket),
                    "score_rows": int(len(group)),
                    "snapshots": int(group["scored_time"].nunique()),
                    "mean_crps": float(group["crps"].mean()),
                    "median_rank": float(group["crps_rank"].median()),
                    "topn_rate": float(group["crps_rank"].le(top_n).mean()),
                }
            )
    return pd.DataFrame(rows)


def score_feature_correlations(score_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in market_state_feature_columns():
        if feature not in score_features:
            continue
        subset = score_features[[feature, "crps", "crps_rank"]].dropna()
        if len(subset) >= 20 and subset[feature].std() > 0:
            rows.append(
                {
                    "feature": feature,
                    "rows": int(len(subset)),
                    "corr_crps": float(subset[feature].corr(subset["crps"])),
                    "corr_rank": float(subset[feature].corr(subset["crps_rank"])),
                }
            )
    return pd.DataFrame(rows).sort_values("corr_crps") if rows else pd.DataFrame(rows)


def analyze_asset(
    asset: str,
    args: argparse.Namespace,
    rewards: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
) -> dict[str, Any]:
    print(f"\n=== {asset} ===", flush=True)
    scores = fetch_historical_scores(
        asset=asset,
        config_path=args.config,
        start=start,
        end=end,
        chunk_hours=args.chunk_hours,
        timeout_seconds=args.synth_timeout_seconds,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    if scores.empty:
        raise RuntimeError(f"No valid historical scores returned for {asset}.")
    horizon = pd.Timedelta(seconds=int(load_asset_config(args.config, asset)["synth"]["time_length"]))
    scores["feature_timestamp"] = scores["scored_time"] - horizon

    daily_miner, daily_baseline = daily_miner_performance(scores, args.top_n)
    consistency = miner_consistency(daily_miner, rewards)
    cohort_uids = consistency.head(args.top_n)["miner_uid"].astype(int).tolist()
    cohort_daily = cohort_daily_performance(daily_miner, cohort_uids)

    scores.to_csv(output_dir / f"{asset.lower()}_historical_scores.csv", index=False)
    daily_miner.to_csv(output_dir / f"{asset.lower()}_daily_miner_crps.csv", index=False)
    daily_baseline.to_csv(output_dir / f"{asset.lower()}_daily_baseline.csv", index=False)
    consistency.to_csv(output_dir / f"{asset.lower()}_miner_consistency.csv", index=False)

    result: dict[str, Any] = {
        "asset": asset,
        "score_rows": int(len(scores)),
        "snapshots": int(scores["scored_time"].nunique()),
        "days": int(scores["date"].nunique()),
        "top_n": int(args.top_n),
        "cohort_uids": cohort_uids,
        "top_consistent": consistency.head(15).to_dict("records"),
    }

    if args.skip_polygon:
        return result

    minute_features = fetch_polygon_minute_state_features(
        asset,
        args.config,
        scores["feature_timestamp"].min().floor("min"),
        scores["feature_timestamp"].max().ceil("min"),
    )
    score_features = score_feature_join(
        scores=scores,
        features=minute_features,
        tolerance_minutes=int(args.feature_tolerance_minutes),
    )
    feature_buckets = feature_bucket_performance(score_features, args.top_n)
    miner_feature = miner_feature_consistency(score_features, args.top_n)
    score_corr = score_feature_correlations(score_features)

    minute_features.to_csv(output_dir / f"{asset.lower()}_market_state_features_1m.csv", index=False)
    score_features.to_csv(output_dir / f"{asset.lower()}_score_feature_rows.csv", index=False)
    feature_buckets.to_csv(output_dir / f"{asset.lower()}_feature_bucket_performance.csv", index=False)
    miner_feature.to_csv(output_dir / f"{asset.lower()}_miner_feature_consistency.csv", index=False)
    score_corr.to_csv(output_dir / f"{asset.lower()}_score_feature_correlations.csv", index=False)
    result["score_feature_analysis"] = {
        "feature_rows": int(len(minute_features)),
        "score_feature_rows": int(len(score_features)),
        "score_feature_matched_rows": int(score_features["timestamp"].notna().sum()),
        "features": market_state_feature_columns(),
        "feature_correlations": score_corr.to_dict("records"),
    }

    features = fetch_polygon_features(asset, args.config, start.floor("D"), end.ceil("D"))
    daily_regimes = daily_market_regimes(features)
    session_detail, session_wide = session_regimes(features)
    joined = cohort_daily.merge(daily_regimes, on="date", how="left").merge(
        session_wide,
        on="date",
        how="left",
    )

    features.to_csv(output_dir / f"{asset.lower()}_full_features_5m.csv", index=False)
    daily_regimes.to_csv(output_dir / f"{asset.lower()}_daily_regimes.csv", index=False)
    session_detail.to_csv(output_dir / f"{asset.lower()}_session_regimes.csv", index=False)
    joined.to_csv(output_dir / f"{asset.lower()}_regime_joined_performance.csv", index=False)

    for column in ["vol_regime", "vol_of_vol_regime", "direction_regime", "trend_regime"]:
        grouped_performance(joined, column).to_csv(
            output_dir / f"{asset.lower()}_{column}_performance.csv",
            index=False,
        )
    session_perf = session_vol_performance(session_detail, cohort_daily)
    composite = composite_session_conditions(joined)
    correlations = feature_correlations(joined)
    session_perf.to_csv(output_dir / f"{asset.lower()}_session_vol_performance.csv", index=False)
    composite.to_csv(output_dir / f"{asset.lower()}_session_composite_performance.csv", index=False)
    correlations.to_csv(output_dir / f"{asset.lower()}_feature_correlations.csv", index=False)

    result["feature_correlations"] = correlations.to_dict("records")
    result["regime_tables"] = {
        column: grouped_performance(joined, column).to_dict("records")
        for column in ["vol_regime", "vol_of_vol_regime", "direction_regime", "trend_regime"]
    }
    result["session_vol_performance"] = session_perf.to_dict("records")
    result["session_composite_performance"] = composite.to_dict("records")
    return result


def main() -> None:
    args = parse_args()
    smoke_only = bool(args.polygon_minute_smoke and args.polygon_smoke_only)
    if not args.skip_polygon and not smoke_only and not os.getenv("POLYGON_API_KEY"):
        raise SystemExit("POLYGON_API_KEY is required unless --skip-polygon is used.")

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(utc_now()).floor("min")
    start = end - pd.Timedelta(days=int(args.days))

    if args.discover_synth_equities:
        discovered = discover_synth_assets(
            config_path=args.config,
            candidates=args.synth_equity_candidates,
            start=end - pd.Timedelta(days=1),
            end=end,
            timeout_seconds=args.synth_timeout_seconds,
        )
        discovered.to_csv(output_dir / "synth_equity_prompt_coverage.csv", index=False)
        print(discovered.to_string(index=False))
        if args.discover_only:
            print(f"\nSaved Synth equity coverage probe to {output_dir}")
            return

    selected_assets = [asset.upper() for asset in args.assets]
    if args.equities and selected_assets == list(DEFAULT_ASSETS):
        selected_assets = list(DEFAULT_EQUITY_ASSETS)

    if args.polygon_minute_smoke and not args.skip_polygon:
        smoke_rows = []
        for asset in selected_assets:
            smoke = polygon_minute_smoke(
                asset,
                args.config,
                end=end,
                lookback_hours=int(args.polygon_smoke_lookback_hours),
            )
            print(f"polygon_minute_smoke {smoke}", flush=True)
            smoke_rows.append(smoke)
        pd.DataFrame(smoke_rows).to_csv(output_dir / "polygon_minute_smoke.csv", index=False)
        if args.polygon_smoke_only:
            print(f"\nSaved Polygon minute smoke to {output_dir}")
            return

    summaries = []
    for asset in selected_assets:
        rewards = latest_reward_weights(args.config, asset)
        rewards.to_csv(output_dir / f"{asset.lower()}_latest_reward_weights.csv", index=False)
        summaries.append(analyze_asset(asset, args, rewards, start, end, output_dir))

    summary = {
        "generated_at": end.isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": int(args.days),
        "assets": selected_assets,
        "top_n": int(args.top_n),
        "output_dir": str(output_dir),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str)[:8000])
    print(f"\nSaved top-miner regime research outputs to {output_dir}")


if __name__ == "__main__":
    main()
