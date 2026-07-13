#!/usr/bin/env python3
"""Research persistent Synth leaders against Polygon market regimes.

This script is analysis tooling, not part of the live miner loop. It:

1. Fetches Synth historical scores in API-friendly chunks.
2. Finds miners that persistently appear in the top N by CRPS.
3. Fetches Polygon 5m bars for the same window.
4. Rebuilds the public feature set used by the shadow model.
5. Buckets days/sessions into volatility, vol-of-vol, direction, and trend regimes.
6. Writes repeatable CSV/JSON outputs for research and comparison.

Keep ``POLYGON_API_KEY`` in the environment. Do not put it in this file or git.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from synth_shadow.assets import apply_asset  # noqa: E402
from synth_shadow.config import load_config  # noqa: E402
from synth_shadow.data.polygon_client import PolygonClient  # noqa: E402
from synth_shadow.data.schema import repair_missing_bars  # noqa: E402
from synth_shadow.features.pipeline import build_feature_frame  # noqa: E402
from synth_shadow.synth.client import SynthClient  # noqa: E402
from synth_shadow.utils.time import utc_now  # noqa: E402

DEFAULT_ASSETS = ("BTC", "ETH")
SESSION_NAMES = ("eu", "eu_us_overlap", "us", "outside_market_hours", "weekend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to Synth shadow config.")
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS), help="Assets to analyze.")
    parser.add_argument("--days", type=int, default=90, help="Lookback window in calendar days.")
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=47,
        help="Historical score chunk size. Keep below the Synth API window limit.",
    )
    parser.add_argument("--top-n", type=int, default=25, help="Persistent leader cohort size.")
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
    return parser.parse_args()


def load_asset_config(config_path: str | Path, asset: str) -> dict[str, Any]:
    return apply_asset(load_config(str(REPO_ROOT / config_path)), asset.upper())


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
) -> pd.DataFrame:
    cfg = load_asset_config(config_path, asset)
    client = SynthClient(cfg)
    rows: list[dict[str, Any]] = []
    cur = start
    chunk_index = 0

    while cur < end:
        chunk_index += 1
        chunk_end = min(cur + pd.Timedelta(hours=chunk_hours), end)
        try:
            chunk = client.historical_scores(start=cur, end=chunk_end)
            rows.extend(chunk)
            print(
                f"{asset} scores chunk {chunk_index}: {cur} -> {chunk_end}, rows={len(chunk)}",
                flush=True,
            )
        except requests.HTTPError as exc:
            print(f"{asset} scores chunk {chunk_index} failed: {exc}", flush=True)
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


def latest_reward_weights(config_path: str | Path) -> pd.DataFrame:
    cfg = load_asset_config(config_path, "BTC")
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


def analyze_asset(
    asset: str,
    args: argparse.Namespace,
    rewards: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
) -> dict[str, Any]:
    print(f"\n=== {asset} ===", flush=True)
    scores = fetch_historical_scores(asset, args.config, start, end, args.chunk_hours)
    if scores.empty:
        raise RuntimeError(f"No valid historical scores returned for {asset}.")

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
    if not args.skip_polygon and not os.getenv("POLYGON_API_KEY"):
        raise SystemExit("POLYGON_API_KEY is required unless --skip-polygon is used.")

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(utc_now()).floor("min")
    start = end - pd.Timedelta(days=int(args.days))

    rewards = latest_reward_weights(args.config)
    rewards.to_csv(output_dir / "latest_reward_weights.csv", index=False)

    summaries = []
    for asset in [asset.upper() for asset in args.assets]:
        summaries.append(analyze_asset(asset, args, rewards, start, end, output_dir))

    summary = {
        "generated_at": end.isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": int(args.days),
        "assets": [asset.upper() for asset in args.assets],
        "top_n": int(args.top_n),
        "output_dir": str(output_dir),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str)[:8000])
    print(f"\nSaved top-miner regime research outputs to {output_dir}")


if __name__ == "__main__":
    main()
