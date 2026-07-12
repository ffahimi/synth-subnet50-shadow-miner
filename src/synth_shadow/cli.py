"""Command line entrypoint for local shadow runs."""

from __future__ import annotations

import argparse
import json
import os

from synth_shadow.assets import apply_asset
from synth_shadow.backtest.rolling import run_rolling_backtest
from synth_shadow.config import load_config
from synth_shadow.inspection.forecast import inspect_forecast
from synth_shadow.inspection.live_sanity import run_live_forecast_sanity
from synth_shadow.models.btc_generator import run_btc_forecast
from synth_shadow.orchestration.shadow_cycle import (
    fetch_benchmarks,
    generate_for_latest_prompt,
    generate_sanity_for_latest_prompt,
    run_shadow_cycle,
    run_shadow_cycle_sanity,
    sync_prompts,
)
from synth_shadow.scoring.evaluator import score_forecast_dir, score_matured_forecasts
from synth_shadow.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synth-shadow")
    parser.add_argument(
        "command",
        choices=[
            "fetch-btc",
            "generate-btc",
            "generate-forecast",
            "forecast-sanity",
            "sync-prompts",
            "generate-latest-prompt",
            "generate-sanity-latest-prompt",
            "inspect-latest",
            "score-forecast",
            "score-matured",
            "benchmarks",
            "backtest-rolling",
            "run-synth-shadow",
            "run-synth-shadow-sanity",
        ],
        help="Shadow workflow command to run.",
    )
    parser.add_argument("--config", default=os.getenv("SYNTH_SHADOW_CONFIG", "config/default.yaml"))
    parser.add_argument("--asset", choices=["BTC", "ETH", "XAU"], help="Asset to run: BTC, ETH, or XAU.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and debug metadata.")
    parser.add_argument("--forecast-dir", help="Forecast directory for inspect/score commands.")
    parser.add_argument("--num-paths", type=int, help="Override live forecast path count.")
    parser.add_argument("--backtest-days", type=float, help="Rolling backtest origin window in days.")
    parser.add_argument("--backtest-stride-minutes", type=int, help="Rolling backtest origin stride.")
    parser.add_argument("--backtest-max-origins", type=int, help="Limit backtest origins for quick checks.")
    parser.add_argument("--backtest-num-paths", type=int, help="Override simulated paths per backtest origin.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = apply_asset(load_config(args.config), args.asset)
    if args.debug:
        config["debug"] = True
    if args.num_paths is not None:
        config["forecast"]["num_paths"] = int(args.num_paths)
    configure_logging(debug=bool(config.get("debug", False)))

    if args.command in {"fetch-btc", "generate-btc", "generate-forecast"}:
        result = run_btc_forecast(config)
        print(json.dumps(result, indent=2))
        return

    if args.command == "forecast-sanity":
        result = run_live_forecast_sanity(config)
    elif args.command == "sync-prompts":
        result = sync_prompts(config)
    elif args.command == "generate-latest-prompt":
        result = generate_for_latest_prompt(config)
    elif args.command == "generate-sanity-latest-prompt":
        result = generate_sanity_for_latest_prompt(config)
    elif args.command == "inspect-latest":
        result = inspect_forecast(config, args.forecast_dir)
    elif args.command == "score-forecast":
        if not args.forecast_dir:
            raise SystemExit("--forecast-dir is required for score-forecast")
        result = score_forecast_dir(config, args.forecast_dir)
    elif args.command == "score-matured":
        result = score_matured_forecasts(config)
    elif args.command == "benchmarks":
        result = fetch_benchmarks(config)
    elif args.command == "backtest-rolling":
        result = run_rolling_backtest(
            config,
            days=args.backtest_days,
            stride_minutes=args.backtest_stride_minutes,
            max_origins=args.backtest_max_origins,
            num_paths=args.backtest_num_paths,
        )
    elif args.command == "run-synth-shadow":
        result = run_shadow_cycle(config)
    elif args.command == "run-synth-shadow-sanity":
        result = run_shadow_cycle_sanity(config)
    else:
        raise SystemExit(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
