"""Command line entrypoint for local shadow runs."""

from __future__ import annotations

import argparse
import json
import os

from synth_shadow.config import load_config
from synth_shadow.inspection.forecast import inspect_forecast
from synth_shadow.models.btc_generator import run_btc_forecast
from synth_shadow.orchestration.shadow_cycle import (
    fetch_benchmarks,
    generate_for_latest_prompt,
    run_shadow_cycle,
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
            "sync-prompts",
            "generate-latest-prompt",
            "inspect-latest",
            "score-forecast",
            "score-matured",
            "benchmarks",
            "run-synth-shadow",
        ],
        help="Shadow workflow command to run.",
    )
    parser.add_argument("--config", default=os.getenv("SYNTH_SHADOW_CONFIG", "config/default.yaml"))
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and debug metadata.")
    parser.add_argument("--forecast-dir", help="Forecast directory for inspect/score commands.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.debug:
        config["debug"] = True
    configure_logging(debug=bool(config.get("debug", False)))

    if args.command in {"fetch-btc", "generate-btc"}:
        result = run_btc_forecast(config)
        print(json.dumps(result, indent=2))
        return

    if args.command == "sync-prompts":
        result = sync_prompts(config)
    elif args.command == "generate-latest-prompt":
        result = generate_for_latest_prompt(config)
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
    elif args.command == "run-synth-shadow":
        result = run_shadow_cycle(config)
    else:
        raise SystemExit(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
