"""Command line entrypoint for local shadow runs."""

from __future__ import annotations

import argparse
import json
import os

from synth_shadow.config import load_config
from synth_shadow.models.btc_generator import run_btc_forecast
from synth_shadow.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synth-shadow")
    parser.add_argument(
        "command",
        choices=["fetch-btc", "generate-btc", "score-matured"],
        help="Shadow workflow command to run.",
    )
    parser.add_argument("--config", default=os.getenv("SYNTH_SHADOW_CONFIG", "config/default.yaml"))
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and debug metadata.")
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

    raise SystemExit("score-matured is planned for the next step.")


if __name__ == "__main__":
    main()
