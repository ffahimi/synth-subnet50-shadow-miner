"""Command line entrypoint for local shadow runs."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synth-shadow")
    parser.add_argument(
        "command",
        choices=["fetch-btc", "generate-btc", "score-matured"],
        help="Shadow workflow command to run.",
    )
    parser.add_argument("--config", default="config/default.yaml")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(f"{args.command} is scaffolded; implementation comes next.")


if __name__ == "__main__":
    main()
