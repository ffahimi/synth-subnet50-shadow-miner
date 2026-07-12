"""Fetch recent Polygon BTC data into data/raw.

This script currently runs the full first-stage pipeline so raw and processed
data can be debugged alongside the forecast.
"""

import sys

from synth_shadow.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "fetch-btc")
    main()
