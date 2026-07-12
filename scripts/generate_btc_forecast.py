"""Generate a BTC 24h shadow forecast.
"""

import sys

from synth_shadow.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "generate-btc")
    main()
