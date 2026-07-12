"""Run the prompt/forecast/score shadow loop.

This remains a placeholder until the first offline BTC generator is working.
"""

from synth_shadow.cli import main


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run-synth-shadow")
    main()
