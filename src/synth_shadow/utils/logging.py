"""Logging setup helpers."""

from __future__ import annotations

import logging
import os


def configure_logging(debug: bool = False) -> None:
    """Configure concise console logging."""
    level_name = os.getenv("LOG_LEVEL", "DEBUG" if debug else "INFO")
    level = logging.DEBUG if debug else getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
