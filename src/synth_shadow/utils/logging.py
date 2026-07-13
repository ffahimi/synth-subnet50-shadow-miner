"""Logging setup helpers."""

from __future__ import annotations

import logging
import os

RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def configure_logging(debug: bool = False) -> None:
    """Configure concise console logging."""
    level_name = os.getenv("LOG_LEVEL", "DEBUG" if debug else "INFO")
    level = logging.DEBUG if debug else getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def colored_debug(logger: logging.Logger, message: str, *args: object, color: str = CYAN) -> None:
    """Emit a colored debug line when debug logging is enabled."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s%s%s", color, message % args if args else message, RESET)
