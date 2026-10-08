"""Structured logging configuration using Loguru."""

import sys
from loguru import logger
from veritas_crag.config.settings import settings


def setup_logger() -> None:
    """Configures loguru logger with formatting and levels."""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.LOG_LEVEL,
        colorize=True,
    )


setup_logger()

__all__ = ["logger", "setup_logger"]
