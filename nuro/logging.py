"""Nuro logging configuration."""

from __future__ import annotations

import logging

# Create the nuro logger hierarchy
logger = logging.getLogger("nuro")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure Nuro logging with a console handler.

    Parameters
    ----------
    level : int
        Logging level. Default ``logging.INFO``.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[nuro] %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
