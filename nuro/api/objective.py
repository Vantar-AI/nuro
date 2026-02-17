"""Objectives — what the system should optimize, not how."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Objective:
    """Stub objective specification.

    This class is a placeholder. Objective-based optimization is planned
    for a future release.
    """

    type: str = "none"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        warnings.warn(
            "Objective is a stub and has no effect in v0.1. "
            "Objective-based optimization is planned for a future release.",
            stacklevel=2,
        )
