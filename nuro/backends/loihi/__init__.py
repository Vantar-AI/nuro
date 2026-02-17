"""Loihi backend — Intel Loihi 2 deployment via Lava SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuro.backends.loihi.backend import LoihiBackend, LoihiCompiledModel

__all__ = ["LoihiBackend", "LoihiCompiledModel"]
