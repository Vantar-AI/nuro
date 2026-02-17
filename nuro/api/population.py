"""Population primitives — the atomic unit of Nuro computation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

SUPPORTED_DYNAMICS = {"lif", "if"}


@dataclass
class Population:
    """A population of neurons with shared dynamics.

    Parameters
    ----------
    size : int
        Number of neurons in the population.
    dynamics : str
        Neuron dynamics model. Supported: "lif", "if".
    params : dict
        Model-specific parameters (e.g. {"tau": 20e-3} for LIF).
    """

    size: int
    dynamics: str = "lif"
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], repr=False)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"Population size must be >= 1, got {self.size}")
        if self.dynamics not in SUPPORTED_DYNAMICS:
            raise ValueError(
                f"Unsupported dynamics '{self.dynamics}'. "
                f"Supported: {sorted(SUPPORTED_DYNAMICS)}"
            )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Population):
            return NotImplemented
        return self.id == other.id
