"""Connection abstractions — synapses, probabilistic couplings, electrode mappings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nuro.api.population import Population

SUPPORTED_PATTERNS = {"dense", "random_sparse", "one_to_one", "conv1d", "distance_dependent"}
SUPPORTED_PLASTICITY = {"none", "stdp", "stdp_online"}


@dataclass
class Connection:
    """A connection between two populations.

    Parameters
    ----------
    source : Population
        Pre-synaptic population.
    target : Population
        Post-synaptic population.
    pattern : str
        Connectivity pattern. Supported: "dense", "random_sparse".
    plasticity : str
        Learning rule. Supported: "none", "stdp".
    params : dict
        Pattern/plasticity-specific parameters.
    """

    source: Population
    target: Population
    pattern: str = "dense"
    plasticity: str = "none"
    delay: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], repr=False)

    def __post_init__(self) -> None:
        if self.pattern not in SUPPORTED_PATTERNS:
            raise ValueError(
                f"Unsupported pattern '{self.pattern}'. "
                f"Supported: {sorted(SUPPORTED_PATTERNS)}"
            )
        if self.plasticity not in SUPPORTED_PLASTICITY:
            raise ValueError(
                f"Unsupported plasticity '{self.plasticity}'. "
                f"Supported: {sorted(SUPPORTED_PLASTICITY)}"
            )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            return NotImplemented
        return self.id == other.id
