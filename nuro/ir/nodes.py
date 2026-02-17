"""IR node types — DynamicsNode, StochasticNode, SensorNode, ActuatorNode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DynamicsNode:
    """A neuron population in the IR.

    Parameters
    ----------
    id : str
        Unique identifier (from the API Population).
    size : int
        Number of neurons.
    dynamics : str
        Dynamics model name (e.g. "lif", "if").
    params : dict
        Model-specific parameters.
    """

    id: str
    size: int
    dynamics: str
    params: dict[str, Any] = field(default_factory=dict)
