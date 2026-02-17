"""IR edge types — SynapticEdge, ProbabilisticEdge, FeedbackEdge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SynapticEdge:
    """A synaptic connection in the IR.

    Parameters
    ----------
    id : str
        Unique identifier (from the API Connection).
    source_id : str
        ID of the pre-synaptic DynamicsNode.
    target_id : str
        ID of the post-synaptic DynamicsNode.
    pattern : str
        Connectivity pattern (e.g. "dense", "random_sparse").
    plasticity : str
        Learning rule (e.g. "none", "stdp").
    params : dict
        Pattern/plasticity-specific parameters.
    """

    id: str
    source_id: str
    target_id: str
    pattern: str
    plasticity: str = "none"
    params: dict[str, Any] = field(default_factory=dict)
