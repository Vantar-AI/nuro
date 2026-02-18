"""SpiNNaker 2 synapse builder — maps IR SynapticEdge to snn.Projection."""

from __future__ import annotations

from typing import Any

import numpy as np

from nuro.ir.edges import SynapticEdge
from nuro.ir.nodes import DynamicsNode


def build_s2_projection(
    edge: SynapticEdge,
    source_node: DynamicsNode,
    target_node: DynamicsNode,
    pre_pop: Any,
    post_pop: Any,
    weight_matrix: np.ndarray | None = None,
    delay: int = 0,
) -> Any:
    """Build a py-spinnaker2 Projection from an IR SynapticEdge.

    Parameters
    ----------
    edge : SynapticEdge
        The IR edge describing the connection.
    source_node : DynamicsNode
        Source population IR node (for size).
    target_node : DynamicsNode
        Target population IR node (for size).
    pre_pop : snn.Population
        Source py-spinnaker2 Population.
    post_pop : snn.Population
        Target py-spinnaker2 Population.
    weight_matrix : np.ndarray, optional
        Shape ``(post, pre)``. When ``None``, random small weights are used.
    delay : int
        Synaptic delay in timesteps [0, 7]. Default 0.

    Returns
    -------
    snn.Projection
    """
    from spinnaker2 import snn
    from nuro.backends.spinnaker2.transfer import weights_to_connection_list

    if weight_matrix is None:
        weight_matrix = (
            np.random.randn(target_node.size, source_node.size).astype(np.float32)
            * 0.1
        )

    connections = weights_to_connection_list(weight_matrix, delay=delay)

    # Empty connection list — create a single zero-weight connection as placeholder
    # to satisfy py-spinnaker2 which requires at least one connection per Projection
    if not connections:
        connections = [[0, 0, 0.0, delay]]

    return snn.Projection(
        pre=pre_pop,
        post=post_pop,
        connections=connections,
    )
