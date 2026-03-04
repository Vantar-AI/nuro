"""Nuro Intermediate Representation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import networkx as nx

from nuro.ir.edges import SynapticEdge
from nuro.ir.nodes import DynamicsNode

if TYPE_CHECKING:
    from nuro.api.graph import Graph as APIGraph

__all__ = ["DynamicsNode", "IRGraph", "SynapticEdge"]


class IRGraph:
    """Lowered graph representation used by backends.

    Holds DynamicsNodes and SynapticEdges in a networkx DiGraph.
    """

    def __init__(self) -> None:
        self._digraph = nx.DiGraph()
        self.nodes: dict[str, DynamicsNode] = {}
        self.edges: list[SynapticEdge] = []
        self.inputs: list[Any] = []  # Carried from API Input objects

    @classmethod
    def from_api_graph(cls, graph: APIGraph) -> IRGraph:
        """Lower an API Graph into an IRGraph."""
        ir = cls()

        for pop in graph.populations:
            node = DynamicsNode(
                id=pop.id,
                size=pop.size,
                dynamics=pop.dynamics,
                params=dict(pop.params),
            )
            ir.nodes[node.id] = node
            ir._digraph.add_node(node.id, ir_node=node)

        for conn in graph.connections:
            edge = SynapticEdge(
                id=conn.id,
                source_id=conn.source.id,
                target_id=conn.target.id,
                pattern=conn.pattern,
                plasticity=conn.plasticity,
                delay=getattr(conn, "delay", 0.0),
                params=dict(conn.params),
            )
            ir.edges.append(edge)
            ir._digraph.add_edge(edge.source_id, edge.target_id, ir_edge=edge)

        # Carry input specs through to backends
        ir.inputs = list(getattr(graph, "inputs", []))

        return ir

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def is_cyclic(self) -> bool:
        """True if the graph contains at least one cycle."""
        return not nx.is_directed_acyclic_graph(self._digraph)

    def topological_order(self) -> list[str]:
        """Return node IDs in topological order. Raises if cyclic."""
        return list(nx.topological_sort(self._digraph))
