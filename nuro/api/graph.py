"""Graph builder — assembles populations, connections, and objectives into a compilable graph."""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx

from nuro.api.connection import Connection
from nuro.api.population import Population


class Graph:
    """A computation graph of populations and connections.

    Parameters
    ----------
    populations : sequence of Population
        All neuron populations in the network.
    connections : sequence of Connection
        All connections between populations.
    objectives : sequence, optional
        Objective specifications (stub — not yet implemented).
    """

    def __init__(
        self,
        populations: Sequence[Population],
        connections: Sequence[Connection],
        objectives: Sequence[Any] | None = None,
    ) -> None:
        self.populations = list(populations)
        self.connections = list(connections)
        self.objectives = list(objectives) if objectives else []

        self._pop_ids = {p.id for p in self.populations}
        self._digraph = nx.DiGraph()

        for pop in self.populations:
            self._digraph.add_node(pop.id, population=pop)

        for conn in self.connections:
            if conn.source.id not in self._pop_ids:
                raise ValueError(
                    f"Connection source '{conn.source.id}' not in graph populations"
                )
            if conn.target.id not in self._pop_ids:
                raise ValueError(
                    f"Connection target '{conn.target.id}' not in graph populations"
                )
            self._digraph.add_edge(
                conn.source.id, conn.target.id, connection=conn
            )

    @property
    def num_populations(self) -> int:
        return len(self.populations)

    @property
    def num_connections(self) -> int:
        return len(self.connections)

    @property
    def digraph(self) -> nx.DiGraph:
        return self._digraph
