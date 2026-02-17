"""Tests for the Graph builder."""

import pytest

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.population import Population


class TestGraph:
    def test_construction(self):
        p1 = Population(size=100)
        p2 = Population(size=10)
        conn = Connection(source=p1, target=p2)
        g = Graph([p1, p2], [conn])
        assert g.num_populations == 2
        assert g.num_connections == 1

    def test_empty_connections(self):
        p1 = Population(size=50)
        g = Graph([p1], [])
        assert g.num_populations == 1
        assert g.num_connections == 0

    def test_dangling_source_rejected(self):
        p1 = Population(size=100)
        p2 = Population(size=10)
        orphan = Population(size=5)
        conn = Connection(source=orphan, target=p2)
        with pytest.raises(ValueError, match="source.*not in graph"):
            Graph([p1, p2], [conn])

    def test_dangling_target_rejected(self):
        p1 = Population(size=100)
        p2 = Population(size=10)
        orphan = Population(size=5)
        conn = Connection(source=p1, target=orphan)
        with pytest.raises(ValueError, match="target.*not in graph"):
            Graph([p1, p2], [conn])

    def test_digraph_property(self):
        p1 = Population(size=100)
        p2 = Population(size=10)
        conn = Connection(source=p1, target=p2)
        g = Graph([p1, p2], [conn])
        assert len(g.digraph.nodes) == 2
        assert len(g.digraph.edges) == 1

    def test_multiple_connections(self):
        p1 = Population(size=100)
        p2 = Population(size=50)
        p3 = Population(size=10)
        c1 = Connection(source=p1, target=p2)
        c2 = Connection(source=p2, target=p3)
        g = Graph([p1, p2, p3], [c1, c2])
        assert g.num_populations == 3
        assert g.num_connections == 2
