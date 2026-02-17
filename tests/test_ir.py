"""Tests for the IR layer."""

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.population import Population
from nuro.ir import IRGraph
from nuro.ir.edges import SynapticEdge
from nuro.ir.nodes import DynamicsNode


class TestIRLowering:
    def test_lowering_node_count(self):
        p1 = Population(size=100, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="if")
        conn = Connection(source=p1, target=p2, pattern="dense", plasticity="stdp")
        graph = Graph([p1, p2], [conn])

        ir = IRGraph.from_api_graph(graph)
        assert ir.num_nodes == 2
        assert ir.num_edges == 1

    def test_lowering_preserves_params(self):
        p1 = Population(size=100, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="if")
        conn = Connection(source=p1, target=p2, pattern="random_sparse", plasticity="stdp")
        graph = Graph([p1, p2], [conn])

        ir = IRGraph.from_api_graph(graph)

        node = ir.nodes[p1.id]
        assert isinstance(node, DynamicsNode)
        assert node.dynamics == "lif"
        assert node.size == 100
        assert node.params["tau"] == 20e-3

        edge = ir.edges[0]
        assert isinstance(edge, SynapticEdge)
        assert edge.source_id == p1.id
        assert edge.target_id == p2.id
        assert edge.pattern == "random_sparse"
        assert edge.plasticity == "stdp"

    def test_lowering_ids_match(self):
        p1 = Population(size=50)
        p2 = Population(size=20)
        conn = Connection(source=p1, target=p2)
        graph = Graph([p1, p2], [conn])

        ir = IRGraph.from_api_graph(graph)
        assert p1.id in ir.nodes
        assert p2.id in ir.nodes
        assert ir.edges[0].source_id == p1.id
        assert ir.edges[0].target_id == p2.id

    def test_empty_graph(self):
        p = Population(size=10)
        graph = Graph([p], [])
        ir = IRGraph.from_api_graph(graph)
        assert ir.num_nodes == 1
        assert ir.num_edges == 0
