"""Tests for recurrent (cyclic) graph support."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.population import Population
from nuro.backends.gpu.backend import GPUBackend
from nuro.ir import IRGraph


class TestGraphCycleDetection:
    def test_dag_is_not_cyclic(self):
        p1 = Population(size=10, dynamics="lif")
        p2 = Population(size=5, dynamics="lif")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        assert not graph.is_cyclic

    def test_self_loop_is_cyclic(self):
        p1 = Population(size=10, dynamics="lif")
        p2 = Population(size=5, dynamics="lif")
        conn_fwd = Connection(source=p1, target=p2, pattern="dense")
        conn_back = Connection(source=p2, target=p1, pattern="dense")
        graph = Graph([p1, p2], [conn_fwd, conn_back])
        assert graph.is_cyclic

    def test_three_node_cycle(self):
        p1 = Population(size=10, dynamics="lif")
        p2 = Population(size=10, dynamics="lif")
        p3 = Population(size=10, dynamics="lif")
        c1 = Connection(source=p1, target=p2, pattern="dense")
        c2 = Connection(source=p2, target=p3, pattern="dense")
        c3 = Connection(source=p3, target=p1, pattern="dense")
        graph = Graph([p1, p2, p3], [c1, c2, c3])
        assert graph.is_cyclic


class TestIRCycleDetection:
    def test_ir_is_cyclic(self):
        p1 = Population(size=10, dynamics="lif")
        p2 = Population(size=5, dynamics="lif")
        conn_fwd = Connection(source=p1, target=p2, pattern="dense")
        conn_back = Connection(source=p2, target=p1, pattern="dense")
        graph = Graph([p1, p2], [conn_fwd, conn_back])
        ir = IRGraph.from_api_graph(graph)
        assert ir.is_cyclic

    def test_ir_topological_order_dag(self):
        p1 = Population(size=10, dynamics="lif")
        p2 = Population(size=5, dynamics="lif")
        p3 = Population(size=5, dynamics="lif")
        c1 = Connection(source=p1, target=p2, pattern="dense")
        c2 = Connection(source=p2, target=p3, pattern="dense")
        graph = Graph([p1, p2, p3], [c1, c2])
        ir = IRGraph.from_api_graph(graph)
        order = ir.topological_order()
        assert order.index(p1.id) < order.index(p2.id)
        assert order.index(p2.id) < order.index(p3.id)


class TestRecurrentRun:
    def test_recurrent_two_pop(self):
        """Two populations with bidirectional connections (mutual inhibition)."""
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        conn_fwd = Connection(source=p1, target=p2, pattern="dense")
        conn_back = Connection(source=p2, target=p1, pattern="dense")
        graph = Graph([p1, p2], [conn_fwd, conn_back])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["num_steps"] == 100
        assert model.metrics["total_spikes"] >= 0

    def test_recurrent_three_pop_ring(self):
        """Three populations in a ring (A→B→C→A)."""
        pA = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        pB = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        pC = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        c1 = Connection(source=pA, target=pB, pattern="dense")
        c2 = Connection(source=pB, target=pC, pattern="dense")
        c3 = Connection(source=pC, target=pA, pattern="dense")
        graph = Graph([pA, pB, pC], [c1, c2, c3])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["num_steps"] == 100

    def test_dag_still_works(self):
        """Ensure DAG execution with topological sort still works."""
        p1 = Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=20, dynamics="lif", params={"tau": 10e-3})
        p3 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        c1 = Connection(source=p1, target=p2, pattern="dense")
        c2 = Connection(source=p2, target=p3, pattern="dense")
        graph = Graph([p1, p2, p3], [c1, c2])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["num_steps"] == 100
        assert model.metrics["total_spikes"] >= 0

    def test_recurrent_with_external_input(self):
        """Recurrent graph with user-provided Poisson input."""
        from nuro.api.input import Input

        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        conn_fwd = Connection(source=p1, target=p2, pattern="dense")
        conn_back = Connection(source=p2, target=p1, pattern="dense")
        inp = Input(population=p1, mode="poisson", rate=100.0)
        graph = Graph([p1, p2], [conn_fwd, conn_back], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["total_spikes"] >= 0
