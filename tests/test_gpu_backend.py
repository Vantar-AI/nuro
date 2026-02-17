"""Tests for the GPU backend."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.population import Population
from nuro.backends.gpu.backend import GPUBackend
from nuro.ir import IRGraph


class TestGPUBackend:
    def test_compile_and_run(self):
        p1 = Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        backend = GPUBackend()
        model = backend.compile(ir)
        model.run(duration=0.1, dt=1e-3)

        assert model.metrics["num_steps"] == 100
        assert model.metrics["total_spikes"] >= 0
        assert p1.id in model.metrics["spike_counts"]
        assert p2.id in model.metrics["spike_counts"]

    def test_if_dynamics(self):
        p1 = Population(size=20, dynamics="if")
        p2 = Population(size=5, dynamics="if")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        backend = GPUBackend()
        model = backend.compile(ir)
        model.run(duration=0.05)
        assert model.metrics["num_steps"] == 50

    def test_random_sparse(self):
        p1 = Population(size=100, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(
            source=p1, target=p2,
            pattern="random_sparse",
            params={"sparsity": 0.9},
        )
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        backend = GPUBackend()
        model = backend.compile(ir)
        model.run(duration=0.05)
        assert model.metrics["total_spikes"] >= 0

    def test_reset(self):
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        backend = GPUBackend()
        model = backend.compile(ir)
        model.run(duration=0.05)
        assert model.metrics["total_spikes"] >= 0

        model.reset()
        assert model.metrics == {}
