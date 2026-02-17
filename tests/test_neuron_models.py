"""Tests for Izhikevich and AdEx neuron models."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.population import Population
from nuro.backends.gpu.backend import GPUBackend
from nuro.backends.gpu.neurons import AdExNode, IzhikevichNode
from nuro.ir import IRGraph


class TestIzhikevichNode:
    def test_basic_forward(self):
        node = IzhikevichNode(size=10, dt=1e-3)
        x = torch.ones(10)
        spikes = node(x)
        assert spikes.shape == (10,)

    def test_fires_with_strong_input(self):
        node = IzhikevichNode(size=10, dt=1e-3)
        total_spikes = 0
        for _ in range(200):
            spikes = node(torch.ones(10))
            total_spikes += int(spikes.sum().item())
        assert total_spikes > 0

    def test_preset(self):
        params = IzhikevichNode.PRESETS["fast_spiking"]
        node = IzhikevichNode(size=5, **params, dt=1e-3)
        assert node.a == 0.1

    def test_reset(self):
        node = IzhikevichNode(size=5, dt=1e-3)
        node(torch.ones(5))
        node.reset()
        assert torch.allclose(node.v, torch.full((5,), node.c))


class TestAdExNode:
    def test_basic_forward(self):
        node = AdExNode(size=10, dt=1e-3)
        x = torch.ones(10)
        spikes = node(x)
        assert spikes.shape == (10,)

    def test_fires_with_strong_input(self):
        node = AdExNode(size=10, dt=1e-3)
        total_spikes = 0
        for _ in range(500):
            spikes = node(torch.ones(10) * 2.0)
            total_spikes += int(spikes.sum().item())
        assert total_spikes > 0

    def test_reset(self):
        node = AdExNode(size=5, dt=1e-3)
        node(torch.ones(5))
        node.reset()
        assert torch.allclose(node.v, torch.full((5,), node.E_L))


class TestNeuronIntegration:
    def test_izhikevich_network(self):
        p1 = Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="izhikevich")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["num_steps"] == 100
        assert model.metrics["total_spikes"] >= 0

    def test_izhikevich_preset(self):
        p1 = Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="izhikevich", params={"preset": "fast_spiking"})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["total_spikes"] >= 0

    def test_adex_network(self):
        p1 = Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="adex")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["num_steps"] == 100
        assert model.metrics["total_spikes"] >= 0

    def test_all_izhikevich_network(self):
        """Both populations using Izhikevich dynamics."""
        p1 = Population(size=50, dynamics="izhikevich")
        p2 = Population(size=10, dynamics="izhikevich", params={"preset": "chattering"})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["total_spikes"] >= 0
