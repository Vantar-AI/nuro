"""Tests for STDP plasticity."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

import nuro
from nuro.backends.gpu.backend import GPUBackend
from nuro.ir import IRGraph


class TestSTDP:
    def test_stdp_runs(self):
        p1 = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = nuro.Connection(
            source=p1, target=p2,
            pattern="dense",
            plasticity="stdp",
        )
        graph = nuro.Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        backend = GPUBackend()
        model = backend.compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["total_spikes"] >= 0

    def test_stdp_modifies_weights(self):
        p1 = nuro.Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = nuro.Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = nuro.Connection(
            source=p1, target=p2,
            pattern="dense",
            plasticity="stdp",
        )
        graph = nuro.Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        backend = GPUBackend()
        model = backend.compile(ir)

        key = f"{p1.id}__{p2.id}"
        weights_before = model._snn.synapses[key].weight.clone()
        model.run(duration=0.5, dt=1e-3)
        weights_after = model._snn.synapses[key].weight.clone()

        # Weights should have been modified by STDP (unless no spikes at all)
        if model.metrics["total_spikes"] > 0:
            assert not torch.equal(weights_before, weights_after)

    def test_stdp_weights_clamped(self):
        p1 = nuro.Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = nuro.Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = nuro.Connection(
            source=p1, target=p2,
            pattern="dense",
            plasticity="stdp",
        )
        graph = nuro.Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        backend = GPUBackend()
        model = backend.compile(ir)
        model.run(duration=0.5, dt=1e-3)

        key = f"{p1.id}__{p2.id}"
        weights = model._snn.synapses[key].weight
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0
