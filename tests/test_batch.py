"""Tests for batch support (v0.3.0)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.input import Input
from nuro.api.population import Population
from nuro.backends.gpu.backend import GPUBackend
from nuro.ir import IRGraph


class TestBatchBackwardCompat:
    """batch_size=1 (default) must behave identically to v0.2.0."""

    def test_default_batch_size_is_one(self):
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3)

        assert model.metrics["num_steps"] == 50
        assert model.metrics["total_spikes"] >= 0
        assert model.metrics["batch_size"] == 1

    def test_batch_one_recording_shape(self):
        """With batch_size=1, recorded shapes should be (steps, pop_size)."""
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.record("spikes", population=p2)
        model.run(duration=0.05, dt=1e-3)

        spikes = model.get_state("spikes", population=p2)
        assert spikes.shape == (50, 10)

    def test_batch_one_izh(self):
        """Izhikevich neurons with batch_size=1 still work."""
        p1 = Population(size=20, dynamics="izhikevich", params={"preset": "regular_spiking"})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        inp = Input(population=p1, mode="poisson", rate=100.0)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3)
        assert model.metrics["num_steps"] == 50

    def test_batch_one_adex(self):
        """AdEx neurons with batch_size=1 still work."""
        p1 = Population(size=20, dynamics="adex")
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        inp = Input(population=p1, mode="poisson", rate=100.0)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3)
        assert model.metrics["num_steps"] == 50


class TestBatchLIF:
    """Batched simulation with LIF neurons."""

    def test_batched_poisson(self):
        p1 = Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3, batch_size=8)

        assert model.metrics["num_steps"] == 50
        assert model.metrics["batch_size"] == 8
        assert model.metrics["total_spikes"] >= 0

    def test_batched_recording_shape(self):
        """Batched spikes should be (steps, batch, pop_size)."""
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.record("spikes", population=p2)
        model.run(duration=0.05, dt=1e-3, batch_size=4)

        spikes = model.get_state("spikes", population=p2)
        assert spikes.shape == (50, 4, 10)

    def test_batched_voltage_recording_shape(self):
        """Batched voltages should be (steps, batch, pop_size) for custom neurons."""
        p1 = Population(size=20, dynamics="izhikevich", params={"preset": "regular_spiking"})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        inp = Input(population=p1, mode="poisson", rate=100.0)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.record("voltages", population=p1)
        model.run(duration=0.05, dt=1e-3, batch_size=4)

        voltages = model.get_state("voltages", population=p1)
        assert voltages.shape == (50, 4, 20)

    def test_batched_weight_recording_unchanged(self):
        """Weights are shared across batch — shape stays (recordings, out, in)."""
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.record("weights", connection=conn, interval=10)
        model.run(duration=0.05, dt=1e-3, batch_size=4)

        weights = model.get_state("weights", connection=conn)
        assert weights.shape == (5, 10, 20)  # 50 steps / 10 interval = 5 recordings

    def test_batched_static_input(self):
        """Static tensor input with batch dim: (steps, batch, pop_size)."""
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        # (steps, batch, pop_size)
        data = torch.rand(50, 4, 10)
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3, batch_size=4)
        assert model.metrics["num_steps"] == 50

    def test_batched_generator_input(self):
        """Generator returning (batch, pop_size)."""
        batch = 4
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        inp = Input(population=p1, generator=lambda step: (torch.rand(batch, 10) < 0.1).float())
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3, batch_size=batch)
        assert model.metrics["num_steps"] == 50


class TestBatchIzhikevich:
    """Batched simulation with Izhikevich neurons."""

    def test_batched_izh(self):
        p1 = Population(size=20, dynamics="izhikevich", params={"preset": "regular_spiking"})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        inp = Input(population=p1, mode="poisson", rate=100.0)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3, batch_size=8)

        assert model.metrics["num_steps"] == 50
        assert model.metrics["batch_size"] == 8

    def test_batched_izh_presets(self):
        """All Izhikevich presets work with batching."""
        for preset in ["regular_spiking", "fast_spiking", "chattering"]:
            p1 = Population(size=10, dynamics="izhikevich", params={"preset": preset})
            p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
            conn = Connection(source=p1, target=p2, pattern="dense")
            inp = Input(population=p1, mode="poisson", rate=100.0)
            graph = Graph([p1, p2], [conn], inputs=[inp])
            ir = IRGraph.from_api_graph(graph)

            model = GPUBackend().compile(ir)
            model.run(duration=0.02, dt=1e-3, batch_size=4)
            assert model.metrics["num_steps"] == 20


class TestBatchAdEx:
    """Batched simulation with AdEx neurons."""

    def test_batched_adex(self):
        p1 = Population(size=20, dynamics="adex")
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        inp = Input(population=p1, mode="poisson", rate=100.0)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3, batch_size=8)

        assert model.metrics["num_steps"] == 50
        assert model.metrics["batch_size"] == 8


class TestBatchRecurrent:
    """Batched simulation with recurrent graphs."""

    def test_batched_recurrent_lif(self):
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=20, dynamics="lif", params={"tau": 10e-3})
        c1 = Connection(source=p1, target=p2, pattern="dense")
        c2 = Connection(source=p2, target=p1, pattern="dense")
        inp = Input(population=p1, mode="poisson", rate=100.0)
        graph = Graph([p1, p2], [c1, c2], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3, batch_size=4)

        assert model.metrics["num_steps"] == 50
        assert model.metrics["batch_size"] == 4


class TestBatchSTDP:
    """Batched simulation with STDP plasticity."""

    def test_batched_stdp(self):
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense", plasticity="stdp")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3, batch_size=4)

        assert model.metrics["num_steps"] == 50
        assert model.metrics["batch_size"] == 4


class TestBatchVariance:
    """Verify that different batch elements produce different spike patterns."""

    def test_batch_variance(self):
        """Different batch elements should have different spike counts (Poisson input)."""
        p1 = Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.record("spikes", population=p2)
        model.run(duration=0.1, dt=1e-3, batch_size=32)

        spikes = model.get_state("spikes", population=p2)  # (100, 32, 10)
        per_batch = spikes.sum(dim=(0, 2))  # (32,) spike counts per trial
        # With 32 trials, we expect some variance
        assert per_batch.std() > 0 or per_batch.sum() == 0
