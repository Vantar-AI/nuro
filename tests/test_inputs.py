"""Tests for the user input system."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.input import Input
from nuro.api.population import Population
from nuro.backends.gpu.backend import GPUBackend
from nuro.ir import IRGraph


class TestInputAPI:
    def test_poisson_default(self):
        pop = Population(size=10, dynamics="lif")
        inp = Input(population=pop)
        assert inp.mode == "poisson"
        assert inp.rate == 50.0
        assert inp.id == pop.id

    def test_poisson_explicit(self):
        pop = Population(size=10, dynamics="lif")
        inp = Input(population=pop, mode="poisson", rate=100.0)
        assert inp.mode == "poisson"
        assert inp.rate == 100.0

    def test_static_data(self):
        pop = Population(size=5, dynamics="lif")
        data = torch.zeros(100, 5)
        inp = Input(population=pop, data=data)
        assert inp.data is not None
        assert inp.mode is None

    def test_generator(self):
        pop = Population(size=5, dynamics="lif")
        gen = lambda step: torch.zeros(5)
        inp = Input(population=pop, generator=gen)
        assert inp.generator is not None
        assert inp.mode is None

    def test_multiple_sources_error(self):
        pop = Population(size=5, dynamics="lif")
        with pytest.raises(ValueError, match="exactly one"):
            Input(population=pop, data=torch.zeros(10, 5), mode="poisson")

    def test_unsupported_mode(self):
        pop = Population(size=5, dynamics="lif")
        with pytest.raises(ValueError, match="Unsupported input mode"):
            Input(population=pop, mode="gaussian")


class TestInputIntegration:
    def test_graph_with_inputs(self):
        p1 = Population(size=50, dynamics="lif")
        p2 = Population(size=10, dynamics="lif")
        conn = Connection(source=p1, target=p2, pattern="dense")
        inp = Input(population=p1, mode="poisson", rate=100.0)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        assert len(graph.inputs) == 1

    def test_graph_without_inputs_backward_compat(self):
        p1 = Population(size=50, dynamics="lif")
        p2 = Population(size=10, dynamics="lif")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        assert graph.inputs == []

    def test_static_data_run(self):
        p1 = Population(size=20, dynamics="lif")
        p2 = Population(size=5, dynamics="lif")
        conn = Connection(source=p1, target=p2, pattern="dense")

        # All ones → every source neuron fires every step
        data = torch.ones(100, 20)
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        assert model.metrics["total_spikes"] > 0

    def test_generator_run(self):
        p1 = Population(size=10, dynamics="lif")
        p2 = Population(size=5, dynamics="lif")
        conn = Connection(source=p1, target=p2, pattern="dense")

        def gen(step):
            return (torch.rand(10) < 0.1).float()

        inp = Input(population=p1, generator=gen)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.05, dt=1e-3)
        assert model.metrics["num_steps"] == 50

    def test_custom_poisson_rate(self):
        p1 = Population(size=50, dynamics="lif")
        p2 = Population(size=10, dynamics="lif")
        conn = Connection(source=p1, target=p2, pattern="dense")

        inp = Input(population=p1, mode="poisson", rate=200.0)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        model.run(duration=0.1, dt=1e-3)
        # Higher rate should produce more source spikes
        assert model.metrics["spike_counts"][p1.id] > 0
