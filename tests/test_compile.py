"""Tests for the compile() entry point."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

import nuro


class TestCompile:
    def test_compile_auto(self):
        p1 = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
        p2 = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = nuro.Connection(source=p1, target=p2, pattern="dense")
        graph = nuro.Graph([p1, p2], [conn])

        model = nuro.compile(graph, target="auto")
        model.run(duration=0.1)
        assert model.metrics["total_spikes"] >= 0

    def test_compile_gpu(self):
        p1 = nuro.Population(size=30, dynamics="lif", params={"tau": 20e-3})
        p2 = nuro.Population(size=5, dynamics="if")
        conn = nuro.Connection(source=p1, target=p2, pattern="dense")
        graph = nuro.Graph([p1, p2], [conn])

        model = nuro.compile(graph, target="gpu")
        model.run(duration=0.05)
        assert model.metrics["num_steps"] == 50

    def test_compile_unknown_backend(self):
        p1 = nuro.Population(size=10)
        graph = nuro.Graph([p1], [])
        with pytest.raises(ValueError, match="Unknown backend"):
            nuro.compile(graph, target="loihi")

    def test_version(self):
        assert nuro.__version__ == "0.2.0"
