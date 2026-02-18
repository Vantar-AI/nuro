"""Tests for the Intel Loihi backend (Lava SDK)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("lava.magma.core")

import nuro
from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.input import Input
from nuro.api.population import Population
from nuro.ir import IRGraph


def _simple_graph():
    """Build a minimal 2-population LIF graph for testing."""
    p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
    p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
    conn = Connection(source=p1, target=p2, pattern="dense")
    return Graph([p1, p2], [conn]), p1, p2, conn


class TestLoihiCompile:
    def test_lif_compiles(self):
        graph, p1, p2, conn = _simple_graph()
        model = nuro.compile(graph, target="loihi")
        assert model is not None
        assert hasattr(model, "run")

    def test_if_compiles(self):
        p1 = Population(size=15, dynamics="if")
        p2 = Population(size=5, dynamics="if")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        model = nuro.compile(graph, target="loihi")
        assert model is not None

    def test_izhikevich_warns_simulation_only(self):
        p1 = Population(size=10, dynamics="izhikevich", params={"preset": "regular_spiking"})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = nuro.compile(graph, target="loihi")
            sim_warnings = [x for x in w if "simulation-only" in str(x.message)]
            assert len(sim_warnings) >= 1

    def test_requires_grad_raises(self):
        graph, *_ = _simple_graph()
        with pytest.raises(ValueError, match="requires_grad"):
            nuro.compile(graph, target="loihi", requires_grad=True)

    def test_batch_size_raises(self):
        graph, *_ = _simple_graph()
        model = nuro.compile(graph, target="loihi")
        with pytest.raises(ValueError, match="batch_size"):
            model.run(duration=0.01, batch_size=2)


class TestLoihiRun:
    def test_run_produces_metrics(self):
        graph, p1, p2, conn = _simple_graph()
        model = nuro.compile(graph, target="loihi")
        model.run(duration=0.01)
        assert "total_spikes" in model.metrics
        assert "num_steps" in model.metrics
        assert model.metrics["num_steps"] == 10
        assert model.metrics["batch_size"] == 1

    def test_reset_clears_metrics(self):
        graph, *_ = _simple_graph()
        model = nuro.compile(graph, target="loihi")
        model.run(duration=0.01)
        assert model.metrics["num_steps"] == 10
        model.reset()
        assert model.metrics == {}


class TestLoihiInputs:
    def test_static_input(self):
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        data = np.random.rand(20, 10).astype(np.float32)
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])

        model = nuro.compile(graph, target="loihi")
        model.run(duration=0.02)
        assert model.metrics["num_steps"] == 20

    def test_generator_input_raises(self):
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        inp = Input(population=p1, generator=lambda step: np.random.rand(10))
        graph = Graph([p1, p2], [conn], inputs=[inp])

        model = nuro.compile(graph, target="loihi")
        with pytest.raises(ValueError, match="Generator"):
            model.run(duration=0.01)


class TestLoihiRecording:
    def test_record_spikes(self):
        graph, p1, p2, conn = _simple_graph()
        model = nuro.compile(graph, target="loihi")
        model.record("spikes", population=p2)
        model.run(duration=0.01)
        spikes = model.get_state("spikes", population=p2)
        assert isinstance(spikes, np.ndarray)
        # Shape should be (steps, neurons) = (10, 10)
        assert spikes.shape == (10, 10)


class TestLoihiWeightTransfer:
    def test_weight_transfer_from_gpu(self, tmp_path):
        """Train on GPU, save checkpoint, load into Loihi backend."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("spikingjelly")

        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])

        # Train on GPU and save
        gpu_model = nuro.compile(graph, target="gpu")
        gpu_model.run(duration=0.01)
        checkpoint_path = str(tmp_path / "trained.pt")
        gpu_model.save(checkpoint_path)

        # Load into Loihi
        loihi_model = nuro.compile(
            graph, target="loihi", weights_from=checkpoint_path
        )
        assert loihi_model is not None
        loihi_model.run(duration=0.01)
        assert "total_spikes" in loihi_model.metrics


class TestLoihiQuantization:
    def test_quantize_weights_range(self):
        """Quantized weights must stay within 8-bit Loihi range [-256, 254]."""
        from nuro.backends.loihi.transfer import quantize_weights

        weights = np.random.randn(10, 20).astype(np.float32) * 5.0
        q, scale = quantize_weights(weights, num_bits=8)
        assert q.dtype == np.int32
        assert q.min() >= -256
        assert q.max() <= 254

    def test_quantize_weights_even(self):
        """Loihi requires even integer weights (LSB=0)."""
        from nuro.backends.loihi.transfer import quantize_weights

        weights = np.random.randn(8, 8).astype(np.float32)
        q, _ = quantize_weights(weights, num_bits=8)
        assert np.all(q % 2 == 0)

    def test_quantize_zero_weights(self):
        """All-zero weights should return zeros without division by zero."""
        from nuro.backends.loihi.transfer import quantize_weights

        weights = np.zeros((5, 5), dtype=np.float32)
        q, scale = quantize_weights(weights)
        assert np.all(q == 0)
        assert scale == 1.0

    def test_quantize_preserves_sign(self):
        """Positive weights → positive integers, negative → negative."""
        from nuro.backends.loihi.transfer import quantize_weights

        weights = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
        q, _ = quantize_weights(weights)
        assert q[0, 0] > 0
        assert q[0, 1] < 0

    def test_compile_with_quantize_flag(self):
        """compile(..., quantize=True) should succeed without error."""
        graph, p1, p2, conn = _simple_graph()
        # quantize=True without weights_from is a no-op (no weights to quantize)
        model = nuro.compile(graph, target="loihi", quantize=True)
        assert model is not None
        model.run(duration=0.01)


class TestLoihiAPI:
    def test_compile_via_api_entry_point(self):
        """Test that nuro.compile(graph, target='loihi') works end-to-end."""
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])

        model = nuro.compile(graph, target="loihi")
        model.run(duration=0.01)
        assert model.metrics["total_spikes"] >= 0
