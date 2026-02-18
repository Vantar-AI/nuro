"""Tests for the SpiNNaker 2 backend (py-spinnaker2)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("spinnaker2")

import nuro
from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.input import Input
from nuro.api.population import Population
from nuro.ir import IRGraph


def _simple_graph():
    """Build a minimal 2-population LIF graph for testing."""
    p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
    p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
    conn = Connection(source=p1, target=p2, pattern="dense")
    return Graph([p1, p2], [conn]), p1, p2, conn


class TestSpiNNaker2Compile:
    def test_lif_compiles(self):
        graph, p1, p2, conn = _simple_graph()
        model = nuro.compile(graph, target="spinnaker2")
        assert model is not None
        assert hasattr(model, "run")

    def test_if_compiles(self):
        p1 = Population(size=8, dynamics="if")
        p2 = Population(size=4, dynamics="if")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        model = nuro.compile(graph, target="spinnaker2")
        assert model is not None

    def test_izhikevich_warns_lif_approximation(self):
        p1 = Population(size=8, dynamics="izhikevich", params={"preset": "regular_spiking"})
        p2 = Population(size=4, dynamics="lif")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = nuro.compile(graph, target="spinnaker2")
            approx_warnings = [x for x in w if "approximated as LIF" in str(x.message)]
            assert len(approx_warnings) >= 1

    def test_requires_grad_raises(self):
        graph, *_ = _simple_graph()
        with pytest.raises(ValueError, match="requires_grad"):
            nuro.compile(graph, target="spinnaker2", requires_grad=True)

    def test_batch_size_raises(self):
        graph, *_ = _simple_graph()
        model = nuro.compile(graph, target="spinnaker2")
        with pytest.raises(ValueError, match="batch_size"):
            model.run(duration=0.01, batch_size=2)

    def test_unknown_dynamics_raises(self):
        p1 = Population(size=5, dynamics="lif")
        p2 = Population(size=5, dynamics="lif")
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)
        ir.nodes[p1.id].dynamics = "unsupported_model"
        from nuro.backends.spinnaker2.backend import SpiNNaker2Backend
        with pytest.raises(ValueError, match="does not support dynamics"):
            SpiNNaker2Backend().compile(ir)


class TestSpiNNaker2Run:
    def test_run_produces_metrics(self):
        graph, p1, p2, conn = _simple_graph()
        model = nuro.compile(graph, target="spinnaker2")
        model.run(duration=0.01)
        assert "total_spikes" in model.metrics
        assert "num_steps" in model.metrics
        assert model.metrics["num_steps"] == 10
        assert model.metrics["batch_size"] == 1

    def test_reset_clears_metrics(self):
        graph, *_ = _simple_graph()
        model = nuro.compile(graph, target="spinnaker2")
        model.run(duration=0.01)
        assert model.metrics["num_steps"] == 10
        model.reset()
        assert model.metrics == {}


class TestSpiNNaker2Recurrent:
    def test_recurrent_two_pop(self):
        """Bidirectional (recurrent) graph compiles and runs."""
        p1 = Population(size=8, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=8, dynamics="lif", params={"tau": 20e-3})
        conn_fwd = Connection(source=p1, target=p2, pattern="dense")
        conn_back = Connection(source=p2, target=p1, pattern="dense")
        graph = Graph([p1, p2], [conn_fwd, conn_back])
        model = nuro.compile(graph, target="spinnaker2")
        model.run(duration=0.01)
        assert model.metrics["num_steps"] == 10

    def test_recurrent_ring(self):
        """Three-population ring (A→B→C→A) compiles and runs."""
        pA = Population(size=5, dynamics="lif", params={"tau": 20e-3})
        pB = Population(size=5, dynamics="lif", params={"tau": 20e-3})
        pC = Population(size=5, dynamics="lif", params={"tau": 20e-3})
        c1 = Connection(source=pA, target=pB, pattern="dense")
        c2 = Connection(source=pB, target=pC, pattern="dense")
        c3 = Connection(source=pC, target=pA, pattern="dense")
        graph = Graph([pA, pB, pC], [c1, c2, c3])
        model = nuro.compile(graph, target="spinnaker2")
        model.run(duration=0.01)
        assert model.metrics["num_steps"] == 10


class TestSpiNNaker2Inputs:
    def test_static_input(self):
        p1 = Population(size=8, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=4, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        data = np.random.rand(20, 8).astype(np.float32)
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])
        model = nuro.compile(graph, target="spinnaker2")
        model.run(duration=0.02)
        assert model.metrics["num_steps"] == 20

    def test_generator_input_raises(self):
        p1 = Population(size=8, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=4, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        inp = Input(population=p1, generator=lambda step: np.random.rand(8))
        graph = Graph([p1, p2], [conn], inputs=[inp])
        with pytest.raises(ValueError, match="Generator"):
            nuro.compile(graph, target="spinnaker2")


class TestSpiNNaker2WeightTransfer:
    def test_weight_transfer_from_gpu(self, tmp_path):
        """Train on GPU, save checkpoint, load into SpiNNaker 2 backend."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("spikingjelly")

        p1 = Population(size=8, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=4, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])

        gpu_model = nuro.compile(graph, target="gpu")
        gpu_model.run(duration=0.01)
        checkpoint_path = str(tmp_path / "trained.pt")
        gpu_model.save(checkpoint_path)

        s2_model = nuro.compile(
            graph, target="spinnaker2", weights_from=checkpoint_path, quantize=True
        )
        assert s2_model is not None
        s2_model.run(duration=0.01)
        assert "total_spikes" in s2_model.metrics


class TestSpiNNaker2Quantization:
    def test_quantize_range(self):
        """Quantized weights must stay within [-15, 15]."""
        from nuro.backends.spinnaker2.transfer import quantize_weights_s2

        weights = np.random.randn(10, 20).astype(np.float32) * 5.0
        q, scale = quantize_weights_s2(weights)
        assert q.dtype == np.int32
        assert q.min() >= -15
        assert q.max() <= 15

    def test_quantize_zero_weights(self):
        from nuro.backends.spinnaker2.transfer import quantize_weights_s2

        weights = np.zeros((5, 5), dtype=np.float32)
        q, scale = quantize_weights_s2(weights)
        assert np.all(q == 0)
        assert scale == 1.0

    def test_quantize_preserves_sign(self):
        from nuro.backends.spinnaker2.transfer import quantize_weights_s2

        weights = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
        q, _ = quantize_weights_s2(weights)
        assert q[0, 0] > 0
        assert q[0, 1] < 0

    def test_connection_list_shape(self):
        """weights_to_connection_list produces correct number of entries."""
        from nuro.backends.spinnaker2.transfer import weights_to_connection_list

        w = np.ones((3, 4), dtype=float)  # 3 post × 4 pre = 12 connections
        conns = weights_to_connection_list(w)
        assert len(conns) == 12
        assert len(conns[0]) == 4  # [pre, post, weight, delay]

    def test_connection_list_delay_clamped(self):
        """Delays outside [0, 7] are clamped."""
        from nuro.backends.spinnaker2.transfer import weights_to_connection_list

        w = np.ones((2, 2), dtype=float)
        conns_over = weights_to_connection_list(w, delay=99)
        conns_under = weights_to_connection_list(w, delay=-5)
        assert all(c[3] == 7 for c in conns_over)
        assert all(c[3] == 0 for c in conns_under)


class TestSpiNNaker2API:
    def test_compile_via_api_entry_point(self):
        """nuro.compile(graph, target='spinnaker2') works end-to-end."""
        p1 = Population(size=8, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=4, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        model = nuro.compile(graph, target="spinnaker2")
        model.run(duration=0.01)
        assert model.metrics["total_spikes"] >= 0

    def test_set_hardware_method_exists(self):
        """set_hardware() method is available for real chip deployment."""
        graph, *_ = _simple_graph()
        model = nuro.compile(graph, target="spinnaker2")
        assert hasattr(model, "set_hardware")
        assert callable(model.set_hardware)
