"""Tests for BrainChip Akida backend."""

from __future__ import annotations

import pytest
import numpy as np

from nuro.backends.akida.dynamics import get_akida_layer_config
from nuro.backends.akida.transfer import quantize_weights_akida
from nuro.ir.nodes import DynamicsNode


class TestAkidaDynamics:
    """Test Akida dynamics mapping."""

    def test_lif_supported(self):
        node = DynamicsNode(id="a", size=10, dynamics="lif", params={})
        config = get_akida_layer_config(node)
        assert config["type"] == "dense"
        assert config["units"] == 10

    def test_if_supported(self):
        node = DynamicsNode(id="b", size=5, dynamics="if", params={})
        config = get_akida_layer_config(node)
        assert config["units"] == 5

    def test_izhikevich_unsupported(self):
        node = DynamicsNode(id="c", size=5, dynamics="izhikevich", params={})
        with pytest.raises(ValueError, match="Akida does not support"):
            get_akida_layer_config(node)

    def test_adex_unsupported(self):
        node = DynamicsNode(id="d", size=5, dynamics="adex", params={})
        with pytest.raises(ValueError, match="Akida does not support"):
            get_akida_layer_config(node)


class TestAkidaQuantization:
    """Test Akida weight quantization."""

    def test_4bit_quantization(self):
        w = np.random.randn(4, 3).astype(np.float32)
        q, scale = quantize_weights_akida(w, num_bits=4)
        assert q.dtype == np.int8
        assert np.all(q >= -8)
        assert np.all(q <= 7)

    def test_8bit_quantization(self):
        w = np.random.randn(4, 3).astype(np.float32)
        q, scale = quantize_weights_akida(w, num_bits=8)
        assert np.all(q >= -128)
        assert np.all(q <= 127)

    def test_zero_weights(self):
        w = np.zeros((3, 3), dtype=np.float32)
        q, scale = quantize_weights_akida(w)
        assert np.all(q == 0)


class TestAkidaBackend:
    """Test Akida backend compilation (without Akida SDK)."""

    def test_compile_without_akida_sdk(self):
        """Should compile even without akida package (mock mode)."""
        import nuro

        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])

        model = nuro.compile(graph, target="akida")
        assert model is not None

    def test_akida_run(self):
        """Run should work even without Akida SDK."""
        import nuro

        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])

        model = nuro.compile(graph, target="akida")
        model.run(duration=0.01, dt=1e-3)
        assert model.metrics["num_steps"] == 10

    def test_akida_rejects_grad(self):
        """Akida should reject requires_grad=True."""
        import nuro

        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])

        with pytest.raises(ValueError, match="requires_grad"):
            nuro.compile(graph, target="akida", requires_grad=True)

    def test_akida_batch_size_error(self):
        """Akida should reject batch_size > 1."""
        import nuro

        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])

        model = nuro.compile(graph, target="akida")
        with pytest.raises(ValueError, match="batch_size"):
            model.run(duration=0.01, batch_size=2)

    def test_backend_registered(self):
        """Akida should be in the backend registry."""
        from nuro.backends import _REGISTRY
        assert "akida" in _REGISTRY
