"""Tests for auto-quantization and QAT."""

from __future__ import annotations

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from nuro.backends.gpu.quantization import (
    FakeQuantize,
    QuantizedLinear,
    enable_qat,
    quantize_model,
)
from nuro.backends.loihi.transfer import quantize_weights
from nuro.backends.spinnaker2.transfer import quantize_weights_s2


class TestQuantizeWeightsLoihi:
    """Test Loihi quantization."""

    def test_basic_quantization(self):
        w = np.random.randn(4, 3).astype(np.float32)
        q, scale = quantize_weights(w)
        assert q.dtype == np.int32
        assert np.all(q >= -256)
        assert np.all(q <= 254)

    def test_zero_weights(self):
        w = np.zeros((3, 3), dtype=np.float32)
        q, scale = quantize_weights(w)
        assert np.all(q == 0)

    def test_even_integers(self):
        """Loihi requires even integers."""
        w = np.random.randn(5, 5).astype(np.float32)
        q, _ = quantize_weights(w)
        assert np.all(q % 2 == 0)


class TestQuantizeWeightsS2:
    """Test SpiNNaker 2 quantization."""

    def test_basic_quantization(self):
        w = np.random.randn(4, 3).astype(np.float32)
        q, scale = quantize_weights_s2(w)
        assert q.dtype == np.int32
        assert np.all(q >= -15)
        assert np.all(q <= 15)

    def test_zero_weights(self):
        w = np.zeros((3, 3), dtype=np.float32)
        q, scale = quantize_weights_s2(w)
        assert np.all(q == 0)


class TestFakeQuantize:
    """Test fake quantization for QAT."""

    def test_forward(self):
        x = torch.randn(4, 3)
        y = FakeQuantize.apply(x, 8, -128.0, 127.0)
        assert y.shape == x.shape

    def test_gradient_passthrough(self):
        x = torch.randn(4, 3, requires_grad=True)
        y = FakeQuantize.apply(x, 8, -128.0, 127.0)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestQuantizedLinear:
    """Test QuantizedLinear wrapper."""

    def test_forward(self):
        linear = torch.nn.Linear(5, 3, bias=False)
        ql = QuantizedLinear(linear, num_bits=8, target="loihi")
        x = torch.randn(2, 5)
        y = ql(x)
        assert y.shape == (2, 3)

    def test_gradient_flows(self):
        linear = torch.nn.Linear(5, 3, bias=False)
        ql = QuantizedLinear(linear, num_bits=8, target="loihi")
        x = torch.randn(2, 5, requires_grad=True)
        y = ql(x)
        y.sum().backward()
        assert linear.weight.grad is not None


class TestQuantizeModel:
    """Test unified quantize_model interface."""

    def test_loihi_target(self):
        weights = {"a__b": np.random.randn(4, 3).astype(np.float32)}
        q = quantize_model(weights, target="loihi")
        assert q["a__b"].dtype == np.int32
        assert np.all(np.abs(q["a__b"]) <= 256)

    def test_spinnaker2_target(self):
        weights = {"a__b": np.random.randn(4, 3).astype(np.float32)}
        q = quantize_model(weights, target="spinnaker2")
        assert q["a__b"].dtype == np.int32
        assert np.all(np.abs(q["a__b"]) <= 15)


class TestAutoQuantization:
    """Test auto-quantization in compile()."""

    def test_auto_quantize_defaults_off_for_gpu(self):
        """GPU target should not auto-quantize."""
        import nuro

        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])

        # Should compile without quantization
        model = nuro.compile(graph, target="gpu")
        assert model is not None
