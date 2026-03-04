"""Tests for ANN-to-SNN conversion."""

from __future__ import annotations

import pytest
import numpy as np

torch = pytest.importorskip("torch")
nn = torch.nn

from nuro.conversion.ann2snn import convert_ann, normalize_weights
from nuro.api.graph import Graph


class TestConvertANN:
    """Test ANN to SNN conversion."""

    def test_simple_mlp(self):
        """Convert a 2-layer MLP to SNN."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        graph = convert_ann(model, input_shape=(10,))

        assert isinstance(graph, Graph)
        # input + hidden + output = 3 populations
        assert graph.num_populations == 3
        assert graph.num_connections == 2

        # Check sizes
        sizes = [p.size for p in graph.populations]
        assert sizes == [10, 20, 5]

        # All should be IF neurons
        for pop in graph.populations:
            assert pop.dynamics == "if"

        # Connections should have weights
        for conn in graph.connections:
            assert "weights" in conn.params
            assert conn.params["weights"] is not None

    def test_mlp_with_batchnorm(self):
        """Convert MLP with BatchNorm (BN folding)."""
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        # Run a forward pass to initialize BN running stats
        model.eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        graph = convert_ann(model, input_shape=(8,))

        assert graph.num_populations == 3
        assert graph.num_connections == 2

    def test_weight_extraction(self):
        """Verify extracted weights match original model."""
        model = nn.Sequential(nn.Linear(4, 3, bias=False))
        with torch.no_grad():
            model[0].weight.fill_(0.5)

        graph = convert_ann(model, input_shape=(4,))

        w = graph.connections[0].params["weights"]
        np.testing.assert_allclose(w, 0.5, atol=1e-6)

    def test_single_layer(self):
        """Convert a single linear layer."""
        model = nn.Sequential(nn.Linear(5, 3))
        graph = convert_ann(model, input_shape=(5,))

        assert graph.num_populations == 2
        assert graph.num_connections == 1

    def test_deep_mlp(self):
        """Convert a deeper MLP."""
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        graph = convert_ann(model, input_shape=(784,))

        assert graph.num_populations == 4  # input + 3 layers
        assert graph.num_connections == 3


class TestNormalizeWeights:
    """Test weight normalization."""

    def test_robust_normalization(self):
        """Robust normalization scales weights by percentile."""
        model = nn.Sequential(nn.Linear(4, 3, bias=False))
        graph = convert_ann(model, input_shape=(4,))
        normalized = normalize_weights(graph, method="robust")

        for conn in normalized.connections:
            w = np.array(conn.params["weights"])
            assert np.all(np.abs(w) <= 1.0 + 1e-6)

    def test_max_normalization(self):
        """Max normalization scales by max absolute value."""
        model = nn.Sequential(nn.Linear(4, 3, bias=False))
        with torch.no_grad():
            model[0].weight.uniform_(-5, 5)
        graph = convert_ann(model, input_shape=(4,))
        normalized = normalize_weights(graph, method="max")

        for conn in normalized.connections:
            w = np.array(conn.params["weights"])
            assert np.max(np.abs(w)) <= 1.0 + 1e-6

    def test_preserves_structure(self):
        """Normalization preserves graph structure."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )
        graph = convert_ann(model, input_shape=(10,))
        normalized = normalize_weights(graph)

        assert normalized.num_populations == graph.num_populations
        assert normalized.num_connections == graph.num_connections


class TestConvertAndCompile:
    """Test full convert → compile → run pipeline."""

    def test_convert_compile_run(self):
        """Convert MLP → compile to GPU → run."""
        import nuro

        model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        graph = convert_ann(model, input_shape=(10,))
        compiled = nuro.compile(graph, target="gpu")
        compiled.run(duration=0.01, dt=1e-3)

        assert compiled.metrics["num_steps"] == 10
        assert compiled.metrics["total_spikes"] >= 0
