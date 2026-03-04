"""Tests for connectivity patterns."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import nuro


class TestOneToOne:
    """Test one-to-one (diagonal) connectivity."""

    def test_basic(self):
        pop1 = nuro.Population(size=10, dynamics="lif")
        pop2 = nuro.Population(size=10, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2, pattern="one_to_one")
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        model.run(duration=0.01, dt=1e-3)
        assert model.metrics["num_steps"] == 10

    def test_mismatched_sizes_raises(self):
        pop1 = nuro.Population(size=10, dynamics="lif")
        pop2 = nuro.Population(size=5, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2, pattern="one_to_one")
        graph = nuro.Graph([pop1, pop2], [conn])
        with pytest.raises(ValueError, match="one_to_one"):
            nuro.compile(graph, target="gpu")

    def test_diagonal_weights(self):
        """Verify weights are diagonal."""
        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=4, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2, pattern="one_to_one")
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        w = list(model.snn.synapses.values())[0].weight.detach()
        # Off-diagonal should be zero
        mask = torch.eye(4).bool()
        assert torch.all(w[~mask] == 0)


class TestConv1d:
    """Test 1D convolutional connectivity."""

    def test_basic(self):
        pop1 = nuro.Population(size=20, dynamics="lif")
        pop2 = nuro.Population(size=10, dynamics="lif")
        conn = nuro.Connection(
            source=pop1, target=pop2, pattern="conv1d",
            params={"kernel_size": 5, "stride": 2},
        )
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        model.run(duration=0.01, dt=1e-3)
        assert model.metrics["num_steps"] == 10

    def test_default_params(self):
        pop1 = nuro.Population(size=10, dynamics="lif")
        pop2 = nuro.Population(size=10, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2, pattern="conv1d")
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        assert model is not None


class TestDistanceDependent:
    """Test distance-dependent connectivity."""

    def test_basic(self):
        pop1 = nuro.Population(size=20, dynamics="lif")
        pop2 = nuro.Population(size=10, dynamics="lif")
        conn = nuro.Connection(
            source=pop1, target=pop2, pattern="distance_dependent",
            params={"sigma": 0.2},
        )
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        model.run(duration=0.01, dt=1e-3)
        assert model.metrics["num_steps"] == 10

    def test_nearby_neurons_more_connected(self):
        """Nearby neurons should have more connections than distant ones."""
        pop1 = nuro.Population(size=100, dynamics="lif")
        pop2 = nuro.Population(size=100, dynamics="lif")
        conn = nuro.Connection(
            source=pop1, target=pop2, pattern="distance_dependent",
            params={"sigma": 0.1},
        )
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        w = list(model.snn.synapses.values())[0].weight.detach()
        # Diagonal region should have more non-zero weights than corners
        diag_count = (w[:10, :10] != 0).sum().item()
        corner_count = (w[:10, 90:] != 0).sum().item()
        assert diag_count >= corner_count
