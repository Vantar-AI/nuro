"""Tests for Population and Connection API classes."""

import pytest

from nuro.api.connection import Connection
from nuro.api.population import Population


class TestPopulation:
    def test_create_lif(self):
        pop = Population(size=100, dynamics="lif", params={"tau": 20e-3})
        assert pop.size == 100
        assert pop.dynamics == "lif"
        assert pop.params["tau"] == 20e-3
        assert isinstance(pop.id, str)

    def test_create_if(self):
        pop = Population(size=64, dynamics="if")
        assert pop.size == 64
        assert pop.dynamics == "if"

    def test_invalid_dynamics(self):
        with pytest.raises(ValueError, match="Unsupported dynamics"):
            Population(size=10, dynamics="hodgkin_huxley")

    def test_invalid_size(self):
        with pytest.raises(ValueError, match="size must be >= 1"):
            Population(size=0)

    def test_unique_ids(self):
        p1 = Population(size=10)
        p2 = Population(size=10)
        assert p1.id != p2.id

    def test_hashable(self):
        p = Population(size=10)
        s = {p}
        assert p in s


class TestConnection:
    def test_create_dense(self):
        src = Population(size=100)
        tgt = Population(size=10)
        conn = Connection(source=src, target=tgt, pattern="dense")
        assert conn.source is src
        assert conn.target is tgt
        assert conn.pattern == "dense"
        assert conn.plasticity == "none"

    def test_create_with_stdp(self):
        src = Population(size=50)
        tgt = Population(size=20)
        conn = Connection(source=src, target=tgt, pattern="dense", plasticity="stdp")
        assert conn.plasticity == "stdp"

    def test_random_sparse(self):
        src = Population(size=100)
        tgt = Population(size=10)
        conn = Connection(source=src, target=tgt, pattern="random_sparse")
        assert conn.pattern == "random_sparse"

    def test_invalid_pattern(self):
        src = Population(size=10)
        tgt = Population(size=10)
        with pytest.raises(ValueError, match="Unsupported pattern"):
            Connection(source=src, target=tgt, pattern="conv2d")

    def test_invalid_plasticity(self):
        src = Population(size=10)
        tgt = Population(size=10)
        with pytest.raises(ValueError, match="Unsupported plasticity"):
            Connection(source=src, target=tgt, plasticity="bcm")
