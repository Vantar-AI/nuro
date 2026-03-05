"""Tests for nuro.recording — hardware-agnostic Recording class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nuro.recording import Recording


class TestProbes:
    def test_add_probe_returns_key(self):
        rec = Recording()
        key = rec.add_probe("spikes", target_id="pop_a")
        assert key == "spikes:pop_a"

    def test_add_probe_without_target(self):
        rec = Recording()
        key = rec.add_probe("loss")
        assert key == "loss"

    def test_probes_list(self):
        rec = Recording()
        rec.add_probe("spikes")
        rec.add_probe("voltages", target_id="pop_a")
        assert len(rec.probes) == 2


class TestAppendGet:
    def test_append_and_get(self):
        rec = Recording(dt=1e-3)
        rec.add_probe("spikes")
        rec.append("spikes", np.array([1, 0, 1]))
        rec.append("spikes", np.array([0, 1, 0]))
        data = rec.get("spikes")
        assert data.shape == (2, 3)
        np.testing.assert_array_equal(data[0], [1, 0, 1])

    def test_append_auto_creates_probe(self):
        rec = Recording()
        rec.append("spikes", np.array([1, 0]))
        assert len(rec.probes) == 1
        assert rec.get("spikes").shape == (1, 2)

    def test_extend_batch(self):
        rec = Recording()
        batch = np.random.randint(0, 2, (10, 5))
        rec.extend("spikes", batch)
        data = rec.get("spikes")
        assert data.shape == (10, 5)
        np.testing.assert_array_equal(data, batch)

    def test_get_empty_probe(self):
        rec = Recording()
        rec.add_probe("spikes")
        data = rec.get("spikes")
        assert data.size == 0


class TestProperties:
    def test_num_steps_and_duration(self):
        rec = Recording(dt=0.001)
        rec.extend("spikes", np.zeros((100, 10)))
        assert rec.num_steps == 100
        assert abs(rec.duration - 0.1) < 1e-9

    def test_time_axis(self):
        rec = Recording(dt=0.5)
        rec.extend("x", np.zeros((4,)))
        t = rec.time_axis()
        np.testing.assert_allclose(t, [0.0, 0.5, 1.0, 1.5])

    def test_reset(self):
        rec = Recording()
        rec.append("spikes", np.array([1]))
        rec.reset()
        assert rec.num_steps == 0
        assert len(rec.probes) == 1  # probes kept


class TestTorchCompat:
    def test_torch_tensor_auto_converts(self):
        pytest.importorskip("torch")
        import torch

        rec = Recording()
        rec.append("spikes", torch.tensor([1, 0, 1]))
        data = rec.get("spikes")
        assert isinstance(data, np.ndarray)
        np.testing.assert_array_equal(data[0], [1, 0, 1])


class TestHDF5:
    def test_save_and_load_roundtrip(self):
        pytest.importorskip("h5py")

        rec = Recording(dt=0.001, source="test", metadata={"chip": "dynap"})
        rec.add_probe("spikes", unit="binary")
        spikes = np.random.randint(0, 2, (50, 8))
        rec.extend("spikes", spikes)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "rec.h5")
            rec.save_hdf5(path)
            loaded = Recording.load_hdf5(path)

        assert loaded.id == rec.id
        assert loaded.dt == rec.dt
        assert loaded.source == "test"
        assert loaded.num_steps == 50
        np.testing.assert_array_equal(loaded.get("spikes"), spikes)
