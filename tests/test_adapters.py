"""Tests for nuro.adapters — file and hardware adapters."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nuro.recording import Recording


class TestFileAdapters:
    def test_from_numpy(self):
        from nuro.adapters.file import from_numpy
        data = np.random.randn(50, 4)
        rec = from_numpy(data, probe_name="voltages", dt=0.001)
        assert rec.num_steps == 50
        np.testing.assert_array_equal(rec.get("voltages"), data)

    def test_from_csv(self):
        from nuro.adapters.file import from_csv
        data = np.random.randn(20, 3)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            np.savetxt(f, data, delimiter=",")
            path = f.name

        rec = from_csv(path, dt=0.001)
        assert rec.num_steps == 20
        np.testing.assert_allclose(rec.get("data"), data, atol=1e-6)

    def test_from_hdf5(self):
        pytest.importorskip("h5py")
        import h5py

        data = np.random.randn(30, 5)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        with h5py.File(path, "w") as hf:
            hf.create_dataset("spikes", data=data)

        from nuro.adapters.file import from_hdf5
        rec = from_hdf5(path, dt=0.002)
        assert rec.num_steps == 30
        np.testing.assert_array_equal(rec.get("spikes"), data)


class TestSamnaOffline:
    def test_from_events(self):
        from nuro.adapters.samna import SamnaAdapter

        events = [(0.001, 0), (0.002, 1), (0.005, 0), (0.009, 2)]
        rec = SamnaAdapter.from_events(events, num_neurons=3, dt=0.001, duration=0.01)

        assert rec.num_steps == 10
        spikes = rec.get("spikes")
        assert spikes.shape == (10, 3)
        assert spikes[1, 0] == 1.0  # event at t=0.001, neuron 0
        assert spikes[2, 1] == 1.0  # event at t=0.002, neuron 1

    def test_from_events_empty(self):
        from nuro.adapters.samna import SamnaAdapter
        rec = SamnaAdapter.from_events([], num_neurons=5, dt=0.001, duration=0.1)
        assert rec.get("spikes").size == 0  # empty probe, no data extended
