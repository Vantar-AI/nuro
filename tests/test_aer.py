"""Tests for nuro.adapters.aer — AER binary format adapter."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nuro.adapters.aer import from_aer_events, from_aedat, from_aer_binary


class TestFromAerEvents:
    def test_basic(self):
        neuron_ids = np.array([0, 1, 0, 2])
        timestamps = np.array([0.001, 0.002, 0.005, 0.009])
        rec = from_aer_events(neuron_ids, timestamps, num_neurons=3, dt=0.001)
        spikes = rec.get("spikes")
        assert spikes.shape[1] == 3
        assert spikes[1, 0] == 1.0  # t=0.001, neuron 0
        assert spikes[2, 1] == 1.0  # t=0.002, neuron 1

    def test_empty(self):
        rec = from_aer_events(np.array([]), np.array([]), num_neurons=5, dt=0.001)
        assert rec.get("spikes").size == 0

    def test_vectorized_binning(self):
        n_events = 10000
        neuron_ids = np.random.randint(0, 100, n_events)
        timestamps = np.sort(np.random.uniform(0, 1.0, n_events))
        rec = from_aer_events(neuron_ids, timestamps, num_neurons=100, dt=0.001)
        spikes = rec.get("spikes")
        assert spikes.shape == (1000, 100)

    def test_out_of_range_events_ignored(self):
        neuron_ids = np.array([0, 999, 1])  # 999 out of range
        timestamps = np.array([0.001, 0.002, 0.003])
        rec = from_aer_events(neuron_ids, timestamps, num_neurons=10, dt=0.001)
        spikes = rec.get("spikes")
        assert spikes.shape[1] == 10


class TestFromAedat:
    def _make_aedat(self, events, path):
        """Create a minimal aedat 2.0 file."""
        with open(path, "wb") as f:
            f.write(b"# comment line\n")
            for addr, ts_us in events:
                f.write(struct.pack(">ii", addr, ts_us))

    def test_basic_parse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.aedat"
            # neuron 0 at 1000us, neuron 1 at 2000us, neuron 0 at 5000us
            self._make_aedat([(0, 1000), (1, 2000), (0, 5000)], path)
            rec = from_aedat(path, num_neurons=2, dt=0.001)
            spikes = rec.get("spikes")
            assert spikes.shape[1] == 2
            assert spikes[0, 0] == 1.0  # first event at step 0

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.aedat"
            path.write_bytes(b"# header\n")
            rec = from_aedat(path, num_neurons=10, dt=0.001)
            assert rec.get("spikes").size == 0

    def test_addr_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "masked.aedat"
            # Address has extra bits: 0x00010003 -> neuron 3 with mask 0xFFFF
            self._make_aedat([(0x00010003, 0), (0x00020001, 1000)], path)
            rec = from_aedat(path, num_neurons=5, dt=0.001, addr_mask=0xFFFF)
            spikes = rec.get("spikes")
            assert spikes[0, 3] == 1.0


class TestFromAerBinary:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.bin"
            with open(path, "wb") as f:
                # 4-byte addr + 4-byte timestamp, big endian
                f.write(struct.pack(">II", 0, 0))
                f.write(struct.pack(">II", 1, 1000))
                f.write(struct.pack(">II", 0, 5000))
            rec = from_aer_binary(path, num_neurons=2, dt=0.001, ts_scale=1e-6)
            spikes = rec.get("spikes")
            assert spikes.shape[1] == 2
