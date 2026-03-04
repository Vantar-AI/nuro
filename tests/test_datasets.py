"""Tests for neuromorphic dataset loaders."""

from __future__ import annotations

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from nuro.datasets.utils import get_cache_dir, events_to_spike_tensor
from nuro.datasets.vision import NMNIST, DVSCifar10, DVSGesture


class TestUtils:
    """Test dataset utilities."""

    def test_cache_dir_default(self):
        d = get_cache_dir()
        assert d.exists()
        assert "nuro" in str(d)

    def test_cache_dir_custom(self, tmp_path):
        d = get_cache_dir(str(tmp_path / "custom"))
        assert d.exists()
        assert "custom" in str(d)

    def test_events_to_spike_tensor_empty(self):
        events = np.zeros((0, 4), dtype=np.int64)
        tensor = events_to_spike_tensor(events, num_neurons=100, duration_ms=10.0)
        assert tensor.shape == (10, 100)
        assert np.all(tensor == 0)

    def test_events_to_spike_tensor_basic(self):
        # Create simple events: x=0, y=0, t=[0,5,10], p=0
        events = np.array([
            [0, 0, 0, 0],
            [0, 0, 5000, 0],
            [0, 0, 10000, 0],
        ], dtype=np.int64)
        tensor = events_to_spike_tensor(events, num_neurons=100, duration_ms=10.0)
        assert tensor.shape == (10, 100)
        assert tensor.sum() > 0


class TestNMNIST:
    """Test N-MNIST loader."""

    def test_init(self, tmp_path):
        dataset = NMNIST(root=str(tmp_path), train=True)
        assert dataset.NUM_NEURONS == 34 * 34 * 2
        assert dataset.NUM_CLASSES == 10

    def test_missing_data_raises(self, tmp_path):
        dataset = NMNIST(root=str(tmp_path / "nonexistent"), train=True)
        with pytest.raises(FileNotFoundError, match="N-MNIST"):
            len(dataset)

    def test_read_bin_empty(self, tmp_path):
        """Empty bin file should return empty array."""
        bin_path = tmp_path / "empty.bin"
        bin_path.write_bytes(b"")
        events = NMNIST._read_nmnist_bin(bin_path)
        assert len(events) == 0

    def test_with_synthetic_data(self, tmp_path):
        """Create synthetic N-MNIST data and load."""
        train_dir = tmp_path / "nmnist" / "Train" / "0"
        train_dir.mkdir(parents=True)

        # Create a synthetic bin file (5 bytes per event)
        events = []
        for i in range(10):
            x, y, p, t = 10, 10, 1, i * 1000
            byte2 = (p << 7) | ((t >> 16) & 0x7F)
            byte3 = (t >> 8) & 0xFF
            byte4 = t & 0xFF
            events.extend([x, y, byte2, byte3, byte4])

        bin_path = train_dir / "00001.bin"
        bin_path.write_bytes(bytes(events))

        dataset = NMNIST(root=str(tmp_path), train=True, num_steps=50)
        assert len(dataset) == 1
        tensor, label = dataset[0]
        assert tensor.shape == (50, 34 * 34 * 2)
        assert label == 0


class TestDVSCifar10:
    """Test DVS-CIFAR10 loader."""

    def test_init(self, tmp_path):
        dataset = DVSCifar10(root=str(tmp_path), train=True)
        assert dataset.NUM_NEURONS == 128 * 128 * 2
        assert dataset.NUM_CLASSES == 10

    def test_missing_data_raises(self, tmp_path):
        dataset = DVSCifar10(root=str(tmp_path / "nonexistent"), train=True)
        with pytest.raises(FileNotFoundError, match="DVS-CIFAR10"):
            len(dataset)


class TestDVSGesture:
    """Test DVS Gesture loader."""

    def test_init(self, tmp_path):
        dataset = DVSGesture(root=str(tmp_path), train=True)
        assert dataset.NUM_NEURONS == 128 * 128 * 2
        assert dataset.NUM_CLASSES == 11

    def test_missing_data_raises(self, tmp_path):
        dataset = DVSGesture(root=str(tmp_path / "nonexistent"), train=True)
        with pytest.raises(FileNotFoundError, match="DVS Gesture"):
            len(dataset)
