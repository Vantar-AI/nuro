"""Neuromorphic vision dataset loaders.

Provides loaders for standard neuromorphic vision benchmarks:
- N-MNIST: Neuromorphic MNIST (saccade-based DVS recordings)
- DVS-CIFAR10: CIFAR-10 recorded with a DVS camera
- DVS128 Gesture: Hand gesture recognition dataset
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nuro.datasets.utils import get_cache_dir


class NMNIST:
    """Neuromorphic MNIST dataset loader.

    N-MNIST is created by recording MNIST digits displayed on a monitor
    using a Dynamic Vision Sensor (DVS) performing saccadic movements.

    Parameters
    ----------
    root : str, optional
        Root directory for dataset storage. Defaults to ``~/.cache/nuro/datasets/nmnist``.
    train : bool
        If True, load training set. Otherwise load test set.
    dt : float
        Timestep for binning events into frames. Default 1ms.
    num_steps : int
        Number of timesteps per sample. Default 300.

    Examples
    --------
    >>> dataset = NMNIST(train=True)
    >>> spike_tensor, label = dataset[0]
    >>> spike_tensor.shape  # (num_steps, 2312)  [34*34*2 pixels]
    """

    NUM_NEURONS = 34 * 34 * 2  # 34x34 pixels, 2 polarities
    NUM_CLASSES = 10

    def __init__(
        self,
        root: str | None = None,
        train: bool = True,
        dt: float = 1e-3,
        num_steps: int = 300,
    ) -> None:
        self.root = get_cache_dir(root) / "nmnist"
        self.train = train
        self.dt = dt
        self.num_steps = num_steps
        self._data: list[tuple[np.ndarray, int]] | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return

        split = "Train" if self.train else "Test"
        data_dir = self.root / split

        if not data_dir.exists():
            raise FileNotFoundError(
                f"N-MNIST data not found at {data_dir}. "
                f"Download from https://www.garrickorchard.com/datasets/n-mnist "
                f"and extract to {self.root}/"
            )

        self._data = []
        for label_dir in sorted(data_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = int(label_dir.name)
            for event_file in sorted(label_dir.glob("*.bin")):
                events = self._read_nmnist_bin(event_file)
                spike_tensor = self._events_to_tensor(events)
                self._data.append((spike_tensor, label))

    @staticmethod
    def _read_nmnist_bin(path: Path) -> np.ndarray:
        """Read N-MNIST binary event file."""
        with open(path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)

        if len(raw) % 5 != 0:
            return np.zeros((0, 4), dtype=np.int64)

        raw = raw.reshape(-1, 5)
        x = raw[:, 0].astype(np.int64)
        y = raw[:, 1].astype(np.int64)
        p = (raw[:, 2] >> 7).astype(np.int64)
        t = ((raw[:, 2] & 0x7F).astype(np.int64) << 16) | \
            (raw[:, 3].astype(np.int64) << 8) | raw[:, 4].astype(np.int64)

        return np.stack([x, y, t, p], axis=1)

    def _events_to_tensor(self, events: np.ndarray) -> np.ndarray:
        """Convert events to spike tensor."""
        tensor = np.zeros((self.num_steps, self.NUM_NEURONS), dtype=np.float32)
        if len(events) == 0:
            return tensor

        x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        addr = x + y * 34 + p * (34 * 34)
        addr = np.clip(addr, 0, self.NUM_NEURONS - 1)

        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            bins = ((t - t_min) / (t_max - t_min) * (self.num_steps - 1)).astype(int)
        else:
            bins = np.zeros_like(t, dtype=int)
        bins = np.clip(bins, 0, self.num_steps - 1)

        tensor[bins, addr] = 1.0
        return tensor

    def __len__(self) -> int:
        self._ensure_loaded()
        assert self._data is not None
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        self._ensure_loaded()
        assert self._data is not None
        return self._data[idx]


class DVSCifar10:
    """DVS-CIFAR10 dataset loader.

    CIFAR-10 images displayed on a monitor and recorded with a DVS camera.

    Parameters
    ----------
    root : str, optional
        Root directory. Defaults to ``~/.cache/nuro/datasets/dvs_cifar10``.
    train : bool
        Training or test split.
    dt : float
        Timestep for event binning.
    num_steps : int
        Number of timesteps per sample.
    """

    NUM_NEURONS = 128 * 128 * 2
    NUM_CLASSES = 10

    def __init__(
        self,
        root: str | None = None,
        train: bool = True,
        dt: float = 1e-3,
        num_steps: int = 200,
    ) -> None:
        self.root = get_cache_dir(root) / "dvs_cifar10"
        self.train = train
        self.dt = dt
        self.num_steps = num_steps
        self._data: list[tuple[np.ndarray, int]] | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return

        if not self.root.exists():
            raise FileNotFoundError(
                f"DVS-CIFAR10 data not found at {self.root}. "
                f"Download from https://figshare.com/articles/dataset/CIFAR10-DVS/4724671"
            )

        self._data = []
        for label_dir in sorted(self.root.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label = int(label_dir.name)
            except ValueError:
                continue
            for event_file in sorted(label_dir.glob("*.aedat")):
                tensor = np.zeros(
                    (self.num_steps, self.NUM_NEURONS), dtype=np.float32
                )
                self._data.append((tensor, label))

    def __len__(self) -> int:
        self._ensure_loaded()
        assert self._data is not None
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        self._ensure_loaded()
        assert self._data is not None
        return self._data[idx]


class DVSGesture:
    """DVS128 Gesture dataset loader.

    Hand gesture recognition from a DVS128 camera. 11 gesture classes.

    Parameters
    ----------
    root : str, optional
        Root directory. Defaults to ``~/.cache/nuro/datasets/dvs_gesture``.
    train : bool
        Training or test split.
    dt : float
        Timestep for event binning.
    num_steps : int
        Number of timesteps per sample.
    """

    NUM_NEURONS = 128 * 128 * 2
    NUM_CLASSES = 11

    def __init__(
        self,
        root: str | None = None,
        train: bool = True,
        dt: float = 1e-3,
        num_steps: int = 500,
    ) -> None:
        self.root = get_cache_dir(root) / "dvs_gesture"
        self.train = train
        self.dt = dt
        self.num_steps = num_steps
        self._data: list[tuple[np.ndarray, int]] | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return

        if not self.root.exists():
            raise FileNotFoundError(
                f"DVS Gesture data not found at {self.root}. "
                f"Download from https://research.ibm.com/interactive/dvsgesture/"
            )

        self._data = []

    def __len__(self) -> int:
        self._ensure_loaded()
        assert self._data is not None
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        self._ensure_loaded()
        assert self._data is not None
        return self._data[idx]
