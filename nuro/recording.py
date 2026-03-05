"""Hardware-agnostic recording for neuromorphic experiments.

Captures spike trains, membrane voltages, weights, and arbitrary signals
from any hardware platform. Data is stored as numpy arrays and serialized
to HDF5 for language-agnostic persistence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Probe:
    """A named recording channel within a Recording."""

    name: str
    target_id: str | None = None
    interval: int = 1
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Recording:
    """Hardware-agnostic data recording.

    Stores time-series data from probes as numpy arrays. Designed
    for neuromorphic experiments where data may come from GPU simulation,
    Loihi monitors, Samna event streams, or offline analysis.

    Parameters
    ----------
    dt : float
        Timestep in seconds (default 1e-3 = 1 ms).
    source : str
        Label for the data source (e.g. "gpu", "loihi2", "dynap-se2").
    metadata : dict, optional
        Arbitrary metadata attached to the recording.
    """

    def __init__(
        self,
        dt: float = 1e-3,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.dt = dt
        self.source = source
        self.metadata = metadata or {}
        self._probes: dict[str, Probe] = {}
        self._buffers: dict[str, list[np.ndarray]] = {}

    def add_probe(
        self,
        name: str,
        target_id: str | None = None,
        interval: int = 1,
        unit: str = "",
        **metadata: Any,
    ) -> str:
        """Register a recording probe.

        Returns the probe key (``name`` or ``name:target_id``).
        """
        key = self._make_key(name, target_id)
        self._probes[key] = Probe(
            name=name,
            target_id=target_id,
            interval=interval,
            unit=unit,
            metadata=metadata,
        )
        self._buffers[key] = []
        return key

    def append(self, name: str, data: Any, target_id: str | None = None) -> None:
        """Append a single timestep of data."""
        key = self._make_key(name, target_id)
        if key not in self._buffers:
            self.add_probe(name, target_id)
        arr = self._to_numpy(data)
        self._buffers[key].append(arr)

    def extend(self, name: str, data: Any, target_id: str | None = None) -> None:
        """Append a batch of timesteps (first axis = time)."""
        key = self._make_key(name, target_id)
        if key not in self._buffers:
            self.add_probe(name, target_id)
        arr = self._to_numpy(data)
        if arr.ndim == 0:
            self._buffers[key].append(arr.reshape(1))
        elif arr.ndim == 1:
            for val in arr:
                self._buffers[key].append(np.atleast_1d(val))
        else:
            for row in arr:
                self._buffers[key].append(row)

    def get(self, name: str, target_id: str | None = None) -> np.ndarray:
        """Return recorded data as a stacked array (time, ...)."""
        key = self._make_key(name, target_id)
        buf = self._buffers.get(key, [])
        if not buf:
            return np.array([])
        return np.stack(buf)

    def reset(self) -> None:
        """Clear all recorded data but keep probe registrations."""
        for key in self._buffers:
            self._buffers[key] = []

    @property
    def probes(self) -> list[Probe]:
        return list(self._probes.values())

    @property
    def num_steps(self) -> int:
        """Number of timesteps in the longest probe buffer."""
        if not self._buffers:
            return 0
        return max(len(b) for b in self._buffers.values())

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.num_steps * self.dt

    def time_axis(self) -> np.ndarray:
        """Return a time vector in seconds."""
        return np.arange(self.num_steps) * self.dt

    def save_hdf5(self, path: str) -> None:
        """Serialize the recording to an HDF5 file."""
        h5py = _import_h5py()
        with h5py.File(path, "w") as f:
            f.attrs["id"] = self.id
            f.attrs["dt"] = self.dt
            f.attrs["source"] = self.source
            # Store simple metadata as attributes
            for mk, mv in self.metadata.items():
                try:
                    f.attrs[f"meta/{mk}"] = mv
                except TypeError:
                    f.attrs[f"meta/{mk}"] = str(mv)

            for key, probe in self._probes.items():
                grp = f.create_group(f"probes/{key}")
                grp.attrs["name"] = probe.name
                grp.attrs["target_id"] = probe.target_id or ""
                grp.attrs["interval"] = probe.interval
                grp.attrs["unit"] = probe.unit
                data = self.get(probe.name, probe.target_id)
                if data.size > 0:
                    grp.create_dataset("data", data=data, compression="gzip")

    @classmethod
    def load_hdf5(cls, path: str) -> Recording:
        """Load a recording from an HDF5 file."""
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:
            rec = cls(
                dt=float(f.attrs["dt"]),
                source=str(f.attrs.get("source", "")),
            )
            rec.id = str(f.attrs["id"])

            # Restore metadata
            for ak in f.attrs:
                if ak.startswith("meta/"):
                    rec.metadata[ak[5:]] = f.attrs[ak]

            if "probes" not in f:
                return rec

            for key in f["probes"]:
                grp = f[f"probes/{key}"]
                name = str(grp.attrs["name"])
                tid = str(grp.attrs.get("target_id", "")) or None
                rec.add_probe(
                    name,
                    target_id=tid,
                    interval=int(grp.attrs.get("interval", 1)),
                    unit=str(grp.attrs.get("unit", "")),
                )
                if "data" in grp:
                    data = np.array(grp["data"])
                    rec.extend(name, data, target_id=tid)

            return rec

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray:
        """Convert input to numpy, handling torch tensors."""
        if isinstance(data, np.ndarray):
            return data
        if hasattr(data, "detach"):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _make_key(name: str, target_id: str | None) -> str:
        if target_id:
            return f"{name}:{target_id}"
        return name

    def __repr__(self) -> str:
        return (
            f"Recording(id={self.id!r}, source={self.source!r}, "
            f"probes={len(self._probes)}, steps={self.num_steps}, "
            f"duration={self.duration:.3f}s)"
        )


def _import_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 serialization. "
            "Install with: pip install nuro[devtools]"
        )
