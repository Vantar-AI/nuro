"""Chip calibration profiles for analog neuromorphic hardware.

Analog chips have transistor mismatch - every neuron behaves differently.
CalibrationProfile stores per-neuron measurements (threshold, tau, gain)
and global parameters (bias currents, supply voltages) so experiments
are reproducible even when the same chip drifts over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class CalibrationProfile:
    """Measured characteristics of a specific neuromorphic chip.

    Parameters
    ----------
    chip_id : str
        Unique identifier for the physical chip (e.g. serial number, board label).
    chip_type : str
        Chip family (e.g. "dynap-se2", "brainscales-2", "loihi2").
    date : str
        When calibration was performed (ISO 8601).
    neuron_params : dict
        Per-neuron measured parameters. Maps neuron_id (int) to a dict
        of measured values (e.g. {"threshold": -52.3, "tau": 18.5e-3}).
    global_params : dict
        Chip-wide settings active during calibration
        (e.g. {"bias_current_nA": 150, "vdd": 1.8}).
    notes : str
        Free-text notes about the calibration session.
    """

    chip_id: str
    chip_type: str
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    neuron_params: dict[int, dict[str, Any]] = field(default_factory=dict)
    global_params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def set_neuron(self, neuron_id: int, **params: Any) -> None:
        """Set measured parameters for a single neuron."""
        self.neuron_params[neuron_id] = params

    def set_neurons_bulk(self, param_name: str, values: np.ndarray) -> None:
        """Set one parameter across all neurons from an array.

        Parameters
        ----------
        param_name : str
            Parameter name (e.g. "threshold").
        values : ndarray, shape (num_neurons,)
            Measured value per neuron.
        """
        for i, val in enumerate(values):
            if i not in self.neuron_params:
                self.neuron_params[i] = {}
            self.neuron_params[i][param_name] = float(val)

    def get_neuron(self, neuron_id: int) -> dict[str, Any]:
        """Get measured parameters for a neuron."""
        return self.neuron_params.get(neuron_id, {})

    def get_param_array(self, param_name: str) -> np.ndarray:
        """Get one parameter as an array across all neurons.

        Returns array of NaN for neurons without this parameter.
        """
        if not self.neuron_params:
            return np.array([])
        max_id = max(self.neuron_params.keys())
        result = np.full(max_id + 1, np.nan)
        for nid, params in self.neuron_params.items():
            if param_name in params:
                result[nid] = params[param_name]
        return result

    @property
    def num_neurons(self) -> int:
        return len(self.neuron_params)

    def mismatch_stats(self, param_name: str) -> dict[str, float]:
        """Compute mismatch statistics for a parameter across neurons."""
        values = self.get_param_array(param_name)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return {}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def save_hdf5(self, path: str) -> None:
        """Save calibration profile to HDF5."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required: pip install nuro[devtools]")

        with h5py.File(path, "w") as f:
            f.attrs["chip_id"] = self.chip_id
            f.attrs["chip_type"] = self.chip_type
            f.attrs["date"] = self.date
            f.attrs["notes"] = self.notes

            for key, val in self.global_params.items():
                try:
                    f.attrs[f"global/{key}"] = val
                except TypeError:
                    f.attrs[f"global/{key}"] = str(val)

            if self.neuron_params:
                # Collect all param names
                all_params = set()
                for params in self.neuron_params.values():
                    all_params.update(params.keys())

                for param_name in all_params:
                    arr = self.get_param_array(param_name)
                    f.create_dataset(f"neurons/{param_name}", data=arr)

    @classmethod
    def load_hdf5(cls, path: str) -> CalibrationProfile:
        """Load calibration profile from HDF5."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required: pip install nuro[devtools]")

        with h5py.File(path, "r") as f:
            profile = cls(
                chip_id=str(f.attrs["chip_id"]),
                chip_type=str(f.attrs["chip_type"]),
                date=str(f.attrs.get("date", "")),
                notes=str(f.attrs.get("notes", "")),
            )

            for key in f.attrs:
                if key.startswith("global/"):
                    profile.global_params[key[7:]] = f.attrs[key]

            if "neurons" in f:
                for param_name in f["neurons"]:
                    arr = np.array(f[f"neurons/{param_name}"])
                    profile.set_neurons_bulk(param_name, arr)

            return profile

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict (for experiment.json)."""
        return {
            "chip_id": self.chip_id,
            "chip_type": self.chip_type,
            "date": self.date,
            "notes": self.notes,
            "global_params": self.global_params,
            "num_neurons": self.num_neurons,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationProfile:
        """Reconstruct metadata from dict (neuron data comes from HDF5)."""
        return cls(
            chip_id=d["chip_id"],
            chip_type=d["chip_type"],
            date=d.get("date", ""),
            notes=d.get("notes", ""),
            global_params=d.get("global_params", {}),
        )

    def __repr__(self) -> str:
        return (
            f"CalibrationProfile(chip={self.chip_id!r}, type={self.chip_type!r}, "
            f"neurons={self.num_neurons}, date={self.date[:10]!r})"
        )
