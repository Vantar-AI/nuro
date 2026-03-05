"""Experiment tracking for neuromorphic research.

Captures network descriptions, hardware configs, recordings, and metrics
in a single Experiment object. Persists to disk as JSON metadata + HDF5
recordings for reproducibility and comparison.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nuro.recording import Recording


@dataclass
class HardwareConfig:
    """Describes the hardware platform for an experiment."""

    platform: str  # "gpu", "loihi2", "spinnaker2", "dynap-se2", "akida", ...
    chip_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"platform": self.platform, "chip_id": self.chip_id, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareConfig:
        return cls(platform=d["platform"], chip_id=d.get("chip_id", ""), params=d.get("params", {}))


class Experiment:
    """A neuromorphic experiment with recordings, metrics, and metadata.

    Parameters
    ----------
    name : str
        Human-readable experiment name.
    project : str
        Project or group name for organization.
    description : str
        What this experiment tests or measures.
    tags : list of str
        Searchable tags.
    """

    def __init__(
        self,
        name: str,
        project: str = "",
        description: str = "",
        tags: list[str] | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.project = project
        self.description = description
        self.tags = tags or []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "running"

        self._hardware: HardwareConfig | None = None
        self._network: dict[str, Any] | None = None
        self._recordings: dict[str, Recording] = {}
        self._metrics: dict[str, Any] = {}
        self._params: dict[str, Any] = {}

    def set_hardware(self, platform: str, **kwargs: Any) -> None:
        """Set the hardware configuration."""
        self._hardware = HardwareConfig(platform=platform, params=kwargs)

    def set_network(
        self,
        graph: Any = None,
        ir_graph: Any = None,
        nir_graph: Any = None,
    ) -> None:
        """Attach a network description.

        Accepts a Nuro API Graph, IRGraph, or NIR graph. Serializes to dict.
        """
        if ir_graph is not None:
            from nuro.backends.cloud.serializer import serialize_ir_graph
            self._network = serialize_ir_graph(ir_graph)
        elif graph is not None:
            from nuro.ir import IRGraph
            ir = IRGraph.from_api_graph(graph)
            from nuro.backends.cloud.serializer import serialize_ir_graph
            self._network = serialize_ir_graph(ir)
        elif nir_graph is not None:
            from nuro.ir.nir_compat import from_nir
            ir = from_nir(nir_graph)
            from nuro.backends.cloud.serializer import serialize_ir_graph
            self._network = serialize_ir_graph(ir)
        else:
            raise ValueError("Provide one of: graph, ir_graph, nir_graph")

    def set_params(self, **params: Any) -> None:
        """Set experiment parameters (hyperparameters, config)."""
        self._params.update(params)

    def new_recording(
        self,
        label: str,
        dt: float = 1e-3,
        source: str = "",
    ) -> Recording:
        """Create and register a new Recording."""
        rec = Recording(dt=dt, source=source or (self._hardware.platform if self._hardware else ""))
        self._recordings[label] = rec
        return rec

    def add_recording(self, label: str, recording: Recording) -> None:
        """Attach an existing Recording."""
        self._recordings[label] = recording

    def get_recording(self, label: str) -> Recording:
        """Retrieve a recording by label."""
        return self._recordings[label]

    @property
    def recordings(self) -> dict[str, Recording]:
        return dict(self._recordings)

    def log_metric(self, key: str, value: Any) -> None:
        """Log a single metric."""
        self._metrics[key] = value

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log multiple metrics at once."""
        self._metrics.update(metrics)

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def complete(self) -> None:
        """Mark experiment as completed."""
        self.status = "completed"

    def save(self, directory: str | Path) -> Path:
        """Persist experiment to disk.

        Creates::

            <directory>/<experiment_id>/
                experiment.json
                network.json          (if set)
                recording_<label>.h5  (per recording)
        """
        base = Path(directory) / self.id
        base.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta = {
            "id": self.id,
            "name": self.name,
            "project": self.project,
            "description": self.description,
            "tags": self.tags,
            "status": self.status,
            "created_at": self.created_at,
            "hardware": self._hardware.to_dict() if self._hardware else None,
            "params": self._params,
            "metrics": self._metrics,
            "recordings": list(self._recordings.keys()),
        }
        (base / "experiment.json").write_text(json.dumps(meta, indent=2, default=str))

        # Network
        if self._network:
            (base / "network.json").write_text(json.dumps(self._network, indent=2, default=str))

        # Recordings
        for label, rec in self._recordings.items():
            rec.save_hdf5(str(base / f"recording_{label}.h5"))

        return base

    @classmethod
    def load(cls, directory: str | Path) -> Experiment:
        """Load an experiment from disk."""
        base = Path(directory)
        meta = json.loads((base / "experiment.json").read_text())

        exp = cls(
            name=meta["name"],
            project=meta.get("project", ""),
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
        )
        exp.id = meta["id"]
        exp.status = meta.get("status", "completed")
        exp.created_at = meta.get("created_at", "")
        exp._params = meta.get("params", {})
        exp._metrics = meta.get("metrics", {})

        if meta.get("hardware"):
            exp._hardware = HardwareConfig.from_dict(meta["hardware"])

        # Network
        net_path = base / "network.json"
        if net_path.exists():
            exp._network = json.loads(net_path.read_text())

        # Recordings
        for label in meta.get("recordings", []):
            h5_path = base / f"recording_{label}.h5"
            if h5_path.exists():
                exp._recordings[label] = Recording.load_hdf5(str(h5_path))

        return exp

    def __repr__(self) -> str:
        return (
            f"Experiment(name={self.name!r}, project={self.project!r}, "
            f"status={self.status!r}, recordings={len(self._recordings)}, "
            f"metrics={len(self._metrics)})"
        )
