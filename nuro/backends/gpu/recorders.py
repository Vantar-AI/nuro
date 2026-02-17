"""GPU state recorders — probe voltages, spikes, and weights during simulation."""

from __future__ import annotations

from typing import Any

import torch


class Recorder:
    """Accumulates state tensors on CPU during a simulation run.

    Each probe has a name, an optional population/connection target, and an
    optional recording interval.  Data is stored as a list of CPU tensors and
    stacked into a single tensor on demand via :meth:`get`.
    """

    def __init__(self) -> None:
        self._probes: list[dict[str, Any]] = []
        self._buffers: dict[str, list[torch.Tensor]] = {}

    def add_probe(
        self,
        name: str,
        *,
        population_id: str | None = None,
        connection_key: str | None = None,
        interval: int = 1,
    ) -> None:
        """Register a new recording probe.

        Parameters
        ----------
        name : str
            Human-readable name (e.g. ``"voltages"``, ``"spikes"``).
        population_id : str, optional
            Target population ID (for voltage / spike recording).
        connection_key : str, optional
            Target connection key ``"src__tgt"`` (for weight recording).
        interval : int
            Record every *interval* steps (default 1 = every step).
        """
        key = self._make_key(name, population_id, connection_key)
        self._probes.append({
            "name": name,
            "population_id": population_id,
            "connection_key": connection_key,
            "interval": interval,
            "key": key,
        })
        self._buffers[key] = []

    def record_step(
        self,
        step: int,
        spikes: dict[str, torch.Tensor],
        snn: Any,
    ) -> None:
        """Called each timestep to capture data for active probes."""
        for probe in self._probes:
            if step % probe["interval"] != 0:
                continue

            key = probe["key"]
            name = probe["name"]
            pid = probe["population_id"]
            ckey = probe["connection_key"]

            if name == "spikes" and pid is not None:
                if pid in spikes:
                    self._buffers[key].append(spikes[pid].detach().cpu())

            elif name == "voltages" and pid is not None:
                if pid in snn.neurons:
                    neuron_layer = snn.neurons[pid]
                    if hasattr(neuron_layer, "v"):
                        self._buffers[key].append(neuron_layer.v.detach().cpu().clone())

            elif name == "weights" and ckey is not None:
                if ckey in snn.synapses:
                    synapse = snn.synapses[ckey]
                    self._buffers[key].append(
                        synapse.weight.detach().cpu().clone()
                    )

    def get(
        self,
        name: str,
        *,
        population_id: str | None = None,
        connection_key: str | None = None,
    ) -> torch.Tensor:
        """Return recorded data as a stacked tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(num_recordings, ...)`` where ``...`` depends on the probe
            type: ``(pop_size,)`` for spikes/voltages, ``(out, in)`` for weights.
        """
        key = self._make_key(name, population_id, connection_key)
        buf = self._buffers.get(key, [])
        if not buf:
            return torch.tensor([])
        return torch.stack(buf)

    def reset(self) -> None:
        """Clear all recorded data (but keep probe registrations)."""
        for key in self._buffers:
            self._buffers[key] = []

    @property
    def has_probes(self) -> bool:
        return len(self._probes) > 0

    @staticmethod
    def _make_key(
        name: str,
        population_id: str | None,
        connection_key: str | None,
    ) -> str:
        if population_id:
            return f"{name}:{population_id}"
        if connection_key:
            return f"{name}:{connection_key}"
        return name
