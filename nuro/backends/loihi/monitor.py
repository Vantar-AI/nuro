"""Loihi state recording — Monitor-based probes for voltages, spikes, and weights."""

from __future__ import annotations

from typing import Any

import numpy as np


class LoihiRecorder:
    """Records state from Lava Processes using Monitor probes.

    Probes are registered before ``run()``, monitors are created during setup,
    and data is retrieved after the run completes.
    """

    def __init__(self) -> None:
        self._probe_specs: list[dict[str, Any]] = []
        self._monitors: dict[str, Any] = {}  # key → Lava Monitor
        self._data_cache: dict[str, np.ndarray] = {}  # key → collected array

    @property
    def has_probes(self) -> bool:
        return len(self._probe_specs) > 0

    def add_probe(
        self,
        name: str,
        *,
        population_id: str | None = None,
        connection_key: str | None = None,
        interval: int = 1,
    ) -> None:
        """Register a probe spec before run.

        Parameters
        ----------
        name : str
            ``"voltages"``, ``"spikes"``, or ``"weights"``.
        population_id : str, optional
            Target population ID.
        connection_key : str, optional
            Target connection key ``"src__tgt"``.
        interval : int
            Recording interval (steps between samples).
        """
        key = self._make_key(name, population_id, connection_key)
        self._probe_specs.append({
            "name": name,
            "population_id": population_id,
            "connection_key": connection_key,
            "interval": interval,
            "key": key,
        })

    def setup_monitors(
        self,
        num_steps: int,
        neurons: dict[str, Any],
        synapses: dict[str, Any],
    ) -> None:
        """Create Lava Monitor processes for all registered probes.

        Parameters
        ----------
        num_steps : int
            Total simulation steps.
        neurons : dict
            Mapping of population_id → Lava neuron Process.
        synapses : dict
            Mapping of connection_key → Lava Dense Process.
        """
        from lava.proc.monitor.process import Monitor

        for spec in self._probe_specs:
            key = spec["key"]
            name = spec["name"]
            pid = spec["population_id"]
            ckey = spec["connection_key"]

            monitor = Monitor()

            if name == "spikes" and pid is not None and pid in neurons:
                monitor.probe(neurons[pid].s_out, num_steps)
                self._monitors[key] = monitor

            elif name == "voltages" and pid is not None and pid in neurons:
                monitor.probe(neurons[pid].v, num_steps)
                self._monitors[key] = monitor

            elif name == "weights" and ckey is not None and ckey in synapses:
                monitor.probe(synapses[ckey].weights, num_steps)
                self._monitors[key] = monitor

    def collect_all(self) -> None:
        """Harvest data from all monitors while the runtime is still running.

        Must be called before ``root.stop()``. Results are cached and returned
        by subsequent ``get()`` calls.
        """
        for spec in self._probe_specs:
            key = spec["key"]
            if key not in self._monitors:
                continue
            monitor = self._monitors[key]
            data = monitor.get_data()
            if isinstance(data, dict):
                for outer in data.values():
                    if isinstance(outer, dict):
                        for inner in outer.values():
                            data = inner
                            break
                    else:
                        data = outer
                    break
            data = np.array(data)
            if data.ndim == 2:
                data = data.T
            self._data_cache[key] = data

    def get(
        self,
        name: str,
        *,
        population_id: str | None = None,
        connection_key: str | None = None,
    ) -> np.ndarray:
        """Retrieve recorded data after run.

        Returns data in Nuro convention: ``(steps, neurons)`` for spikes/voltages,
        ``(steps, out, in)`` for weights.
        """
        key = self._make_key(name, population_id, connection_key)

        # Return from cache if collect_all() was already called
        if key in self._data_cache:
            return self._data_cache[key]

        monitor = self._monitors.get(key)
        if monitor is None:
            return np.array([])

        data = monitor.get_data()
        # Lava Monitor returns {process_name: {var_name: array}} — unwrap two levels
        if isinstance(data, dict):
            for outer in data.values():
                if isinstance(outer, dict):
                    for inner in outer.values():
                        data = inner
                        break
                else:
                    data = outer
                break

        data = np.array(data)

        # Transpose from Lava convention (neurons, steps) → Nuro (steps, neurons)
        if data.ndim == 2:
            data = data.T

        return data

    def reset(self) -> None:
        """Clear monitors and cache but keep probe specs."""
        self._monitors = {}
        self._data_cache = {}

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
