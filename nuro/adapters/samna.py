"""Adapter for SynSense hardware via Samna SDK.

Provides both live capture (requires samna) and offline analysis
of event lists (no hardware dependency).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nuro.recording import Recording


class SamnaAdapter:
    """Capture spike events from SynSense neuromorphic chips.

    Parameters
    ----------
    dt : float
        Bin width in seconds for converting events to frames.
    """

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

    def capture(self, board: Any, duration: float) -> Recording:
        """Live capture from a Samna board.

        Parameters
        ----------
        board : samna board object
            An opened Samna device.
        duration : float
            Capture duration in seconds.

        Returns
        -------
        Recording
        """
        try:
            import samna
            import time
        except ImportError:
            raise ImportError(
                "samna is required for live capture. "
                "Install with: pip install nuro[samna]"
            )

        graph = samna.graph.EventFilterGraph()
        _, _, sink = graph.sequential(
            [board.get_model_source_node(), "Spike", samna.graph.JitSink()]
        )
        graph.start()

        time.sleep(duration)

        graph.stop()
        events = sink.get_n_events(sink.get_n_available())

        return self._events_to_recording(events, duration)

    def _events_to_recording(self, events: list[Any], duration: float) -> Recording:
        """Convert Samna spike events to a Recording."""
        rec = Recording(dt=self.dt, source="samna")

        if not events:
            rec.add_probe("spikes")
            return rec

        neuron_ids = set()
        event_list = []
        for ev in events:
            t = ev.timestamp * 1e-6  # us -> s
            nid = ev.neuron_id
            neuron_ids.add(nid)
            event_list.append((t, nid))

        num_neurons = max(neuron_ids) + 1
        num_steps = int(duration / self.dt)

        spikes = np.zeros((num_steps, num_neurons), dtype=np.float32)
        for t, nid in event_list:
            step = int(t / self.dt)
            if 0 <= step < num_steps:
                spikes[step, nid] = 1.0

        rec.add_probe("spikes")
        rec.extend("spikes", spikes)
        return rec

    @staticmethod
    def from_events(
        events: list[tuple[float, int]],
        num_neurons: int,
        dt: float = 1e-3,
        duration: float | None = None,
    ) -> Recording:
        """Create a Recording from a list of (time_s, neuron_id) tuples.

        Works without Samna SDK — for offline analysis of exported event data.

        Parameters
        ----------
        events : list of (float, int)
            Each tuple is (time_in_seconds, neuron_id).
        num_neurons : int
            Total number of neurons.
        dt : float
            Bin width in seconds.
        duration : float, optional
            Total duration. If None, inferred from last event.
        """
        rec = Recording(dt=dt, source="samna-offline")
        rec.add_probe("spikes")

        if not events:
            return rec

        max_t = max(t for t, _ in events)
        dur = duration or (max_t + dt)
        num_steps = int(dur / dt)

        spikes = np.zeros((num_steps, num_neurons), dtype=np.float32)
        for t, nid in events:
            step = int(t / dt)
            if 0 <= step < num_steps and 0 <= nid < num_neurons:
                spikes[step, nid] = 1.0

        rec.extend("spikes", spikes)
        return rec
