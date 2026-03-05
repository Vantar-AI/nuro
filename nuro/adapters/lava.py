"""Bridge from Loihi LoihiRecorder to hardware-agnostic Recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nuro.recording import Recording

if TYPE_CHECKING:
    from nuro.backends.loihi.monitor import LoihiRecorder


def recording_from_loihi_model(
    recorder: LoihiRecorder,
    dt: float = 1e-3,
) -> Recording:
    """Convert a LoihiRecorder into a Recording.

    Parameters
    ----------
    recorder : nuro.backends.loihi.monitor.LoihiRecorder
        A populated LoihiRecorder after ``model.run()``.
    dt : float
        Timestep in seconds.

    Returns
    -------
    Recording
    """
    rec = Recording(dt=dt, source="loihi2")

    for spec in recorder._probe_specs:
        name = spec["name"]
        pid = spec.get("population_id")
        ckey = spec.get("connection_key")

        data = recorder.get(name, population_id=pid, connection_key=ckey)
        if data.size == 0:
            continue

        target_id = pid or ckey
        rec.add_probe(name, target_id=target_id, interval=spec.get("interval", 1))
        rec.extend(name, data, target_id=target_id)

    return rec
