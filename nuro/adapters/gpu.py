"""Bridge from GPU Recorder to hardware-agnostic Recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nuro.recording import Recording

if TYPE_CHECKING:
    from nuro.backends.gpu.recorders import Recorder


def recording_from_gpu_model(
    recorder: Recorder,
    dt: float = 1e-3,
) -> Recording:
    """Convert a GPU backend Recorder into a Recording.

    Parameters
    ----------
    recorder : nuro.backends.gpu.recorders.Recorder
        A populated GPU Recorder after ``model.run()``.
    dt : float
        Timestep in seconds.

    Returns
    -------
    Recording
        A new Recording with all probes copied over.
    """
    rec = Recording(dt=dt, source="gpu")

    for probe in recorder._probes:
        name = probe["name"]
        pid = probe.get("population_id")
        ckey = probe.get("connection_key")

        tensor = recorder.get(name, population_id=pid, connection_key=ckey)
        if tensor.numel() == 0:
            continue

        data = tensor.numpy()
        target_id = pid or ckey
        rec.add_probe(name, target_id=target_id, interval=probe.get("interval", 1))
        rec.extend(name, data, target_id=target_id)

    return rec
