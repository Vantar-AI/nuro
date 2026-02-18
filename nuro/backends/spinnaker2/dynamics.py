"""SpiNNaker 2 neuron dynamics — maps IR DynamicsNode to snn.Population."""

from __future__ import annotations

import warnings
from typing import Any

from nuro.ir.nodes import DynamicsNode


# Izhikevich presets (same values as GPU backend, duplicated to avoid
# importing torch here)
_IZHIKEVICH_PRESETS: dict[str, dict[str, float]] = {
    "regular_spiking": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
    "intrinsically_bursting": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
    "chattering": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
    "fast_spiking": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
    "low_threshold_spiking": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},
}


def build_s2_population(node: DynamicsNode, dt: float, record: bool = False) -> Any:
    """Build a py-spinnaker2 Population from an IR DynamicsNode.

    Parameters
    ----------
    node : DynamicsNode
        The IR node specifying dynamics and parameters.
    dt : float
        Simulation timestep in seconds.
    record : bool
        Whether to enable spike + voltage recording on this population.

    Returns
    -------
    snn.Population
    """
    if node.dynamics in ("lif", "if"):
        return _build_lif_population(node, dt, record)
    elif node.dynamics in ("izhikevich", "adex"):
        warnings.warn(
            f"SpiNNaker 2 natively supports LIF neurons only. "
            f"Population '{node.id}' uses '{node.dynamics}' which will be "
            f"approximated as LIF. Use LIF/IF for accurate hardware deployment.",
            stacklevel=2,
        )
        return _build_lif_population(node, dt, record)
    else:
        raise ValueError(
            f"SpiNNaker 2 backend does not support dynamics '{node.dynamics}'. "
            f"Supported: lif, if (izhikevich/adex run as LIF approximation)."
        )


def _build_lif_population(node: DynamicsNode, dt: float, record: bool) -> Any:
    """Map Nuro LIF/IF → py-spinnaker2 LIF Population."""
    from spinnaker2 import snn

    tau_sec = node.params.get("tau", 20e-3)
    v_thresh = node.params.get("v_thresh", 1.0)

    # SpiNNaker2 uses a discrete alpha_decay factor per timestep:
    # v[t+1] = alpha_decay * v[t] + input
    # For IF (no leak): alpha_decay = 1.0
    # For LIF: alpha_decay = exp(-dt / tau)
    import math
    if node.dynamics == "if":
        alpha_decay = 1.0
    else:
        alpha_decay = math.exp(-dt / tau_sec)

    params = {
        "threshold": float(v_thresh),
        "alpha_decay": alpha_decay,
        "i_offset": 0.0,
        "v_reset": 0.0,
        "reset": "reset_by_subtraction",
    }

    record_vars = ["spikes", "v"] if record else []

    return snn.Population(
        size=node.size,
        neuron_model="lif",
        params=params,
        name=node.id,
        record=record_vars,
    )
