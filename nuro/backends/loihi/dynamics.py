"""Loihi neuron dynamics — maps IR DynamicsNode to Lava neuron Processes."""

from __future__ import annotations

import warnings
from typing import Any

from nuro.ir.nodes import DynamicsNode


def build_lava_neuron(node: DynamicsNode, dt: float) -> Any:
    """Build a Lava neuron Process from an IR DynamicsNode.

    Parameters
    ----------
    node : DynamicsNode
        The IR node specifying dynamics and parameters.
    dt : float
        Simulation timestep in seconds.

    Returns
    -------
    lava Process
        A Lava Process with ``a_in`` (InPort), ``s_out`` (OutPort),
        and ``v`` (Var) attributes.
    """
    if node.dynamics == "lif":
        return _build_lif(node, dt)
    elif node.dynamics == "if":
        return _build_if(node, dt)
    elif node.dynamics == "izhikevich":
        warnings.warn(
            f"Izhikevich neuron '{node.id}' uses a simulation-only Lava "
            "ProcessModel. It cannot run on Loihi hardware. Use LIF/IF "
            "for hardware deployment.",
            stacklevel=2,
        )
        return _build_izhikevich(node, dt)
    elif node.dynamics == "adex":
        warnings.warn(
            f"AdEx neuron '{node.id}' uses a simulation-only Lava "
            "ProcessModel. It cannot run on Loihi hardware. Use LIF/IF "
            "for hardware deployment.",
            stacklevel=2,
        )
        return _build_adex(node, dt)
    else:
        raise ValueError(
            f"Loihi backend does not support dynamics '{node.dynamics}'"
        )


def _build_lif(node: DynamicsNode, dt: float):
    """Map Nuro LIF → Lava LIF Process."""
    from lava.proc.lif.process import LIF

    tau_sec = node.params.get("tau", 20e-3)
    v_thresh = node.params.get("v_thresh", 1.0)

    # Convert continuous tau to Lava decay: dv = dt / tau
    dv = dt / tau_sec

    # du: current decay. Default to same as dv for standard LIF.
    du = node.params.get("du", dv)

    return LIF(
        shape=(node.size,),
        dv=dv,
        du=du,
        vth=v_thresh,
        bias_mant=0,
        bias_exp=0,
    )


def _build_if(node: DynamicsNode, dt: float):
    """Map Nuro IF → Lava LIF with no leak (dv=0, du=0)."""
    from lava.proc.lif.process import LIF

    v_thresh = node.params.get("v_thresh", 1.0)

    return LIF(
        shape=(node.size,),
        dv=0.0,
        du=0.0,
        vth=v_thresh,
        bias_mant=0,
        bias_exp=0,
    )


def _build_izhikevich(node: DynamicsNode, dt: float):
    """Map Nuro Izhikevich → custom Lava Process (simulation only)."""
    from nuro.backends.loihi._custom_neurons import build_izhikevich_process

    # Resolve preset if present
    from nuro.backends.gpu.neurons import IzhikevichNode

    params = dict(node.params)
    preset = params.pop("preset", None)
    if preset and preset in IzhikevichNode.PRESETS:
        resolved = dict(IzhikevichNode.PRESETS[preset])
        resolved.update(params)
        params = resolved

    return build_izhikevich_process(node.size, params, dt)


def _build_adex(node: DynamicsNode, dt: float):
    """Map Nuro AdEx → custom Lava Process (simulation only)."""
    from nuro.backends.loihi._custom_neurons import build_adex_process

    return build_adex_process(node.size, dict(node.params), dt)
