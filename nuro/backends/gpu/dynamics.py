"""GPU neuron dynamics — maps IR dynamics strings to SpikingJelly neuron layers."""

from __future__ import annotations

import torch.nn as nn
from spikingjelly.activation_based import neuron

from nuro.ir.nodes import DynamicsNode


def build_neuron_layer(node: DynamicsNode, dt: float) -> nn.Module:
    """Build a SpikingJelly neuron layer from an IR DynamicsNode.

    Parameters
    ----------
    node : DynamicsNode
        The IR node specifying dynamics and parameters.
    dt : float
        Simulation timestep in seconds. Used to convert time constants.

    Returns
    -------
    nn.Module
        A SpikingJelly neuron layer.
    """
    if node.dynamics == "lif":
        tau_sec = node.params.get("tau", 20e-3)
        tau_sj = tau_sec / dt
        return neuron.LIFNode(tau=tau_sj, step_mode="s")
    elif node.dynamics == "if":
        return neuron.IFNode(step_mode="s")
    else:
        raise ValueError(f"GPU backend does not support dynamics '{node.dynamics}'")
