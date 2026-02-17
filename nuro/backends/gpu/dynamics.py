"""GPU neuron dynamics — maps IR dynamics strings to SpikingJelly neuron layers."""

from __future__ import annotations

import torch.nn as nn
from spikingjelly.activation_based import neuron

from nuro.backends.gpu.neurons import AdExNode, IzhikevichNode
from nuro.ir.nodes import DynamicsNode


def build_neuron_layer(node: DynamicsNode, dt: float) -> nn.Module:
    """Build a neuron layer from an IR DynamicsNode.

    Parameters
    ----------
    node : DynamicsNode
        The IR node specifying dynamics and parameters.
    dt : float
        Simulation timestep in seconds. Used to convert time constants.

    Returns
    -------
    nn.Module
        A neuron layer (SpikingJelly for LIF/IF, custom for Izh/AdEx).
    """
    if node.dynamics == "lif":
        tau_sec = node.params.get("tau", 20e-3)
        tau_sj = tau_sec / dt
        return neuron.LIFNode(tau=tau_sj, step_mode="s")
    elif node.dynamics == "if":
        return neuron.IFNode(step_mode="s")
    elif node.dynamics == "izhikevich":
        # Accept preset name or raw a/b/c/d params
        preset = node.params.get("preset")
        if preset and preset in IzhikevichNode.PRESETS:
            kw = dict(IzhikevichNode.PRESETS[preset])
        else:
            kw = {}
        for k in ("a", "b", "c", "d", "v_thresh"):
            if k in node.params:
                kw[k] = node.params[k]
        return IzhikevichNode(size=node.size, dt=dt, **kw)
    elif node.dynamics == "adex":
        kw = {}
        for k in ("C", "g_L", "E_L", "V_T", "delta_T", "a", "b", "tau_w", "V_reset", "V_cutoff"):
            if k in node.params:
                kw[k] = node.params[k]
        return AdExNode(size=node.size, dt=dt, **kw)
    else:
        raise ValueError(f"GPU backend does not support dynamics '{node.dynamics}'")
