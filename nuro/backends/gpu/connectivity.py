"""GPU connectivity — maps IR edge patterns to PyTorch layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from nuro.ir.edges import SynapticEdge
from nuro.ir.nodes import DynamicsNode


def build_synapse_layer(
    edge: SynapticEdge,
    source_node: DynamicsNode,
    target_node: DynamicsNode,
) -> nn.Linear:
    """Build a PyTorch Linear layer from an IR SynapticEdge.

    Parameters
    ----------
    edge : SynapticEdge
        The IR edge specifying pattern and parameters.
    source_node : DynamicsNode
        Pre-synaptic population.
    target_node : DynamicsNode
        Post-synaptic population.

    Returns
    -------
    nn.Linear
        A linear layer (possibly with a sparsity mask applied).
    """
    in_features = source_node.size
    out_features = target_node.size

    layer = nn.Linear(in_features, out_features, bias=False)

    # Scale weights so downstream neurons receive enough current to fire.
    # With Poisson input at ~50Hz (prob ~0.05), ~5% of pre neurons spike
    # per step, contributing ~in_features*0.05 spikes. We want the summed
    # synaptic current to be around 2x the firing threshold (1.0).
    expected_active = max(in_features * 0.05, 1.0)
    weight_scale = 3.0 / expected_active

    if edge.pattern == "dense":
        nn.init.uniform_(layer.weight, 0.0, weight_scale)
    elif edge.pattern == "random_sparse":
        sparsity = edge.params.get("sparsity", 0.8)
        nn.init.uniform_(layer.weight, 0.0, weight_scale)
        mask = (torch.rand_like(layer.weight) > sparsity).float()
        with torch.no_grad():
            layer.weight.mul_(mask)
        layer.register_buffer("_sparsity_mask", mask)
    else:
        raise ValueError(f"GPU backend does not support pattern '{edge.pattern}'")

    return layer
