"""GPU connectivity — maps IR edge patterns to PyTorch layers."""

from __future__ import annotations

import math

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
    elif edge.pattern == "one_to_one":
        # Identity/diagonal connectivity: each source neuron connects to
        # exactly one target neuron. Requires source_size == target_size.
        if in_features != out_features:
            raise ValueError(
                f"one_to_one pattern requires equal population sizes, "
                f"got {in_features} → {out_features}"
            )
        with torch.no_grad():
            layer.weight.copy_(torch.eye(out_features) * weight_scale)
        mask = torch.eye(out_features)
        layer.register_buffer("_sparsity_mask", mask)
    elif edge.pattern == "conv1d":
        # 1D convolutional connectivity with kernel_size and stride
        kernel_size = edge.params.get("kernel_size", 3)
        stride = edge.params.get("stride", 1)
        with torch.no_grad():
            layer.weight.zero_()
            for out_idx in range(out_features):
                center = out_idx * stride
                start = max(0, center - kernel_size // 2)
                end = min(in_features, center + kernel_size // 2 + 1)
                for in_idx in range(start, end):
                    layer.weight[out_idx, in_idx] = torch.empty(1).uniform_(
                        0.0, weight_scale
                    ).item()
    elif edge.pattern == "distance_dependent":
        # Gaussian probability connectivity based on neuron distance
        sigma = edge.params.get("sigma", 0.3)
        with torch.no_grad():
            for i in range(out_features):
                for j in range(in_features):
                    # Normalized distance
                    di = i / max(out_features - 1, 1)
                    dj = j / max(in_features - 1, 1)
                    dist = abs(di - dj)
                    prob = math.exp(-(dist**2) / (2 * sigma**2))
                    if torch.rand(1).item() < prob:
                        layer.weight[i, j] = torch.empty(1).uniform_(
                            0.0, weight_scale
                        ).item()
                    else:
                        layer.weight[i, j] = 0.0
    else:
        raise ValueError(f"GPU backend does not support pattern '{edge.pattern}'")

    return layer
