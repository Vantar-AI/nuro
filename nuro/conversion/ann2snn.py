"""ANN-to-SNN conversion — convert trained PyTorch models to spiking networks.

Implements rate-based conversion where ReLU activations become IF neuron
firing rates, and ANN weights transfer directly to synaptic weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.population import Population


def convert_ann(
    model: Any,
    input_shape: tuple,
    num_steps: int = 100,
) -> Graph:
    """Convert a PyTorch ANN to a Nuro SNN Graph.

    Walks the model's children and maps:
    - ``nn.Linear`` → Population(dynamics="if") + Connection(pattern="dense")
    - ``nn.Conv2d`` → flattened dense connection (spatial mapping in future)
    - ``nn.ReLU`` → absorbed into IF neuron threshold
    - ``nn.BatchNorm1d/2d`` → folded into preceding linear weights

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model. Supports Sequential and flat module lists.
    input_shape : tuple
        Input tensor shape excluding batch dimension, e.g. ``(784,)`` for MNIST.
    num_steps : int
        Number of simulation timesteps for rate coding. Higher values give
        better accuracy but slower inference. Default 100.

    Returns
    -------
    Graph
        Nuro Graph ready for ``nuro.compile()``. Connection params include
        ``"weights"`` extracted from the ANN for weight transfer.
    """
    import torch.nn as nn

    layers = _extract_layers(model)
    populations: list[Population] = []
    connections: list[Connection] = []

    # Create input population
    input_size = int(np.prod(input_shape))
    input_pop = Population(size=input_size, dynamics="if")
    populations.append(input_pop)

    prev_pop = input_pop

    for layer in layers:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

            out_size = layer.out_features
            pop = Population(size=out_size, dynamics="if")
            populations.append(pop)

            conn = Connection(
                source=prev_pop,
                target=pop,
                pattern="dense",
                params={"weights": weight, "bias": bias, "num_steps": num_steps},
            )
            connections.append(conn)
            prev_pop = pop

        elif isinstance(layer, nn.Conv2d):
            # Flatten conv2d to dense for initial implementation
            weight = layer.weight.detach().cpu().numpy()
            # Reshape conv weights to 2D: (out_channels * spatial, in_channels * spatial)
            out_channels = weight.shape[0]

            # For simplicity: flatten to dense connection
            out_size = out_channels  # simplified - full spatial in future version
            flat_weight = weight.reshape(out_channels, -1)

            # Adjust input size to match
            in_size = flat_weight.shape[1]
            if prev_pop.size != in_size:
                # Create adapter population
                prev_pop = Population(size=in_size, dynamics="if")
                populations.append(prev_pop)

            pop = Population(size=out_size, dynamics="if")
            populations.append(pop)

            bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
            conn = Connection(
                source=prev_pop,
                target=pop,
                pattern="dense",
                params={"weights": flat_weight, "bias": bias, "num_steps": num_steps},
            )
            connections.append(conn)
            prev_pop = pop

        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # Fold BN into the most recent connection's weights
            if connections:
                last_conn = connections[-1]
                w = last_conn.params.get("weights")
                b = last_conn.params.get("bias")
                if w is not None:
                    new_w, new_b = _fold_batchnorm(w, b, layer)
                    last_conn.params["weights"] = new_w
                    last_conn.params["bias"] = new_b

        elif isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
            # Activation absorbed into IF neuron dynamics
            pass

        elif isinstance(layer, (nn.Flatten, nn.Dropout, nn.Dropout2d)):
            # Skip structural/regularization layers
            pass

        elif isinstance(layer, nn.MaxPool2d):
            # Skip pooling for now
            pass

    return Graph(populations=populations, connections=connections)


def normalize_weights(
    graph: Graph,
    method: str = "robust",
    percentile: float = 99.0,
    target_range: tuple[float, float] = (-1.0, 1.0),
) -> Graph:
    """Normalize connection weights for hardware deployment.

    Parameters
    ----------
    graph : Graph
        Nuro Graph with weight params from ``convert_ann()``.
    method : str
        Normalization method: ``"robust"`` (percentile-based),
        ``"max"`` (divide by max absolute value).
    percentile : float
        Percentile for robust scaling. Default 99.0.
    target_range : tuple
        Target weight range after normalization.

    Returns
    -------
    Graph
        New Graph with normalized weights.
    """
    new_connections = []
    for conn in graph.connections:
        weights = conn.params.get("weights")
        if weights is None:
            new_connections.append(conn)
            continue

        w = np.array(weights, dtype=np.float32)

        if method == "robust":
            p = np.percentile(np.abs(w), percentile)
            if p > 0:
                w = w / p
        elif method == "max":
            m = np.max(np.abs(w))
            if m > 0:
                w = w / m

        # Clamp to target range
        w = np.clip(w, target_range[0], target_range[1])

        new_params = dict(conn.params)
        new_params["weights"] = w
        new_conn = Connection(
            source=conn.source,
            target=conn.target,
            pattern=conn.pattern,
            plasticity=conn.plasticity,
            params=new_params,
        )
        new_connections.append(new_conn)

    return Graph(
        populations=graph.populations,
        connections=new_connections,
        inputs=graph.inputs if graph.inputs else None,
    )


def _extract_layers(model: Any) -> list:
    """Recursively extract leaf nn.Module layers from a model."""
    layers = []
    for child in model.children():
        if len(list(child.children())) > 0:
            layers.extend(_extract_layers(child))
        else:
            layers.append(child)
    return layers


def _fold_batchnorm(
    weight: np.ndarray,
    bias: np.ndarray | None,
    bn: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm parameters into preceding linear layer weights.

    BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Combined with linear W*x + b:
    y = gamma/sqrt(var+eps) * W * x + gamma/sqrt(var+eps) * b - gamma*mean/sqrt(var+eps) + beta
    """
    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    eps = bn.eps

    scale = gamma / np.sqrt(var + eps)

    # Scale weights
    new_weight = weight * scale[:, np.newaxis]

    # Scale and shift bias
    if bias is not None:
        new_bias = scale * bias - scale * mean + beta
    else:
        new_bias = -scale * mean + beta

    return new_weight, new_bias
