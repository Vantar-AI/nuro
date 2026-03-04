"""Akida neuron dynamics — maps Nuro IR dynamics to Akida layer types."""

from __future__ import annotations

from typing import Any

from nuro.ir.nodes import DynamicsNode


def get_akida_layer_config(node: DynamicsNode) -> dict[str, Any]:
    """Map a Nuro DynamicsNode to Akida layer configuration.

    Parameters
    ----------
    node : DynamicsNode
        The IR node specifying neuron dynamics.

    Returns
    -------
    dict
        Configuration for creating an Akida layer.

    Raises
    ------
    ValueError
        If the dynamics type is not supported on Akida.
    """
    if node.dynamics in ("lif", "if"):
        return {
            "type": "dense",
            "units": node.size,
            "activation": True,  # Spiking activation
        }
    elif node.dynamics == "izhikevich":
        raise ValueError(
            "Akida does not support Izhikevich neurons. "
            "Use LIF or IF dynamics for Akida deployment."
        )
    elif node.dynamics == "adex":
        raise ValueError(
            "Akida does not support AdEx neurons. "
            "Use LIF or IF dynamics for Akida deployment."
        )
    else:
        raise ValueError(
            f"Akida backend does not support dynamics '{node.dynamics}'. "
            f"Supported: 'lif', 'if'."
        )
