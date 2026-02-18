"""SpiNNaker 2 input system — maps Nuro Input specs to spike_list Populations."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_input_population(
    input_spec: Any,
    num_steps: int,
    dt: float,
    pop_id: str,
) -> Any:
    """Build a py-spinnaker2 spike_list Population from a Nuro Input spec.

    Parameters
    ----------
    input_spec : nuro.api.input.Input
        The input specification.
    num_steps : int
        Number of simulation timesteps.
    dt : float
        Simulation timestep in seconds.
    pop_id : str
        Population ID (used as the name).

    Returns
    -------
    snn.Population with neuron_model="spike_list"

    Raises
    ------
    ValueError
        If the input spec uses a generator (not supported on SpiNNaker 2).
    """
    from spinnaker2 import snn

    if input_spec.generator is not None:
        raise ValueError(
            "Generator inputs are not supported on the SpiNNaker 2 backend. "
            "Pre-compute your input as a static tensor or use Poisson mode."
        )

    size = input_spec.population.size

    if input_spec.data is not None:
        # Static tensor: shape (steps, neurons)
        data = np.array(input_spec.data, dtype=float)
        if data.ndim == 3:
            data = data[:, 0, :]  # drop batch dim
        spike_dict = _dense_to_spike_dict(data, size)
    else:
        # Poisson mode
        rate = input_spec.rate if input_spec.rate is not None else 50.0
        prob = rate * dt
        spike_matrix = (np.random.rand(num_steps, size) < prob)
        spike_dict = _dense_to_spike_dict(spike_matrix.astype(float), size)

    return snn.Population(
        size=size,
        neuron_model="spike_list",
        params=spike_dict,
        name=f"{pop_id}_input",
    )


def build_default_input_population(
    size: int,
    num_steps: int,
    dt: float,
    pop_id: str,
    rate: float = 50.0,
) -> Any:
    """Build a default Poisson spike_list Population.

    Parameters
    ----------
    size : int
        Number of neurons.
    num_steps : int
        Number of simulation timesteps.
    dt : float
        Timestep in seconds.
    pop_id : str
        Population ID (used as the name).
    rate : float
        Poisson firing rate in Hz. Default 50 Hz.

    Returns
    -------
    snn.Population with neuron_model="spike_list"
    """
    from spinnaker2 import snn

    prob = rate * dt
    spike_matrix = (np.random.rand(num_steps, size) < prob).astype(float)
    spike_dict = _dense_to_spike_dict(spike_matrix, size)

    return snn.Population(
        size=size,
        neuron_model="spike_list",
        params=spike_dict,
        name=f"{pop_id}_input",
    )


def _dense_to_spike_dict(
    spike_matrix: np.ndarray,
    size: int,
) -> dict[int, list[int]]:
    """Convert a (steps, neurons) spike matrix to py-spinnaker2 spike_list format.

    py-spinnaker2 expects: ``{neuron_idx: [timestep, ...], ...}``
    """
    spike_dict: dict[int, list[int]] = {}
    for neuron_idx in range(size):
        timesteps = list(np.where(spike_matrix[:, neuron_idx] > 0.5)[0].astype(int))
        if timesteps:
            spike_dict[neuron_idx] = timesteps
    return spike_dict
