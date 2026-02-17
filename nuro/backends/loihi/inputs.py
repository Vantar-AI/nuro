"""Loihi input system — maps Nuro Input specs to Lava RingBuffer Processes."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_input_process(input_spec: Any, num_steps: int, dt: float):
    """Build a Lava RingBuffer input Process from a Nuro Input spec.

    Parameters
    ----------
    input_spec : nuro.api.input.Input or None
        The input specification. ``None`` triggers default Poisson at 50 Hz.
    num_steps : int
        Number of simulation timesteps.
    dt : float
        Simulation timestep in seconds.

    Returns
    -------
    lava.proc.io.source.RingBuffer
        A RingBuffer Process pre-loaded with spike data.

    Raises
    ------
    ValueError
        If the input spec uses a generator (not supported on Loihi).
    """
    from lava.proc.io.source import RingBuffer

    if input_spec is None:
        # Default Poisson at 50 Hz
        size = None  # Caller must provide size
        raise ValueError(
            "build_input_process requires an input_spec or explicit size. "
            "Use build_default_input() for default Poisson inputs."
        )

    # Generator mode — not supported on hardware
    if input_spec.generator is not None:
        raise ValueError(
            "Generator inputs are not supported on the Loihi backend. "
            "Pre-compute your input as a static tensor or use Poisson mode."
        )

    size = input_spec.population.size

    if input_spec.data is not None:
        # Static tensor: shape (steps, neurons) → (neurons, steps) for Lava
        data = np.array(input_spec.data, dtype=float)
        if data.ndim == 2:
            # (steps, neurons) → (neurons, steps)
            data = data.T
        elif data.ndim == 3:
            # (steps, batch, neurons) — batch not supported, take first
            data = data[:, 0, :].T
        return RingBuffer(data=data)

    # Poisson mode (explicit or default)
    rate = input_spec.rate if input_spec.rate is not None else 50.0
    prob = rate * dt
    spike_train = (np.random.rand(size, num_steps) < prob).astype(float)
    return RingBuffer(data=spike_train)


def build_default_input(size: int, num_steps: int, dt: float, rate: float = 50.0):
    """Build a default Poisson RingBuffer for source populations without Input specs.

    Parameters
    ----------
    size : int
        Population size (number of neurons).
    num_steps : int
        Number of simulation timesteps.
    dt : float
        Timestep in seconds.
    rate : float
        Firing rate in Hz.

    Returns
    -------
    lava.proc.io.source.RingBuffer
        A RingBuffer with pre-generated Poisson spike data.
    """
    from lava.proc.io.source import RingBuffer

    prob = rate * dt
    spike_train = (np.random.rand(size, num_steps) < prob).astype(float)
    return RingBuffer(data=spike_train)
