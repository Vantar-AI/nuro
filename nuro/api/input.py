"""Input specifications — how external data enters a spiking network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from nuro.api.population import Population


@dataclass
class Input:
    """An input source attached to a population.

    Supports three modes:

    1. **Static tensor** — pre-computed spike/current data::

        Input(population, data=tensor)   # shape (num_steps, pop_size)

    2. **Generator** — callable yielding one tensor per timestep::

        Input(population, generator=fn)  # fn(step) → tensor of shape (pop_size,)

    3. **Poisson** — random spike trains at a given rate::

        Input(population, mode="poisson", rate=100.0)

    Parameters
    ----------
    population : Population
        The target population to receive this input.
    data : tensor-like, optional
        Static input of shape ``(num_steps, pop_size)``.
    generator : callable, optional
        A function ``f(step: int) → tensor`` producing one frame per call.
    mode : str, optional
        Built-in generation mode. Currently only ``"poisson"`` is supported.
    rate : float
        Firing rate in Hz (only used when ``mode="poisson"``).
    """

    population: Population
    data: Any = None
    generator: Callable[..., Any] | None = None
    mode: str | None = None
    rate: float = 50.0
    id: str = field(default_factory=lambda: "", repr=False)

    def __post_init__(self) -> None:
        # Assign id from population for easy lookup
        self.id = self.population.id

        # Validate that exactly one input source is specified
        sources = sum([
            self.data is not None,
            self.generator is not None,
            self.mode is not None,
        ])
        if sources == 0:
            # Default to Poisson mode
            self.mode = "poisson"
        elif sources > 1:
            raise ValueError(
                "Input accepts exactly one of: data, generator, or mode. "
                f"Got {sources} sources."
            )

        if self.mode is not None and self.mode != "poisson":
            raise ValueError(
                f"Unsupported input mode '{self.mode}'. Supported: 'poisson'."
            )
