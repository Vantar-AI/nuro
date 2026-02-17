"""Backend interface — base class for all compilation targets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nuro.ir import IRGraph


class CompiledModel(ABC):
    """A compiled, runnable model returned by a backend."""

    @abstractmethod
    def run(self, duration: float, dt: float = 1e-3, batch_size: int = 1):
        """Run the model for *duration* seconds with timestep *dt*.

        Parameters
        ----------
        duration : float
            Simulation duration in seconds.
        dt : float
            Timestep in seconds.
        batch_size : int
            Number of parallel trials. When 1 (default), tensors have no
            batch dimension for full backward compatibility. When > 1,
            all internal tensors gain a leading ``(batch,)`` dimension.

        Returns
        -------
        None or dict[str, Tensor]
            When ``requires_grad=True``, returns a dict mapping population
            id to output spike tensors (needed for loss computation).
            Otherwise returns ``None``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state."""

    @property
    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """Return metrics from the last run."""

    def record(
        self,
        name: str,
        *,
        population: Any = None,
        connection: Any = None,
        interval: int = 1,
    ) -> None:
        """Register a state probe. Override in subclasses."""
        raise NotImplementedError("This backend does not support recording.")

    def get_state(self, name: str, **kwargs: Any) -> Any:
        """Retrieve recorded state. Override in subclasses."""
        raise NotImplementedError("This backend does not support recording.")


class Backend(ABC):
    """Base class for compilation backends."""

    @abstractmethod
    def compile(self, ir_graph: IRGraph, **kwargs) -> CompiledModel:
        """Compile an IRGraph into a runnable model."""
