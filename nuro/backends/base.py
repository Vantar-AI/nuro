"""Backend interface — base class for all compilation targets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nuro.ir import IRGraph


class CompiledModel(ABC):
    """A compiled, runnable model returned by a backend."""

    @abstractmethod
    def run(self, duration: float, dt: float = 1e-3) -> None:
        """Run the model for *duration* seconds with timestep *dt*."""

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
    def compile(self, ir_graph: IRGraph) -> CompiledModel:
        """Compile an IRGraph into a runnable model."""
