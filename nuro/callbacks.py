"""Training callbacks for MLOps integration.

Provides callback hooks that integrate with Weights & Biases and
TensorBoard for experiment tracking during SNN training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Callback(ABC):
    """Base class for training callbacks."""

    def on_run_start(self, config: dict[str, Any]) -> None:
        """Called at the start of model.run()."""

    def on_step(self, step: int, spikes: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Called after each simulation timestep."""

    def on_run_end(self, metrics: dict[str, Any]) -> None:
        """Called at the end of model.run()."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Called at the end of a training epoch (user-invoked)."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""


class WandbCallback(Callback):
    """Weights & Biases logging callback.

    Parameters
    ----------
    project : str
        W&B project name.
    config : dict, optional
        Experiment configuration to log.
    log_interval : int
        Log metrics every N steps. Default 10.
    """

    def __init__(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        log_interval: int = 10,
    ) -> None:
        self.log_interval = log_interval
        self._run = None

        try:
            import wandb
            self._wandb = wandb
            self._run = wandb.init(project=project, config=config or {})
        except ImportError:
            raise ImportError(
                "wandb is required for WandbCallback. "
                "Install with: pip install wandb"
            )

    def on_step(self, step: int, spikes: dict[str, Any], metrics: dict[str, Any]) -> None:
        if step % self.log_interval == 0:
            log_data = {"step": step}
            for nid, spk in spikes.items():
                count = int(spk.detach().sum().item()) if hasattr(spk, "detach") else 0
                log_data[f"spikes/{nid}"] = count
            log_data.update(metrics)
            self._wandb.log(log_data)

    def on_run_end(self, metrics: dict[str, Any]) -> None:
        if self._run:
            self._wandb.log({"run/" + k: v for k, v in metrics.items()})

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        if self._run:
            self._wandb.log({"epoch": epoch, **metrics})

    def close(self) -> None:
        if self._run:
            self._run.finish()


class TensorBoardCallback(Callback):
    """TensorBoard logging callback.

    Parameters
    ----------
    log_dir : str
        Directory for TensorBoard logs.
    log_interval : int
        Log metrics every N steps. Default 10.
    """

    def __init__(self, log_dir: str = "runs/nuro", log_interval: int = 10) -> None:
        self.log_interval = log_interval

        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            raise ImportError(
                "tensorboard is required for TensorBoardCallback. "
                "Install with: pip install tensorboard"
            )

        self._global_step = 0

    def on_step(self, step: int, spikes: dict[str, Any], metrics: dict[str, Any]) -> None:
        if step % self.log_interval == 0:
            for nid, spk in spikes.items():
                count = int(spk.detach().sum().item()) if hasattr(spk, "detach") else 0
                self._writer.add_scalar(f"spikes/{nid}", count, self._global_step)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(f"metrics/{k}", v, self._global_step)
        self._global_step += 1

    def on_run_end(self, metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"run/{k}", v, self._global_step)
        self._writer.flush()

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"epoch/{k}", v, epoch)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class PrintCallback(Callback):
    """Simple print-based callback for debugging.

    Parameters
    ----------
    log_interval : int
        Print every N steps. Default 100.
    """

    def __init__(self, log_interval: int = 100) -> None:
        self.log_interval = log_interval

    def on_run_start(self, config: dict[str, Any]) -> None:
        print(f"[nuro] Run started: {config}")

    def on_step(self, step: int, spikes: dict[str, Any], metrics: dict[str, Any]) -> None:
        if step % self.log_interval == 0:
            total = sum(
                int(s.detach().sum().item()) if hasattr(s, "detach") else 0
                for s in spikes.values()
            )
            print(f"[nuro] Step {step}: {total} spikes")

    def on_run_end(self, metrics: dict[str, Any]) -> None:
        print(f"[nuro] Run complete: {metrics.get('total_spikes', 0)} total spikes")

    def close(self) -> None:
        pass
