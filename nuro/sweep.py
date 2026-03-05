"""Parameter sweep for analog neuromorphic experiments.

Analog researchers don't train with gradient descent. They sweep physical
parameters (bias currents, thresholds, time constants) and compare spike
patterns across settings. ParameterSweep organizes this workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nuro.experiment import Experiment
from nuro.recording import Recording


class ParameterSweep:
    """A series of experiments varying one parameter.

    Parameters
    ----------
    name : str
        Sweep name.
    parameter : str
        Name of the parameter being swept (e.g. "bias_current_nA").
    unit : str
        Unit of the parameter (e.g. "nA", "mV").
    """

    def __init__(self, name: str, parameter: str, unit: str = "") -> None:
        self.name = name
        self.parameter = parameter
        self.unit = unit
        self._runs: list[dict[str, Any]] = []

    def add_run(
        self,
        value: float,
        experiment: Experiment | None = None,
        recording: Recording | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Register one run at a parameter value.

        Provide either an Experiment or a Recording (or both).
        """
        run: dict[str, Any] = {"value": value, "metrics": metrics or {}}
        if experiment is not None:
            run["experiment"] = experiment
            if not metrics:
                run["metrics"] = experiment.metrics
        if recording is not None:
            run["recording"] = recording
        self._runs.append(run)

    @property
    def values(self) -> np.ndarray:
        """Parameter values across all runs."""
        return np.array([r["value"] for r in self._runs])

    @property
    def num_runs(self) -> int:
        return len(self._runs)

    def summary(self) -> list[dict[str, Any]]:
        """Return list of {value, metrics} for all runs."""
        return [{"value": r["value"], "metrics": r["metrics"]} for r in self._runs]

    def metric_array(self, metric: str) -> np.ndarray:
        """Extract one metric across all runs as an array."""
        return np.array([r["metrics"].get(metric, np.nan) for r in self._runs])

    def best(self, metric: str, mode: str = "max") -> dict[str, Any]:
        """Find the run with the best metric value.

        Parameters
        ----------
        mode : "max" or "min"

        Returns
        -------
        dict with "value", "metric_value", "index"
        """
        arr = self.metric_array(metric)
        if np.all(np.isnan(arr)):
            return {"value": None, "metric_value": None, "index": -1}
        idx = int(np.nanargmax(arr) if mode == "max" else np.nanargmin(arr))
        return {
            "value": self._runs[idx]["value"],
            "metric_value": float(arr[idx]),
            "index": idx,
        }

    def plot(self, metric: str, ax: Any = None, title: str | None = None) -> Any:
        """Plot metric vs parameter value."""
        from nuro.plot import _import_matplotlib
        plt = _import_matplotlib()

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        values = self.values
        metrics = self.metric_array(metric)

        ax.plot(values, metrics, "o-", color="steelblue", markersize=5)
        ax.set_xlabel(f"{self.parameter}" + (f" ({self.unit})" if self.unit else ""))
        ax.set_ylabel(metric)
        ax.set_title(title or f"{self.name}: {metric} vs {self.parameter}")
        return ax

    def save(self, directory: str | Path) -> Path:
        """Persist sweep to disk.

        Creates::

            <directory>/<name>/
                sweep.json
                run_000/  (experiment dir, if experiment attached)
                run_001/
                ...
        """
        base = Path(directory) / self.name
        base.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.name,
            "parameter": self.parameter,
            "unit": self.unit,
            "runs": [],
        }

        for i, run in enumerate(self._runs):
            run_meta: dict[str, Any] = {
                "value": run["value"],
                "metrics": run["metrics"],
            }
            if "experiment" in run:
                run_dir = str(run["experiment"].save(str(base)))
                run_meta["experiment_dir"] = str(Path(run_dir).name)
            meta["runs"].append(run_meta)

        (base / "sweep.json").write_text(json.dumps(meta, indent=2, default=str))
        return base

    @classmethod
    def load(cls, directory: str | Path) -> ParameterSweep:
        """Load a sweep from disk."""
        base = Path(directory)
        meta = json.loads((base / "sweep.json").read_text())

        sweep = cls(
            name=meta["name"],
            parameter=meta["parameter"],
            unit=meta.get("unit", ""),
        )

        for run_meta in meta["runs"]:
            exp = None
            if "experiment_dir" in run_meta:
                exp_path = base / run_meta["experiment_dir"]
                if exp_path.exists():
                    exp = Experiment.load(exp_path)

            sweep.add_run(
                value=run_meta["value"],
                experiment=exp,
                metrics=run_meta.get("metrics", {}),
            )

        return sweep

    def __repr__(self) -> str:
        return (
            f"ParameterSweep(name={self.name!r}, parameter={self.parameter!r}, "
            f"runs={self.num_runs})"
        )
