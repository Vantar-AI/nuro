"""Cloud experiment storage client — push/pull experiments to Vantar Cloud.

Deferred to v0.9.0. This module provides the API surface; actual cloud
endpoints are not yet available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nuro.backends.cloud.client import _get_api_key, _headers, DEFAULT_ENDPOINT


def push_experiment(
    experiment_dir: str | Path,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
) -> str:
    """Upload a saved experiment to Vantar Cloud.

    Parameters
    ----------
    experiment_dir : path
        Local directory from ``Experiment.save()``.
    endpoint : str
        Vantar Cloud API endpoint.
    api_key : str, optional
        Override API key.

    Returns
    -------
    str
        Remote experiment ID.
    """
    raise NotImplementedError(
        "Cloud experiment storage is coming in Nuro v0.9. "
        "Use Experiment.save() for local persistence."
    )


def pull_experiment(
    experiment_id: str,
    target_dir: str | Path = ".",
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
) -> Path:
    """Download an experiment from Vantar Cloud.

    Returns
    -------
    Path
        Local directory containing the experiment.
    """
    raise NotImplementedError(
        "Cloud experiment storage is coming in Nuro v0.9. "
        "Use Experiment.load() for local persistence."
    )


def list_experiments(
    project: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """List experiments on Vantar Cloud."""
    raise NotImplementedError(
        "Cloud experiment storage is coming in Nuro v0.9."
    )


def compare_experiments(
    experiment_ids: list[str],
    metric: str = "spikes",
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Compare experiments on Vantar Cloud."""
    raise NotImplementedError(
        "Cloud experiment storage is coming in Nuro v0.9."
    )
