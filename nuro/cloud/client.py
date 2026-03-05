"""Cloud experiment storage client - push/pull experiments to Vantar Cloud.

Coming in Nuro v0.9. Sign up at https://vantar.xyz/cloud to get notified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nuro.backends.cloud.client import _get_api_key, _headers, DEFAULT_ENDPOINT

_COMING_SOON = (
    "\n"
    "Vantar Cloud experiment storage is coming in v0.9.\n"
    "  - Push experiments to the cloud\n"
    "  - Share recordings with collaborators\n"
    "  - Compare runs across hardware platforms\n"
    "  - Remote access to neuromorphic chips\n"
    "\n"
    "Sign up for early access: https://vantar.xyz/cloud\n"
    "\n"
    "For now, use Experiment.save() / Experiment.load() for local persistence."
)


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
    raise NotImplementedError(_COMING_SOON)


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
    raise NotImplementedError(_COMING_SOON)


def list_experiments(
    project: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """List experiments on Vantar Cloud."""
    raise NotImplementedError(_COMING_SOON)


def compare_experiments(
    experiment_ids: list[str],
    metric: str = "spikes",
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Compare experiments on Vantar Cloud."""
    raise NotImplementedError(_COMING_SOON)
