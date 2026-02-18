"""HTTP client for Vantar Cloud API.

Reads VANTAR_API_KEY from environment or set explicitly via set_api_key().
"""

from __future__ import annotations

import os
import time
from typing import Any

DEFAULT_ENDPOINT = "https://api.vantar.xyz"

_api_key: str | None = None


def set_api_key(key: str) -> None:
    """Set the Vantar Cloud API key programmatically.

    Alternative to setting the VANTAR_API_KEY environment variable.
    """
    global _api_key
    _api_key = key


def _get_api_key() -> str:
    key = _api_key or os.environ.get("VANTAR_API_KEY")
    if not key:
        raise RuntimeError(
            "Vantar Cloud API key not found.\n"
            "Set it via:\n"
            "  export VANTAR_API_KEY=vt_your_key_here\n"
            "  # or\n"
            "  nuro.set_api_key('vt_your_key_here')\n"
            "  # or\n"
            "  nuro.compile(graph, target='cloud', api_key='vt_your_key_here')"
        )
    return key


def _headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key or _get_api_key()
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def compile_graph(
    ir_graph_json: dict[str, Any],
    hardware: str,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
) -> str:
    """Submit an IRGraph for compilation and return job_id.

    Args:
        ir_graph_json: Serialized IRGraph dict.
        hardware: Target hardware ("loihi" | "spinnaker2").
        endpoint: API base URL.
        api_key: Override API key.

    Returns:
        job_id string.

    Raises:
        RuntimeError: On HTTP error or missing API key.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for the cloud backend: pip install requests")

    url = f"{endpoint}/v1/compile"
    payload = {"ir_graph": ir_graph_json, "hardware": hardware}
    resp = requests.post(url, json=payload, headers=_headers(api_key), timeout=30)

    if not resp.ok:
        raise RuntimeError(
            f"Vantar Cloud compile failed: {resp.status_code} {resp.text}"
        )

    data = resp.json()
    return data["job_id"]


def poll_job(
    job_id: str,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
    timeout_s: float = 300.0,
    poll_interval_s: float = 2.0,
) -> dict[str, Any]:
    """Poll until job reaches 'compiled' or 'error' status.

    Args:
        job_id: Job identifier from compile_graph().
        endpoint: API base URL.
        api_key: Override API key.
        timeout_s: Maximum seconds to wait.
        poll_interval_s: Seconds between polls.

    Returns:
        Final job status dict.

    Raises:
        RuntimeError: On error status or timeout.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for the cloud backend: pip install requests")

    url = f"{endpoint}/v1/jobs/{job_id}"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp = requests.get(url, headers=_headers(api_key), timeout=10)
        if not resp.ok:
            raise RuntimeError(f"Vantar Cloud status check failed: {resp.status_code} {resp.text}")

        data = resp.json()
        status = data.get("status")

        if status == "compiled":
            return data
        if status == "error":
            raise RuntimeError(f"Vantar Cloud compilation error: {data.get('error')}")

        time.sleep(poll_interval_s)

    raise RuntimeError(f"Vantar Cloud job {job_id} timed out after {timeout_s}s")


def execute_job(
    job_id: str,
    duration: float,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
) -> str:
    """Trigger hardware execution and return run_id.

    Args:
        job_id: Compiled job identifier.
        duration: Simulation duration in seconds.
        endpoint: API base URL.
        api_key: Override API key.

    Returns:
        run_id string.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for the cloud backend: pip install requests")

    url = f"{endpoint}/v1/execute/{job_id}"
    resp = requests.post(url, json={"duration": duration}, headers=_headers(api_key), timeout=30)

    if not resp.ok:
        raise RuntimeError(f"Vantar Cloud execute failed: {resp.status_code} {resp.text}")

    return resp.json()["run_id"]


def get_results(
    run_id: str,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Poll until run completes and return results.

    Args:
        run_id: Run identifier from execute_job().
        endpoint: API base URL.
        api_key: Override API key.
        timeout_s: Maximum wait time.
        poll_interval_s: Seconds between polls.

    Returns:
        Dict with "metrics" and "spike_data" keys.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for the cloud backend: pip install requests")

    url = f"{endpoint}/v1/results/{run_id}"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp = requests.get(url, headers=_headers(api_key), timeout=10)
        if not resp.ok:
            raise RuntimeError(f"Vantar Cloud results failed: {resp.status_code} {resp.text}")

        data = resp.json()
        if data.get("status") == "complete":
            return data

        time.sleep(poll_interval_s)

    raise RuntimeError(f"Vantar Cloud run {run_id} timed out after {timeout_s}s")
