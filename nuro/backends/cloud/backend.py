"""Cloud backend — compile and run on Vantar Cloud API.

Usage::

    model = nuro.compile(
        graph,
        target="cloud",
        hardware="loihi",
        weights_from="trained.pt",
        api_key="vt_...",       # or VANTAR_API_KEY env var
        endpoint="https://api.vantar.xyz",  # optional override
    )
    model.run(duration=1.0)
    print(model.metrics)
"""

from __future__ import annotations

from typing import Any

from nuro.backends.base import Backend, CompiledModel
from nuro.backends.cloud import client, serializer
from nuro.ir import IRGraph


class CloudCompiledModel(CompiledModel):
    """A compiled model running on Vantar Cloud hardware.

    Returned by CloudBackend.compile(). Holds the job_id from the
    compilation step. run() triggers hardware execution and fetches results.
    """

    def __init__(
        self,
        job_id: str,
        endpoint: str,
        api_key: str | None,
    ) -> None:
        self._job_id = job_id
        self._endpoint = endpoint
        self._api_key = api_key
        self._metrics: dict[str, Any] = {}

    def run(self, duration: float, dt: float = 1e-3, batch_size: int = 1) -> None:
        """Execute on cloud hardware and block until results are ready.

        Args:
            duration: Simulation duration in seconds.
            dt: Timestep (informational — hardware uses native dt).
            batch_size: Not supported on hardware (always 1).

        Returns:
            None (results available via model.metrics).
        """
        if batch_size != 1:
            raise ValueError("Cloud backend does not support batch_size > 1.")

        run_id = client.execute_job(
            self._job_id,
            duration=duration,
            endpoint=self._endpoint,
            api_key=self._api_key,
        )

        results = client.get_results(
            run_id,
            endpoint=self._endpoint,
            api_key=self._api_key,
        )

        self._metrics = results.get("metrics", {})
        self._spike_data = results.get("spike_data", {})

    def reset(self) -> None:
        """No-op — state reset happens on the cloud server between runs."""
        self._metrics = {}

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics

    def record(self, name: str, **kwargs: Any) -> None:
        """Recording is configured via cloud API, not locally.

        Currently all spike data is returned in metrics automatically.
        Fine-grained recording control is planned for v0.8.
        """
        # No-op for now — spike_data returned in results by default
        pass

    def get_state(self, name: str, **kwargs: Any) -> Any:
        """Retrieve spike data from the last run.

        Args:
            name: State name — currently only "spikes" supported.

        Returns:
            Spike data dict or None.
        """
        if name == "spikes":
            return self._spike_data.get("spikes")
        raise NotImplementedError(
            f"Cloud backend get_state('{name}') not yet supported. "
            "Supported: 'spikes'"
        )


class CloudBackend(Backend):
    """Backend that compiles and runs via Vantar Cloud API.

    Serializes the IRGraph to JSON, POSTs to the Vantar Cloud API,
    polls for compilation completion, and returns a CloudCompiledModel
    that can trigger hardware execution.
    """

    def compile(self, ir_graph: IRGraph, **kwargs) -> CloudCompiledModel:
        """Compile IRGraph to cloud job.

        Args:
            ir_graph: Network intermediate representation.
            hardware: Target chip ("loihi" | "spinnaker2"). Default: "loihi".
            api_key: Vantar Cloud API key. Falls back to VANTAR_API_KEY env var.
            endpoint: API base URL. Default: https://api.vantar.xyz.
            weights_from: Path to GPU checkpoint for weight transfer (passed in metadata).

        Returns:
            CloudCompiledModel with job_id ready for .run().
        """
        hardware = kwargs.get("hardware", "loihi")
        api_key = kwargs.get("api_key")
        endpoint = kwargs.get("endpoint", client.DEFAULT_ENDPOINT)
        weights_from = kwargs.get("weights_from")

        if hardware not in ("loihi", "spinnaker2"):
            raise ValueError(
                f"Unsupported cloud hardware '{hardware}'. "
                "Choose 'loihi' or 'spinnaker2'."
            )

        # Serialize graph to JSON
        ir_json = serializer.serialize_ir_graph(ir_graph)

        # Attach weight transfer metadata if provided
        if weights_from:
            ir_json["weights_from"] = weights_from

        # Submit compilation job
        print(f"[Vantar Cloud] Submitting compile job (hardware={hardware})...")
        job_id = client.compile_graph(
            ir_graph_json=ir_json,
            hardware=hardware,
            endpoint=endpoint,
            api_key=api_key,
        )
        print(f"[Vantar Cloud] Job created: {job_id}")

        # Poll until compiled
        print("[Vantar Cloud] Waiting for compilation...")
        client.poll_job(job_id, endpoint=endpoint, api_key=api_key)
        print("[Vantar Cloud] Compilation complete. Ready to run.")

        return CloudCompiledModel(job_id=job_id, endpoint=endpoint, api_key=api_key)
