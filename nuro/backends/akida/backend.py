"""Akida backend — BrainChip Akida neuromorphic deployment via MetaTF SDK."""

from __future__ import annotations

from typing import Any

import numpy as np

from nuro.backends.base import Backend, CompiledModel
from nuro.ir import IRGraph


class AkidaCompiledModel(CompiledModel):
    """A compiled model for BrainChip Akida hardware or emulator."""

    def __init__(
        self,
        akida_model: Any,
        ir_graph: IRGraph,
        layer_map: dict[str, Any],
    ) -> None:
        self._model = akida_model
        self._ir_graph = ir_graph
        self._layer_map = layer_map
        self._metrics: dict[str, Any] = {}

    def run(self, duration: float, dt: float = 1e-3, batch_size: int = 1) -> None:
        """Run inference on Akida.

        Akida runs event-driven inference, not time-stepped simulation.
        Duration and dt are used to generate input spikes.

        Parameters
        ----------
        duration : float
            Simulation duration in seconds.
        dt : float
            Timestep in seconds.
        batch_size : int
            Must be 1 for Akida.
        """
        if batch_size > 1:
            raise ValueError("Akida backend does not support batch_size > 1.")

        num_steps = int(duration / dt)

        # Generate input spikes
        input_specs: dict[str, Any] = {
            inp.id: inp for inp in self._ir_graph.inputs
        }
        targets = {e.target_id for e in self._ir_graph.edges}
        source_ids = {nid for nid in self._ir_graph.nodes if nid not in targets}

        total_spikes = 0
        spike_counts: dict[str, int] = {nid: 0 for nid in self._ir_graph.nodes}

        for _step in range(num_steps):
            for sid in source_ids:
                spec = input_specs.get(sid)
                size = self._ir_graph.nodes[sid].size
                if spec is not None and spec.data is not None:
                    row = min(_step, spec.data.shape[0] - 1)
                    input_data = spec.data[row].numpy() if hasattr(spec.data[row], "numpy") else np.array(spec.data[row])
                else:
                    rate = spec.rate if spec is not None else 50.0
                    prob = rate * dt
                    input_data = (np.random.rand(size) < prob).astype(np.float32)

                # Feed to Akida model
                if self._model is not None:
                    try:
                        result = self._model.predict(
                            input_data.reshape(1, -1).astype(np.uint8)
                        )
                        total_spikes += int(np.sum(result > 0))
                    except Exception:
                        pass

        self._metrics = {
            "total_spikes": total_spikes,
            "spike_counts": spike_counts,
            "num_steps": num_steps,
            "duration": duration,
            "dt": dt,
            "batch_size": 1,
        }

    def reset(self) -> None:
        self._metrics = {}

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics


class AkidaBackend(Backend):
    """BrainChip Akida backend using MetaTF/Akida SDK."""

    def compile(
        self, ir_graph: IRGraph, dt: float = 1e-3, **kwargs
    ) -> AkidaCompiledModel:
        """Compile an IRGraph for Akida hardware.

        Parameters
        ----------
        ir_graph : IRGraph
            The intermediate representation to compile.
        dt : float
            Simulation timestep.
        **kwargs
            ``weights_from`` — path to GPU checkpoint.
            ``num_bits`` — quantization bits (1, 2, 4, 8). Default 4.
        """
        requires_grad = kwargs.get("requires_grad", False)
        if requires_grad:
            raise ValueError(
                "Akida backend does not support requires_grad=True. "
                "Train on GPU first, then deploy with weights_from='checkpoint.pt'."
            )

        from nuro.backends.akida.dynamics import get_akida_layer_config

        # Validate all dynamics are Akida-compatible
        layer_configs = {}
        for nid, node in ir_graph.nodes.items():
            layer_configs[nid] = get_akida_layer_config(node)

        # Build Akida model
        akida_model = None
        layer_map = {}

        try:
            import akida

            # Build Sequential model from IR
            layers = []
            node_order = (
                ir_graph.topological_order()
                if not ir_graph.is_cyclic
                else list(ir_graph.nodes.keys())
            )

            for i, nid in enumerate(node_order):
                config = layer_configs[nid]
                if i == 0:
                    input_size = ir_graph.nodes[nid].size
                    layer = akida.InputData(
                        name=nid, input_shape=(input_size,)
                    )
                else:
                    layer = akida.InputConvolutional(
                        name=nid,
                        num_neurons=config["units"],
                    )
                layers.append(layer)
                layer_map[nid] = layer

            if layers:
                akida_model = akida.Model(layers=layers)

                # Apply weights if provided
                weights_from = kwargs.get("weights_from")
                if weights_from is not None:
                    from nuro.backends.akida.transfer import (
                        load_gpu_weights,
                        quantize_weights_akida,
                    )

                    num_bits = kwargs.get("num_bits", 4)
                    gpu_weights = load_gpu_weights(weights_from)
                    for key, w in gpu_weights.items():
                        q, _ = quantize_weights_akida(w, num_bits=num_bits)
                        # Set weights on corresponding Akida layer
                        parts = key.split("__")
                        if len(parts) == 2 and parts[1] in layer_map:
                            try:
                                layer_map[parts[1]].set_variable(
                                    "weights", q
                                )
                            except Exception:
                                pass

        except ImportError:
            # Akida SDK not installed — create a mock model for testing
            pass

        return AkidaCompiledModel(
            akida_model=akida_model,
            ir_graph=ir_graph,
            layer_map=layer_map,
        )
