"""Loihi backend — Intel Loihi 2 deployment via Lava SDK."""

from __future__ import annotations

from typing import Any

import numpy as np

from nuro.backends.base import Backend, CompiledModel
from nuro.ir import IRGraph


class LoihiCompiledModel(CompiledModel):
    """A compiled model that runs on Lava (Loihi simulator or hardware)."""

    def __init__(
        self,
        neurons: dict[str, Any],
        synapses: dict[str, Any],
        ir_graph: IRGraph,
        dt: float,
    ) -> None:
        self._neurons = neurons
        self._synapses = synapses
        self._ir_graph = ir_graph
        self._dt = dt
        self._metrics: dict[str, Any] = {}
        self._run_cfg = None  # User-overrideable run config

        from nuro.backends.loihi.monitor import LoihiRecorder

        self._recorder = LoihiRecorder()

    def set_run_config(self, cfg: Any) -> None:
        """Override the default Loihi1SimCfg run configuration.

        Use ``Loihi2HwCfg()`` for hardware execution with INRC access.
        """
        self._run_cfg = cfg

    def run(self, duration: float, dt: float = 1e-3, batch_size: int = 1):
        """Run the model on Lava.

        Parameters
        ----------
        duration : float
            Simulation duration in seconds.
        dt : float
            Timestep in seconds.
        batch_size : int
            Must be 1 — Lava does not support batched execution.

        Returns
        -------
        None
        """
        if batch_size > 1:
            raise ValueError(
                "Loihi backend does not support batch_size > 1. "
                "Lava runs single-trial simulations."
            )

        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi1SimCfg

        num_steps = int(duration / dt)

        # Build input processes for source populations
        targets = {e.target_id for e in self._ir_graph.edges}
        source_ids = {nid for nid in self._ir_graph.nodes if nid not in targets}

        input_specs: dict[str, Any] = {}
        for inp in self._ir_graph.inputs:
            input_specs[inp.id] = inp

        driven_ids = source_ids | set(input_specs.keys())

        from nuro.backends.loihi.inputs import build_input_process, build_default_input

        input_processes: dict[str, Any] = {}
        for sid in driven_ids:
            spec = input_specs.get(sid)
            pop_size = self._ir_graph.nodes[sid].size
            if spec is not None:
                inp_proc = build_input_process(spec, num_steps, dt)
            else:
                inp_proc = build_default_input(pop_size, num_steps, dt)
            input_processes[sid] = inp_proc

        # Connect inputs to neurons
        for sid, inp_proc in input_processes.items():
            # Check if there are direct connections from this source
            has_outgoing = False
            for edge in self._ir_graph.edges:
                if edge.source_id == sid:
                    key = f"{edge.source_id}__{edge.target_id}"
                    inp_proc.s_out.connect(self._synapses[key].s_in)
                    has_outgoing = True

            if not has_outgoing:
                # Source with no outgoing edges — connect directly to neuron a_in
                if sid in self._neurons:
                    inp_proc.s_out.connect(self._neurons[sid].a_in)

        # Setup monitors if recording
        if self._recorder.has_probes:
            self._recorder.setup_monitors(
                num_steps, self._neurons, self._synapses
            )

        # Determine run config
        run_cfg = self._run_cfg
        if run_cfg is None:
            run_cfg = Loihi1SimCfg(select_tag="floating_pt")

        # Find root process (first input or first neuron)
        root = None
        if input_processes:
            root = next(iter(input_processes.values()))
        elif self._neurons:
            root = next(iter(self._neurons.values()))

        if root is None:
            self._metrics = {
                "total_spikes": 0,
                "spike_counts": {},
                "num_steps": num_steps,
                "duration": duration,
                "dt": dt,
                "batch_size": 1,
            }
            return None

        # Execute
        root.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_cfg)

        # Collect all probe data before stopping the runtime
        if self._recorder.has_probes:
            self._recorder.collect_all()

        # Collect metrics
        spike_counts: dict[str, int] = {}
        total_spikes = 0

        # Note: spike counting on Lava requires monitors.
        # If no spike probes, we report 0 (hardware has no free counters).
        for spec in self._recorder._probe_specs:
            if spec["name"] == "spikes" and spec["population_id"] is not None:
                pid = spec["population_id"]
                data = self._recorder.get(
                    "spikes", population_id=pid
                )
                count = int(np.sum(data))
                spike_counts[pid] = count
                total_spikes += count

        root.stop()

        self._metrics = {
            "total_spikes": total_spikes,
            "spike_counts": spike_counts,
            "num_steps": num_steps,
            "duration": duration,
            "dt": dt,
            "batch_size": 1,
        }
        return None

    def record(
        self,
        name: str,
        *,
        population: Any = None,
        connection: Any = None,
        interval: int = 1,
    ) -> None:
        """Register a state probe before calling ``run()``.

        Parameters
        ----------
        name : str
            ``"voltages"``, ``"spikes"``, or ``"weights"``.
        population : Population, optional
            Target population.
        connection : Connection, optional
            Target connection.
        interval : int
            Recording interval (steps between samples).
        """
        pop_id = getattr(population, "id", None)
        conn_key = None
        if connection is not None:
            conn_key = f"{connection.source.id}__{connection.target.id}"
        self._recorder.add_probe(
            name,
            population_id=pop_id,
            connection_key=conn_key,
            interval=interval,
        )

    def get_state(self, name: str, **kwargs: Any) -> Any:
        """Retrieve recorded state as a numpy array.

        Returns arrays in Nuro convention: ``(steps, neurons)`` for
        spikes/voltages, ``(steps, out, in)`` for weights.
        """
        population = kwargs.get("population")
        connection = kwargs.get("connection")
        pop_id = getattr(population, "id", None)
        conn_key = None
        if connection is not None:
            conn_key = f"{connection.source.id}__{connection.target.id}"
        return self._recorder.get(
            name, population_id=pop_id, connection_key=conn_key
        )

    def reset(self) -> None:
        self._recorder.reset()
        self._metrics = {}

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics


class LoihiBackend(Backend):
    """Intel Loihi backend using Lava SDK."""

    def compile(
        self, ir_graph: IRGraph, dt: float = 1e-3, **kwargs
    ) -> LoihiCompiledModel:
        """Compile an IRGraph into a Lava process graph.

        Parameters
        ----------
        ir_graph : IRGraph
            The intermediate representation to compile.
        dt : float
            Simulation timestep in seconds.
        **kwargs
            ``requires_grad`` — must be False (training not supported).
            ``weights_from`` — path to GPU checkpoint for weight transfer.

        Returns
        -------
        LoihiCompiledModel
        """
        requires_grad = kwargs.get("requires_grad", False)
        if requires_grad:
            raise ValueError(
                "Loihi backend does not support requires_grad=True. "
                "Train on GPU first, then deploy to Loihi with "
                "weights_from='checkpoint.pt'."
            )

        from lava.proc.dense.process import Dense

        from nuro.backends.loihi.dynamics import build_lava_neuron

        # Build neuron Processes
        neurons: dict[str, Any] = {}
        for nid, node in ir_graph.nodes.items():
            neurons[nid] = build_lava_neuron(node, dt)

        # Build synapse (Dense) Processes and connect ports
        synapses: dict[str, Any] = {}
        for edge in ir_graph.edges:
            key = f"{edge.source_id}__{edge.target_id}"
            source_node = ir_graph.nodes[edge.source_id]
            target_node = ir_graph.nodes[edge.target_id]

            # Initialize weights: (target_size, source_size)
            w_init = np.random.randn(
                target_node.size, source_node.size
            ).astype(np.float32) * 0.1

            dense = Dense(weights=w_init)
            synapses[key] = dense

            # Connect: source.s_out → dense.s_in, dense.a_out → target.a_in
            neurons[edge.source_id].s_out.connect(dense.s_in)
            dense.a_out.connect(neurons[edge.target_id].a_in)

        # Apply GPU weights if provided
        weights_from = kwargs.get("weights_from")
        if weights_from is not None:
            from nuro.backends.loihi.transfer import (
                apply_weights_to_lava,
                load_gpu_weights,
            )

            gpu_weights = load_gpu_weights(weights_from)
            scale_factor = kwargs.get("scale_factor", 1.0)
            quantize = kwargs.get("quantize", False)
            num_bits = kwargs.get("num_bits", 8)
            apply_weights_to_lava(
                gpu_weights, synapses, scale_factor,
                quantize=quantize, num_bits=num_bits,
            )

        return LoihiCompiledModel(neurons, synapses, ir_graph, dt)
