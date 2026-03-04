"""SpiNNaker 2 backend — py-spinnaker2 deployment via Brian2 sim or hardware."""

from __future__ import annotations

from typing import Any

import numpy as np

from nuro.backends.base import Backend, CompiledModel
from nuro.ir import IRGraph


class SpiNNaker2CompiledModel(CompiledModel):
    """A compiled model that runs on py-spinnaker2 (sim or hardware)."""

    def __init__(
        self,
        populations: dict[str, Any],
        projections: dict[str, Any],
        input_populations: dict[str, Any],
        network: Any,
        ir_graph: IRGraph,
        dt: float,
    ) -> None:
        self._populations = populations        # nid → snn.Population
        self._projections = projections        # "src__tgt" → snn.Projection
        self._input_populations = input_populations  # nid → spike_list Population
        self._network = network                # snn.Network
        self._ir_graph = ir_graph
        self._dt = dt
        self._metrics: dict[str, Any] = {}
        self._hw: Any = None   # hardware.SpiNNaker2Chip — set via set_hardware()
        self._recorded_pops: set[str] = set()

    def set_hardware(self, hw: Any) -> None:
        """Set a hardware target for real SpiNNaker 2 execution.

        Parameters
        ----------
        hw : spinnaker2.hardware.SpiNNaker2Chip
            Configured hardware object, e.g.
            ``hardware.SpiNNaker2Chip(eth_ip="192.168.1.1")``.
            When not set, the Brian2 simulator is used.
        """
        self._hw = hw

    def run(self, duration: float, dt: float = 1e-3, batch_size: int = 1) -> None:
        """Run the model.

        Parameters
        ----------
        duration : float
            Simulation duration in seconds.
        dt : float
            Timestep in seconds (ignored — uses dt set at compile time).
        batch_size : int
            Must be 1 — SpiNNaker 2 does not support batched execution.
        """
        if batch_size > 1:
            raise ValueError(
                "SpiNNaker 2 backend does not support batch_size > 1."
            )

        num_steps = int(duration / self._dt)

        if self._hw is not None:
            # Real hardware execution
            self._hw.run(self._network, timesteps=num_steps)
        else:
            # Brian2 simulator fallback
            from spinnaker2 import brian2_sim
            brian2_sim.run(self._network, timesteps=num_steps)

        # Collect spike metrics from recorded populations
        spike_counts: dict[str, int] = {}
        total_spikes = 0

        for nid in self._recorded_pops:
            pop = self._populations.get(nid)
            if pop is None:
                continue
            try:
                spikes = pop.get_spikes()  # {neuron_idx: [timestep, ...]}
                count = sum(len(v) for v in spikes.values())
            except Exception:
                count = 0
            spike_counts[nid] = count
            total_spikes += count

        self._metrics = {
            "total_spikes": total_spikes,
            "spike_counts": spike_counts,
            "num_steps": num_steps,
            "duration": duration,
            "dt": self._dt,
            "batch_size": 1,
        }

    def record(
        self,
        name: str,
        *,
        population: Any = None,
        connection: Any = None,
        interval: int = 1,
    ) -> None:
        """Register a population for spike/voltage recording.

        Note: recording must be enabled at compile time via
        ``nuro.compile(..., record_all=True)`` or this method
        only marks populations for post-run retrieval.
        """
        if population is not None:
            self._recorded_pops.add(population.id)

    def get_state(self, name: str, **kwargs: Any) -> Any:
        """Retrieve recorded state after run.

        Parameters
        ----------
        name : str
            ``"spikes"`` or ``"voltages"``.

        Returns
        -------
        dict
            ``{neuron_idx: [timestep, ...]}`` for spikes,
            ``{neuron_idx: np.ndarray}`` for voltages.
        """
        population = kwargs.get("population")
        if population is None:
            raise ValueError("get_state requires a population= argument.")

        pop = self._populations.get(population.id)
        if pop is None:
            raise KeyError(f"Population '{population.id}' not found.")

        if name == "spikes":
            return pop.get_spikes()
        elif name == "voltages":
            return pop.get_voltages()
        else:
            raise ValueError(f"Unknown state '{name}'. Use 'spikes' or 'voltages'.")

    def reset(self) -> None:
        """Reset metrics and recorded state."""
        self._metrics = {}
        self._recorded_pops = set()

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics


class SpiNNaker2Backend(Backend):
    """SpiNNaker 2 backend using py-spinnaker2."""

    def compile(
        self,
        ir_graph: IRGraph,
        dt: float = 1e-3,
        **kwargs,
    ) -> SpiNNaker2CompiledModel:
        """Compile an IRGraph into a py-spinnaker2 Network.

        Parameters
        ----------
        ir_graph : IRGraph
            The intermediate representation to compile.
        dt : float
            Simulation timestep in seconds. Default 1ms.
        **kwargs
            ``requires_grad`` — must be False (training not supported).
            ``weights_from`` — path to GPU checkpoint for weight transfer.
            ``quantize`` — quantize weights to S2 integer range [-15, 15].
            ``scale_factor`` — manual weight scale override.
            ``delay`` — synaptic delay in timesteps [0, 7]. Default 0.
            ``record_all`` — enable spike+voltage recording on all pops.
            ``online_learning`` — enable on-chip learning rules on ARM cores.
        """
        requires_grad = kwargs.get("requires_grad", False)
        if requires_grad:
            raise ValueError(
                "SpiNNaker 2 backend does not support requires_grad=True. "
                "Train on GPU first, then deploy with weights_from='checkpoint.pt'."
            )

        from spinnaker2 import snn

        from nuro.backends.spinnaker2.dynamics import build_s2_population
        from nuro.backends.spinnaker2.connections import build_s2_projection

        delay = int(kwargs.get("delay", 0))
        record_all = bool(kwargs.get("record_all", False))

        # Build neuron Populations
        populations: dict[str, Any] = {}
        for nid, node in ir_graph.nodes.items():
            populations[nid] = build_s2_population(node, dt, record=record_all)

        # Load GPU weights if provided
        weights_map: dict[str, np.ndarray] = {}
        weights_from = kwargs.get("weights_from")
        if weights_from is not None:
            from nuro.backends.spinnaker2.transfer import (
                load_gpu_weights,
                quantize_weights_s2,
            )

            raw_weights = load_gpu_weights(weights_from)
            quantize = kwargs.get("quantize", True)  # default True for S2
            scale_factor = kwargs.get("scale_factor", None)

            for key, w in raw_weights.items():
                if quantize:
                    q, _ = quantize_weights_s2(w, scale_factor=scale_factor)
                    weights_map[key] = q.astype(float)
                else:
                    weights_map[key] = (
                        w * scale_factor if scale_factor is not None else w
                    )

        # Build Projections
        projections: dict[str, Any] = {}
        for edge in ir_graph.edges:
            key = f"{edge.source_id}__{edge.target_id}"
            weight_matrix = weights_map.get(key)
            projections[key] = build_s2_projection(
                edge,
                ir_graph.nodes[edge.source_id],
                ir_graph.nodes[edge.target_id],
                populations[edge.source_id],
                populations[edge.target_id],
                weight_matrix=weight_matrix,
                delay=delay,
            )

        # Build input Populations for source nodes
        from nuro.backends.spinnaker2.inputs import (
            build_input_population,
            build_default_input_population,
        )

        input_specs: dict[str, Any] = {inp.id: inp for inp in ir_graph.inputs}
        targets = {e.target_id for e in ir_graph.edges}
        ff_source_ids = {nid for nid in ir_graph.nodes if nid not in targets}

        if ff_source_ids:
            driven_ids = ff_source_ids | set(input_specs.keys())
        else:
            driven_ids = set(input_specs.keys()) or set(ir_graph.nodes.keys())

        # Determine num_steps from a representative duration for input gen
        # Use 100 steps as default — actual steps used at run() time
        _default_steps = 100
        input_populations: dict[str, Any] = {}
        for nid in driven_ids:
            spec = input_specs.get(nid)
            size = ir_graph.nodes[nid].size
            if spec is not None:
                input_populations[nid] = build_input_population(
                    spec, _default_steps, dt, nid
                )
            else:
                input_populations[nid] = build_default_input_population(
                    size, _default_steps, dt, nid
                )

        # Build Projections from input pops to driven populations
        input_projections: dict[str, Any] = {}
        for nid, inp_pop in input_populations.items():
            key = f"_input__{nid}"
            size = ir_graph.nodes[nid].size
            # Identity weight matrix: one input neuron drives one output neuron
            w = np.eye(size, dtype=float)
            input_projections[key] = snn.Projection(
                pre=inp_pop,
                post=populations[nid],
                connections=weights_to_connection_list_identity(size, delay=delay),
            )

        # Assemble Network
        net = snn.Network(f"nuro_network")
        for pop in populations.values():
            net.add(pop)
        for pop in input_populations.values():
            net.add(pop)
        for proj in projections.values():
            net.add(proj)
        for proj in input_projections.values():
            net.add(proj)

        return SpiNNaker2CompiledModel(
            populations=populations,
            projections=projections,
            input_populations=input_populations,
            network=net,
            ir_graph=ir_graph,
            dt=dt,
        )


def weights_to_connection_list_identity(size: int, delay: int = 0) -> list:
    """Build a 1-to-1 identity connection list for input → neuron wiring."""
    delay = max(0, min(7, delay))
    return [[i, i, 1.0, delay] for i in range(size)]
