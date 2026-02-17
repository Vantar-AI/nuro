"""GPU backend — SpikingJelly wrapper for development and training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from nuro.backends.base import Backend, CompiledModel
from nuro.backends.gpu.connectivity import build_synapse_layer
from nuro.backends.gpu.dynamics import build_neuron_layer
from nuro.backends.gpu.plasticity import STDPUpdater
from nuro.backends.gpu.recorders import Recorder
from nuro.ir import IRGraph
from nuro.ir.edges import SynapticEdge


class NuroSNN(nn.Module):
    """A spiking neural network built from an IRGraph."""

    def __init__(self, ir_graph: IRGraph, dt: float) -> None:
        super().__init__()
        self.ir_graph = ir_graph
        self.dt = dt
        self._is_cyclic = ir_graph.is_cyclic
        if self._is_cyclic:
            self.pop_order = list(ir_graph.nodes.keys())
        else:
            self.pop_order = ir_graph.topological_order()

        # Build neuron layers
        self.neurons = nn.ModuleDict()
        for nid, node in ir_graph.nodes.items():
            self.neurons[nid] = build_neuron_layer(node, dt)

        # Build synapse layers
        self.synapses = nn.ModuleDict()
        self._edge_map: dict[str, SynapticEdge] = {}
        for edge in ir_graph.edges:
            key = f"{edge.source_id}__{edge.target_id}"
            source_node = ir_graph.nodes[edge.source_id]
            target_node = ir_graph.nodes[edge.target_id]
            self.synapses[key] = build_synapse_layer(edge, source_node, target_node)
            self._edge_map[key] = edge

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        source_ids: set[str],
        prev_spikes: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward one timestep.

        Parameters
        ----------
        inputs : dict mapping population id → input tensor
            External spike trains for source populations.
        source_ids : set of str
            IDs of source populations (no incoming edges). These
            populations use the input tensor directly as spikes
            rather than feeding it through their neuron model.
        prev_spikes : dict, optional
            Previous timestep's spikes for all populations. Used only
            for cyclic graphs (Jacobi iteration).

        Returns
        -------
        dict mapping population id → spike tensor
        """
        spikes: dict[str, torch.Tensor] = {}

        # For cyclic graphs, use prev_spikes as the source for synaptic
        # input lookups (Jacobi: all populations see the *previous* step).
        # For DAGs, use the current step's spikes (already computed in
        # topological order).
        lookup = prev_spikes if (self._is_cyclic and prev_spikes) else spikes

        for nid in self.pop_order:
            if nid in source_ids and nid in inputs:
                spikes[nid] = inputs[nid]
                continue

            device = self._get_device()
            x = torch.zeros(self.ir_graph.nodes[nid].size, device=device)

            for src_id in self.pop_order:
                key = f"{src_id}__{nid}"
                if key in self.synapses and src_id in lookup:
                    x = x + self.synapses[key](lookup[src_id])

            spikes[nid] = self.neurons[nid](x)

        return spikes

    def _get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")


class GPUCompiledModel(CompiledModel):
    """A compiled model that runs on GPU (or CPU fallback) via SpikingJelly."""

    def __init__(self, snn: NuroSNN, ir_graph: IRGraph) -> None:
        self._snn = snn
        self._ir_graph = ir_graph
        self._metrics: dict[str, Any] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._snn.to(self._device)
        self._recorder = Recorder()

        # Build STDP updaters for edges with plasticity="stdp"
        self._stdp_updaters: list[tuple[str, str, STDPUpdater]] = []
        for edge in ir_graph.edges:
            if edge.plasticity == "stdp":
                key = f"{edge.source_id}__{edge.target_id}"
                synapse = self._snn.synapses[key]
                updater = STDPUpdater(synapse)
                self._stdp_updaters.append((edge.source_id, edge.target_id, updater))

    def run(self, duration: float, dt: float = 1e-3) -> None:
        """Run the model for *duration* seconds with timestep *dt*."""
        num_steps = int(duration / dt)
        functional.reset_net(self._snn)
        for updater_tuple in self._stdp_updaters:
            updater_tuple[2].reset()

        # Identify source populations (no incoming edges)
        targets = {e.target_id for e in self._ir_graph.edges}
        source_id_set = {nid for nid in self._ir_graph.nodes if nid not in targets}

        # Build input lookup from IR inputs
        input_specs: dict[str, Any] = {}
        for inp in self._ir_graph.inputs:
            input_specs[inp.id] = inp

        # Populations that need external input: source pops (no incoming edges)
        # plus any pop with an explicit Input spec (even in cyclic graphs)
        driven_ids = source_id_set | set(input_specs.keys())

        spike_counts: dict[str, int] = {nid: 0 for nid in self._ir_graph.nodes}
        total_spikes = 0
        prev_spikes: dict[str, torch.Tensor] | None = None

        with torch.no_grad():
            for _step in range(num_steps):
                inputs: dict[str, torch.Tensor] = {}
                for sid in driven_ids:
                    spec = input_specs.get(sid)
                    if spec is not None and spec.data is not None:
                        # Static tensor input
                        row = min(_step, spec.data.shape[0] - 1)
                        inputs[sid] = spec.data[row].to(self._device).float()
                    elif spec is not None and spec.generator is not None:
                        # Generator input
                        inputs[sid] = spec.generator(_step).to(self._device).float()
                    else:
                        # Poisson mode (default or explicit)
                        rate = spec.rate if spec is not None else 50.0
                        size = self._ir_graph.nodes[sid].size
                        prob = rate * dt
                        spk = (torch.rand(size, device=self._device) < prob).float()
                        inputs[sid] = spk

                spikes = self._snn(inputs, driven_ids, prev_spikes)
                prev_spikes = spikes

                # STDP update
                for src_id, tgt_id, updater in self._stdp_updaters:
                    updater.step(spikes[src_id], spikes[tgt_id])

                # State recording
                if self._recorder.has_probes:
                    self._recorder.record_step(_step, spikes, self._snn)

                # Accumulate metrics
                for nid, spk in spikes.items():
                    count = int(spk.sum().item())
                    spike_counts[nid] += count
                    total_spikes += count

        self._metrics = {
            "total_spikes": total_spikes,
            "spike_counts": spike_counts,
            "num_steps": num_steps,
            "duration": duration,
            "dt": dt,
        }

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
            What to record: ``"voltages"``, ``"spikes"``, or ``"weights"``.
        population : Population, optional
            Target population (for ``"voltages"`` and ``"spikes"``).
        connection : Connection, optional
            Target connection (for ``"weights"``).
        interval : int
            Record every *interval* steps.
        """
        pop_id = getattr(population, "id", None)
        conn_key = None
        if connection is not None:
            conn_key = f"{connection.source.id}__{connection.target.id}"
        self._recorder.add_probe(
            name, population_id=pop_id, connection_key=conn_key, interval=interval
        )

    def get_state(self, name: str, **kwargs: Any) -> Any:
        """Retrieve recorded state as a tensor.

        Parameters
        ----------
        name : str
            The probe name (e.g. ``"voltages"``).
        population : Population, optional
            Target population.
        connection : Connection, optional
            Target connection.

        Returns
        -------
        torch.Tensor
            Shape ``(num_recordings, ...)``.
        """
        population = kwargs.get("population")
        connection = kwargs.get("connection")
        pop_id = getattr(population, "id", None)
        conn_key = None
        if connection is not None:
            conn_key = f"{connection.source.id}__{connection.target.id}"
        return self._recorder.get(name, population_id=pop_id, connection_key=conn_key)

    def save(self, path: str) -> None:
        """Save model weights and graph to a checkpoint file.

        Parameters
        ----------
        path : str
            File path (e.g. ``"checkpoint.pt"``).
        """
        from nuro.backends.gpu.checkpoint import save_checkpoint

        save_checkpoint(self, path)

    def reset(self) -> None:
        functional.reset_net(self._snn)
        for updater_tuple in self._stdp_updaters:
            updater_tuple[2].reset()
        self._recorder.reset()
        self._metrics = {}

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics


class GPUBackend(Backend):
    """GPU backend using SpikingJelly (activation-based mode)."""

    def compile(self, ir_graph: IRGraph, dt: float = 1e-3) -> GPUCompiledModel:
        snn = NuroSNN(ir_graph, dt)
        return GPUCompiledModel(snn, ir_graph)
