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
from nuro.ir import IRGraph
from nuro.ir.edges import SynapticEdge


class NuroSNN(nn.Module):
    """A spiking neural network built from an IRGraph."""

    def __init__(self, ir_graph: IRGraph, dt: float) -> None:
        super().__init__()
        self.ir_graph = ir_graph
        self.dt = dt
        self.pop_order = list(ir_graph.nodes.keys())

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
    ) -> dict[str, torch.Tensor]:
        """Forward one timestep.

        Parameters
        ----------
        inputs : dict mapping population id → input tensor
            External Poisson spike trains for source populations.
        source_ids : set of str
            IDs of source populations (no incoming edges). These
            populations use the input tensor directly as spikes
            rather than feeding it through their neuron model.

        Returns
        -------
        dict mapping population id → spike tensor
        """
        spikes: dict[str, torch.Tensor] = {}

        for nid in self.pop_order:
            if nid in source_ids and nid in inputs:
                # Source populations: use Poisson spikes directly
                spikes[nid] = inputs[nid]
                continue

            # Gather synaptic input from upstream populations
            device = self._get_device()
            x = torch.zeros(self.ir_graph.nodes[nid].size, device=device)

            for src_id in self.pop_order:
                key = f"{src_id}__{nid}"
                if key in self.synapses and src_id in spikes:
                    x = x + self.synapses[key](spikes[src_id])

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

        spike_counts: dict[str, int] = {nid: 0 for nid in self._ir_graph.nodes}
        total_spikes = 0

        with torch.no_grad():
            for _step in range(num_steps):
                # Generate Poisson spike trains for source populations.
                # Each neuron fires independently with probability rate * dt.
                inputs: dict[str, torch.Tensor] = {}
                for sid in source_id_set:
                    size = self._ir_graph.nodes[sid].size
                    rate = 50.0  # 50 Hz Poisson rate
                    prob = rate * dt
                    spk = (torch.rand(size, device=self._device) < prob).float()
                    inputs[sid] = spk

                spikes = self._snn(inputs, source_id_set)

                # STDP update
                for src_id, tgt_id, updater in self._stdp_updaters:
                    updater.step(spikes[src_id], spikes[tgt_id])

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

    def reset(self) -> None:
        functional.reset_net(self._snn)
        for updater_tuple in self._stdp_updaters:
            updater_tuple[2].reset()
        self._metrics = {}

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics


class GPUBackend(Backend):
    """GPU backend using SpikingJelly (activation-based mode)."""

    def compile(self, ir_graph: IRGraph, dt: float = 1e-3) -> GPUCompiledModel:
        snn = NuroSNN(ir_graph, dt)
        return GPUCompiledModel(snn, ir_graph)
