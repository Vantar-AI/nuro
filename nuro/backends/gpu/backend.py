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

    def __init__(
        self, ir_graph: IRGraph, dt: float, surrogate_function=None
    ) -> None:
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
            self.neurons[nid] = build_neuron_layer(
                node, dt, surrogate_function=surrogate_function
            )

        # Build synapse layers
        self.synapses = nn.ModuleDict()
        self._edge_map: dict[str, SynapticEdge] = {}
        self._delay_steps: dict[str, int] = {}
        for edge in ir_graph.edges:
            key = f"{edge.source_id}__{edge.target_id}"
            source_node = ir_graph.nodes[edge.source_id]
            target_node = ir_graph.nodes[edge.target_id]
            self.synapses[key] = build_synapse_layer(edge, source_node, target_node)
            self._edge_map[key] = edge
            delay = getattr(edge, "delay", 0.0)
            self._delay_steps[key] = max(0, int(delay / dt)) if delay > 0 else 0

        # Delay ring buffers: key → deque of spike tensors
        self._delay_buffers: dict[str, list[torch.Tensor | None]] = {}
        self._has_delays = any(d > 0 for d in self._delay_steps.values())

    def init_state(self, batch_size: int) -> None:
        """Initialize batched state buffers for custom neurons (Izh/AdEx).

        Called once before the simulation loop when ``batch_size > 1``.
        SpikingJelly neurons handle batching automatically via their
        ``forward()`` method, so only custom neuron modules need this.
        """
        from nuro.backends.gpu.neurons import IzhikevichNode, AdExNode

        for neuron_layer in self.neurons.values():
            if isinstance(neuron_layer, (IzhikevichNode, AdExNode)):
                neuron_layer.init_state(batch_size)

    def init_delay_buffers(self, batch_size: int = 1) -> None:
        """Initialize delay ring buffers. Call before simulation loop."""
        self._delay_buffers = {}
        for key, delay in self._delay_steps.items():
            if delay > 0:
                self._delay_buffers[key] = [None] * delay

    def _get_delayed_spikes(
        self, key: str, current_spikes: torch.Tensor
    ) -> torch.Tensor | None:
        """Push current spikes into delay buffer and return delayed output."""
        if key not in self._delay_buffers:
            return current_spikes
        buf = self._delay_buffers[key]
        # FIFO: pop oldest, push newest
        delayed = buf.pop(0)
        buf.append(current_spikes)
        return delayed

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        source_ids: set[str],
        prev_spikes: dict[str, torch.Tensor] | None = None,
        batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Forward one timestep.

        Parameters
        ----------
        inputs : dict mapping population id -> input tensor
            External spike trains for source populations.
        source_ids : set of str
            IDs of source populations (no incoming edges).
        prev_spikes : dict, optional
            Previous timestep's spikes for cyclic graphs.
        batch_size : int
            Number of parallel trials (1 = no batch dim).
        """
        spikes: dict[str, torch.Tensor] = {}

        lookup = prev_spikes if (self._is_cyclic and prev_spikes) else spikes

        for nid in self.pop_order:
            if nid in source_ids and nid in inputs:
                spikes[nid] = inputs[nid]
                continue

            device = self._get_device()
            pop_size = self.ir_graph.nodes[nid].size
            if batch_size > 1:
                x = torch.zeros(batch_size, pop_size, device=device)
            else:
                x = torch.zeros(pop_size, device=device)

            for src_id in self.pop_order:
                key = f"{src_id}__{nid}"
                if key in self.synapses and src_id in lookup:
                    src_spikes = lookup[src_id]
                    # Apply synaptic delay if configured
                    if self._has_delays and key in self._delay_buffers:
                        delayed = self._get_delayed_spikes(key, src_spikes)
                        if delayed is not None:
                            x = x + self.synapses[key](delayed)
                    else:
                        x = x + self.synapses[key](src_spikes)

            spikes[nid] = self.neurons[nid](x)

        return spikes

    def _get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")


class GPUCompiledModel(CompiledModel):
    """A compiled model that runs on GPU (or CPU fallback) via SpikingJelly."""

    def __init__(
        self, snn: NuroSNN, ir_graph: IRGraph, requires_grad: bool = False
    ) -> None:
        self._snn = snn
        self._ir_graph = ir_graph
        self._metrics: dict[str, Any] = {}
        self._requires_grad = requires_grad
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

    @property
    def snn(self) -> NuroSNN:
        """Access the underlying SNN module (useful for optimizer param access)."""
        return self._snn

    def run(self, duration: float, dt: float = 1e-3, batch_size: int = 1, callbacks=None):
        """Run the model for *duration* seconds with timestep *dt*.

        Parameters
        ----------
        duration : float
            Simulation duration in seconds.
        dt : float
            Timestep in seconds.
        batch_size : int
            Number of parallel trials. When 1 (default), tensors have no
            batch dimension for full backward compatibility. When > 1,
            all internal tensors gain a leading ``(batch,)`` dimension.
        callbacks : list of Callback, optional
            Training callbacks for logging/MLOps integration.

        Returns
        -------
        None or dict[str, Tensor]
            When ``requires_grad=True``, returns a dict mapping population
            id to the **sum of output spikes** over time (shape matches
            the population tensor shape).  Use this for loss computation.
        """
        callbacks = callbacks or []
        num_steps = int(duration / dt)
        functional.reset_net(self._snn)
        for updater_tuple in self._stdp_updaters:
            updater_tuple[2].reset()

        # Initialize batched state for custom neurons
        if batch_size > 1:
            self._snn.init_state(batch_size)

        # Initialize delay buffers
        if self._snn._has_delays:
            self._snn.init_delay_buffers(batch_size)

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

        # Notify callbacks
        for cb in callbacks:
            cb.on_run_start({"duration": duration, "dt": dt, "batch_size": batch_size, "num_steps": num_steps})

        spike_counts: dict[str, int] = {nid: 0 for nid in self._ir_graph.nodes}
        total_spikes = 0
        prev_spikes: dict[str, torch.Tensor] | None = None

        # Accumulate output spikes for loss computation when training
        output_accum: dict[str, torch.Tensor] | None = None
        if self._requires_grad:
            output_accum = {}

        def _step_loop():
            nonlocal prev_spikes, total_spikes, output_accum

            for _step in range(num_steps):
                inputs: dict[str, torch.Tensor] = {}
                for sid in driven_ids:
                    spec = input_specs.get(sid)
                    if spec is not None and spec.data is not None:
                        # Static tensor input: (steps, pop) or (steps, batch, pop)
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
                        if batch_size > 1:
                            spk = (torch.rand(batch_size, size, device=self._device) < prob).float()
                        else:
                            spk = (torch.rand(size, device=self._device) < prob).float()
                        inputs[sid] = spk

                spikes = self._snn(inputs, driven_ids, prev_spikes, batch_size)
                prev_spikes = spikes

                # STDP update — skip when training (competing gradient updates)
                if not self._requires_grad:
                    for src_id, tgt_id, updater in self._stdp_updaters:
                        updater.step(spikes[src_id], spikes[tgt_id])

                # State recording
                if self._recorder.has_probes:
                    self._recorder.record_step(_step, spikes, self._snn)

                # Accumulate output spikes for training
                if output_accum is not None:
                    for nid, spk in spikes.items():
                        if nid in output_accum:
                            output_accum[nid] = output_accum[nid] + spk
                        else:
                            output_accum[nid] = spk.clone()

                # Accumulate metrics (detached to avoid graph bloat)
                for nid, spk in spikes.items():
                    count = int(spk.detach().sum().item())
                    spike_counts[nid] += count
                    total_spikes += count

                # Notify callbacks
                for cb in callbacks:
                    cb.on_step(_step, spikes, {"total_spikes": total_spikes})

        if self._requires_grad:
            _step_loop()
        else:
            with torch.no_grad():
                _step_loop()

        self._metrics = {
            "total_spikes": total_spikes,
            "spike_counts": spike_counts,
            "num_steps": num_steps,
            "duration": duration,
            "dt": dt,
            "batch_size": batch_size,
        }

        for cb in callbacks:
            cb.on_run_end(self._metrics)

        if self._requires_grad:
            return output_accum
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
            Shape ``(num_recordings, ...)`` — when batched, spikes/voltages
            are ``(num_recordings, batch, pop_size)``.
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

    def compile(
        self, ir_graph: IRGraph, dt: float = 1e-3, **kwargs
    ) -> GPUCompiledModel:
        requires_grad = kwargs.get("requires_grad", False)
        surrogate_name = kwargs.get("surrogate", "atan")

        surrogate_function = None
        if requires_grad:
            from nuro.backends.gpu.surrogates import get_surrogate
            surrogate_function = get_surrogate(surrogate_name)

        snn = NuroSNN(ir_graph, dt, surrogate_function=surrogate_function)
        return GPUCompiledModel(snn, ir_graph, requires_grad=requires_grad)
