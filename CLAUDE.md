# Nuro

**The universal SDK for spiking neural networks.**
Train on GPU. Deploy to neuromorphic silicon. One API, any backend.

**Version:** 0.5.0
**Repo:** https://github.com/Vantar-AI/nuro
**Website:** https://vantar.xyz
**Org:** Vantar AI

## The Big Picture

GPU is the **training workbench**. Neuromorphic chips are the **deployment target**.

```
Define (Python API) → Train (GPU + surrogate gradients) → Deploy (Loihi / SpiNNaker / analog)
                                                             ↑
                                                     This is why Nuro exists.
```

Nuro is the abstraction layer between spiking neural networks and hardware. Researchers define once, train on GPU with PyTorch optimizers, then recompile to neuromorphic silicon with zero code changes. The IR (intermediate representation) is the clean boundary — backends never touch API objects.

**Vantar AI = Nuro SDK (open source) + Vantar Cloud (commercial, coming 2026)**

## Dev Commands

```bash
# Activate venv (always use this — no global installs)
source .venv/bin/activate

# Install (editable, with GPU + dev deps)
pip install -e ".[gpu,dev]"

# Run all tests
pytest tests/ -v

# Lint + format
ruff check . && ruff format .

# Quick version check
python -c "import nuro; print(nuro.__version__)"

# Run examples
python examples/training/train_xor.py
python examples/basics/batched_simulation.py

# Benchmarks
python benchmarks/bench_nuro.py
```

## Architecture

```
nuro/
  api/              # User-facing API — what researchers import
    population.py   # Population (neuron groups)
    connection.py   # Connection (synaptic edges)
    graph.py        # Graph (holds populations + connections)
    input.py        # Input (static, generator, Poisson)
    compile.py      # compile() entry point → dispatches to backends
    objective.py    # Objective/loss functions (planned)
  ir/               # Intermediate Representation — the backend boundary
    __init__.py     # IRGraph.from_api_graph()
    nodes.py        # DynamicsNode dataclass
    edges.py        # SynapticEdge dataclass
    annotations.py  # IR metadata
  backends/         # Compilation targets
    base.py         # Abstract Backend + CompiledModel interfaces
    __init__.py     # Backend registry (lazy imports)
    gpu/            # GPU backend — training workbench (SpikingJelly)
      backend.py    # GPUBackend, NuroSNN (nn.Module), GPUCompiledModel
      dynamics.py   # build_neuron_layer() → neuron modules
      neurons.py    # IzhikevichNode, AdExNode (custom nn.Modules)
      surrogates.py # SurrogateSpike autograd function (atan, sigmoid, triangular)
      connectivity.py # build_synapse_layer() → nn.Linear
      plasticity.py # STDPUpdater (trace-based)
      recorders.py  # Recorder (voltages, spikes, weights)
      checkpoint.py # Save/load model weights + graph
    loihi/          # Loihi backend — neuromorphic deployment (v0.5.0)
      backend.py    # LoihiBackend, LoihiCompiledModel
      dynamics.py   # build_lava_neuron() → Lava LIF/Dense Processes
      _custom_neurons.py # IzhikevichProcess, AdExProcess (simulation-only)
      inputs.py     # build_input_process() → Lava RingBuffer
      monitor.py    # LoihiRecorder (Monitor-based probes)
      transfer.py   # load_gpu_weights(), apply_weights_to_lava()
  compiler/         # Compiler passes (stubs — future use)
  runtime/          # Runtime execution (stubs — future backends)
tests/              # 121 tests — pytest
examples/
  basics/           # Simulation examples
  training/         # Gradient training examples (v0.4.0+)
  deployment/       # Hardware deployment examples (v0.5.0+)
benchmarks/         # Performance benchmarks
```

## Data Flow

```
User code (nuro.Population, nuro.Connection, nuro.Graph)
    ↓ nuro.compile(graph, target="gpu"|"loihi", ...)
IR lowering (IRGraph.from_api_graph)
    ↓ backend.compile(ir_graph)

GPU Backend (training):                    Loihi Backend (deployment):
  NuroSNN (nn.Module)                        Lava Process graph
  ├── SpikingJelly neurons                   ├── lava.proc.lif.LIF
  ├── nn.Linear synapses                     ├── lava.proc.dense.Dense
  ├── Surrogate gradients                    ├── Port connections
  └── PyTorch optimizers                     └── RunConfig (sim or hardware)
    ↓                                          ↓
  model.run() → train with BPTT             model.run() → inference on silicon
  model.save("weights.pt")                  1000x more energy efficient
    ↓
  nuro.compile(graph, target="loihi", weights_from="weights.pt")
```

## Key Patterns

**Adding a neuron model:**
1. Implement `nn.Module` in `backends/gpu/neurons.py` (needs `.v`, `.reset()`, `.init_state()`)
2. Add dynamics name to `SUPPORTED_DYNAMICS` in `api/population.py`
3. Wire into `build_neuron_layer()` in `backends/gpu/dynamics.py`
4. Accept optional `surrogate_function` param for gradient training
5. Add Lava mapping in `backends/loihi/dynamics.py` (when applicable)
6. Add tests in `tests/test_neuron_models.py`

**Adding a backend:**
1. Create `backends/<name>/backend.py` implementing `Backend` + `CompiledModel`
2. Register in `backends/__init__.py` `_REGISTRY`
3. Add tests

**Gradient training (v0.4.0):**
- `nuro.compile(graph, requires_grad=True)` enables surrogate gradients
- `model.run()` returns `dict[str, Tensor]` (spike accumulations per population)
- Access weights via `model.snn.parameters()` for optimizers
- STDP auto-disabled during training
- Custom neurons use `SurrogateSpike.apply(v - thresh, surrogate_fn)`

**Train → Deploy workflow (v0.5.0):**
```python
# Train on GPU
gpu_model = nuro.compile(graph, target="gpu", requires_grad=True)
# ... training loop ...
gpu_model.save("trained.pt")

# Deploy to Loihi (one line change)
loihi_model = nuro.compile(graph, target="loihi", weights_from="trained.pt")
loihi_model.run(duration=1.0)
```

## Neuron Models

| Model | GPU (training) | Loihi (deployment) |
|-------|---------------|-------------------|
| LIF | SpikingJelly + surrogates | Native Lava LIF |
| IF | SpikingJelly + surrogates | Lava LIF (no leak) |
| Izhikevich | Custom + surrogates | Simulation only (v0.5), NcProcess (v0.6) |
| AdEx | Custom + surrogates | Simulation only (v0.5), NcProcess (v0.6) |

## Version History

| Version | What |
|---------|------|
| 0.1.0 | Core API, IR, GPU backend, LIF/IF, STDP, 37 tests |
| 0.2.0 | User inputs, state recording, Izh/AdEx, recurrent graphs, checkpointing, 77 tests |
| 0.3.0 | Batch support, performance benchmarks, 93 tests |
| 0.4.0 | Surrogate gradients, BPTT training, differentiable neurons, 109 tests |
| **0.5.0** | **Intel Loihi 2 backend (Lava), weight transfer GPU→Loihi, train→deploy, 121 tests** |

## Roadmap
- **v0.6.0** — SpiNNaker 2 backend, custom neuron dynamics on Loihi NeuroCores
- **v0.7.0** — Vantar Cloud MVP (remote compile + deploy)
- **v1.0.0** — Stable API, documentation site, model zoo

**Known gaps (v0.5.0):**
- `lava-nc` requires Python ≤3.10 — Loihi tests skip on 3.12. Need Python 3.10 venv or CI job to validate.
- `_custom_neurons.py` dict-based lazy imports need validation with real Lava decorator system.
- Recurrent/cyclic graphs not yet supported on Loihi backend.
- Fixed-point weight quantization stubbed but not implemented (scale_factor param).

## Sibling Projects

All under `/Users/malte/Development/vantar_language/`:
- `nuro/` — This project (SDK, open source)
- `nuro-examples/` — Extended examples (vision, audio, robotics, probabilistic, hybrid)
- `vantar-cloud/` — Cloud infrastructure (API, brokers, compiler, infra) — the commercial product
- `research/` — Research notes and papers

Website: `/Users/malte/Development/vantar-web/` — vantar.xyz (Next.js, push to deploy)

## Stack

- **Python 3.10+** with type annotations
- **PyTorch 2.0+** — GPU training backend
- **SpikingJelly 0.0.0.0.14+** — LIF/IF neuron kernels (GPU)
- **Lava-nc 0.9+** — Loihi backend (optional `[loihi]` extra)
- **NetworkX** — graph analysis (cycle detection, topological sort)
- **pytest** — testing
- **ruff** — lint + format

## Conventions

- Tests mirror source structure in `tests/test_*.py`
- One concept per file
- IR is the hard boundary — backends only see IR objects, never API objects
- `batch_size=1` never adds batch dimension (backward compat)
- GPU backend: custom neurons are `nn.Module` compatible with `functional.reset_net()`
- Loihi backend: use Lava's Process/ProcessModel pattern
- All changes need tests before merge
- Optional deps (`[gpu]`, `[loihi]`) — never require hardware-specific packages by default
