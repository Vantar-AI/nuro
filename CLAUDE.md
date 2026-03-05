# Nuro

**The universal SDK for spiking neural networks.**
Train on GPU. Deploy to neuromorphic silicon. One API, any backend.

**Version:** 0.7.0
**Repo:** https://github.com/Vantar-AI/nuro
**Website:** https://vantar.xyz
**Org:** Vantar AI

## The Big Picture

GPU is the **training workbench**. Neuromorphic chips are the **deployment target**.

```
Define (Python API) → Train (GPU + surrogate gradients) → Deploy (Loihi / SpiNNaker / Akida)
                                                             ↑
                                                     This is why Nuro exists.
```

Nuro is the abstraction layer between spiking neural networks and hardware. Researchers define once, train on GPU with PyTorch optimizers, then recompile to neuromorphic silicon with zero code changes. The IR (intermediate representation) is the clean boundary — backends never touch API objects.

**Vantar AI = Nuro SDK (open source) + Vantar Cloud (commercial, coming 2026)**

## Neuromorphic Skills

7 research-grounded skills for SNN development. Auto-detected by Claude Code.

| Skill | Use When |
|-------|----------|
| `/snn-architect` | Designing network architecture, choosing neuron models, encoding schemes, topology |
| `/snn-train` | Setting up training loops, surrogate gradients, debugging training, QAT |
| `/ann2snn` | Converting trained PyTorch/TF models to SNNs for hardware deployment |
| `/neuromorphic-deploy` | Compiling to Loihi 2, SpiNNaker 2, Akida, Xylo - hardware constraints and optimization |
| `/snn-benchmark` | Evaluating against SOTA results, running standardized benchmarks |
| `/snn-debug` | Dead neurons, spike storms, training instability, hardware accuracy mismatches |
| `/paper-implement` | Translating research papers into Nuro code, adding new neuron models |

All skills contain SOTA references (2024-2026), hardware specs, and code templates mapped to Nuro primitives.

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
    connection.py   # Connection (synaptic edges + delays + plasticity)
    graph.py        # Graph (holds populations + connections)
    input.py        # Input (static, generator, Poisson)
    compile.py      # compile() entry point → dispatches to backends
    objective.py    # Objective/loss functions (planned)
  ir/               # Intermediate Representation — the backend boundary
    __init__.py     # IRGraph.from_api_graph()
    nodes.py        # DynamicsNode dataclass
    edges.py        # SynapticEdge dataclass (with delay field)
    annotations.py  # IR metadata
    nir_compat.py   # NIR ↔ Nuro IR conversion (v0.7)
  conversion/       # ANN-to-SNN conversion (v0.7)
    __init__.py
    ann2snn.py      # convert_ann(), normalize_weights()
  backends/         # Compilation targets
    base.py         # Abstract Backend + CompiledModel interfaces
    __init__.py     # Backend registry (lazy imports)
    gpu/            # GPU backend — training workbench (SpikingJelly)
      backend.py    # GPUBackend, NuroSNN (nn.Module), GPUCompiledModel
      dynamics.py   # build_neuron_layer() → neuron modules
      neurons.py    # IzhikevichNode, AdExNode (custom nn.Modules)
      surrogates.py # SurrogateSpike autograd function (atan, sigmoid, triangular)
      connectivity.py # build_synapse_layer() → dense, one_to_one, conv1d, distance_dependent
      plasticity.py # STDPUpdater (trace-based)
      recorders.py  # Recorder (voltages, spikes, weights)
      checkpoint.py # Save/load model weights + graph
      quantization.py # QAT, post-training quantization (v0.7)
    loihi/          # Loihi 2 backend (Lava)
      backend.py    # LoihiBackend, LoihiCompiledModel (+ on-chip STDP)
      dynamics.py   # build_lava_neuron() → Lava LIF/Dense Processes
      _custom_neurons.py # IzhikevichProcess, AdExProcess (simulation-only)
      inputs.py     # build_input_process() → Lava RingBuffer
      monitor.py    # LoihiRecorder (Monitor-based probes)
      transfer.py   # load_gpu_weights(), apply_weights_to_lava(), quantize_weights()
    spinnaker2/     # SpiNNaker 2 backend (v0.6)
      backend.py    # SpiNNaker2Backend, SpiNNaker2CompiledModel
      connections.py # connection_list format with delays
      transfer.py   # Weight transfer + quantization
    akida/          # BrainChip Akida backend (v0.7)
      backend.py    # AkidaBackend, AkidaCompiledModel
      dynamics.py   # Nuro → Akida layer mapping
      transfer.py   # Weight conversion to Akida format
  datasets/         # Neuromorphic dataset loaders (v0.7)
    vision.py       # NMNIST, DVSCifar10, DVSGesture
    utils.py        # Common loading utilities
  logging.py        # Python logging configuration (v0.7)
  callbacks.py      # WandbCallback, TensorBoardCallback (v0.7)
  compiler/         # Compiler passes (stubs — future use)
  runtime/          # Runtime execution (stubs — future backends)
tests/              # 168 tests — pytest
examples/
  basics/           # Simulation examples
  training/         # Gradient training examples
  deployment/       # Hardware deployment examples
  conversion/       # ANN-to-SNN examples
notebooks/          # Colab tutorials (MNIST, conversion, hardware)
benchmarks/         # Performance benchmarks
```

## Data Flow

```
User code (nuro.Population, nuro.Connection, nuro.Graph)
    ↓ nuro.compile(graph, target="gpu"|"loihi"|"spinnaker2"|"akida", ...)
IR lowering (IRGraph.from_api_graph)
    ↓ backend.compile(ir_graph)

GPU Backend (training):              Hardware Backends (deployment):
  NuroSNN (nn.Module)                  Loihi 2 (Lava) | SpiNNaker 2 | Akida
  ├── SpikingJelly neurons             ├── Auto-quantization (8/16/4-bit)
  ├── nn.Linear synapses               ├── Weight transfer from GPU
  ├── Surrogate gradients              ├── Delay mapping
  ├── Delay buffers                    └── On-chip learning (Loihi/SpiNNaker)
  └── PyTorch optimizers                 ↓
    ↓                                  model.run() → inference on silicon
  model.run() → train with BPTT       1000x more energy efficient
  model.save("weights.pt")
    ↓
  nuro.compile(graph, target="loihi", weights_from="weights.pt")  # auto-quantizes
```

**ANN-to-SNN path:**
```
Trained PyTorch model → nuro.convert_ann(model, input_shape) → SNN Graph → compile to hardware
```

**NIR interop:**
```
External framework (snnTorch/Norse/SpikingJelly) → NIR → nuro.from_nir() → Nuro Graph → compile
```

## Key Patterns

**Adding a neuron model:** (use `/paper-implement` skill)
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

**Train → Deploy workflow:**
```python
# Train on GPU (use /snn-train skill for recipes)
gpu_model = nuro.compile(graph, target="gpu", requires_grad=True, surrogate="atan")
# ... training loop ...
gpu_model.save("trained.pt")

# Deploy to hardware (use /neuromorphic-deploy skill for chip selection)
loihi_model = nuro.compile(graph, target="loihi", weights_from="trained.pt")  # auto-quantizes
akida_model = nuro.compile(graph, target="akida", weights_from="trained.pt")

# Convert existing PyTorch model (use /ann2snn skill)
snn_graph = nuro.convert_ann(pytorch_model, input_shape=(784,), num_steps=100)
```

## Neuron Models

| Model | GPU (training) | Loihi 2 | SpiNNaker 2 | Akida |
|-------|---------------|---------|-------------|-------|
| LIF | SpikingJelly + surrogates | Native Lava LIF | Brian2 LIF | Akida layers |
| IF | SpikingJelly + surrogates | Lava LIF (no leak) | Brian2 IF | Akida layers |
| Izhikevich | Custom + surrogates | NcProcess | Brian2 Izh | - |
| AdEx | Custom + surrogates | NcProcess | Brian2 AdEx | - |

## Connectivity Patterns

| Pattern | Description | v0.7 |
|---------|-------------|------|
| `dense` | Fully connected | All backends |
| `random_sparse` | Random sparse connectivity | GPU |
| `one_to_one` | Diagonal (identity) | All backends |
| `conv1d` | 1D convolutional | GPU |
| `distance_dependent` | Gaussian probability by distance | GPU |

## Version History

| Version | What |
|---------|------|
| 0.1.0 | Core API, IR, GPU backend, LIF/IF, STDP, 37 tests |
| 0.2.0 | User inputs, state recording, Izh/AdEx, recurrent graphs, checkpointing, 77 tests |
| 0.3.0 | Batch support, performance benchmarks, 93 tests |
| 0.4.0 | Surrogate gradients, BPTT training, differentiable neurons, 109 tests |
| 0.5.0 | Intel Loihi 2 backend (Lava), weight transfer GPU→Loihi, train→deploy, 121 tests |
| 0.6.0 | SpiNNaker 2 backend, custom neuron dynamics on Loihi NeuroCores |
| **0.7.0** | **NIR interop, ANN-to-SNN, Akida backend, auto-quantization, synaptic delays, connectivity patterns, datasets, logging/callbacks, on-chip learning, 168 tests** |

## Roadmap
- **v0.8.0** — Vantar Cloud MVP (remote compile + deploy)
- **v0.9.0** — Nuro Copilot (AI-assisted SNN design)
- **v1.0.0** — Stable API, documentation site, model zoo

**Known gaps (v0.7.0):**
- `lava-nc` requires Python ≤3.10 — Loihi tests skip on 3.12.
- Conv2d spatial mapping not yet implemented (flattened to dense in ANN-to-SNN).
- Transformer conversion (MBE neurons) is research-only, not in SDK yet.

## Sibling Projects

- `~/Development/nuro/` — This project (SDK, open source)
- `~/Development/nuro-copilot/` — AI-assisted SNN design (planned)
- `~/Development/vantar_language/` — Vantar DSL (research)
- `~/Development/vantar-web/` — vantar.xyz website (Next.js)

## Stack

- **Python 3.10+** with type annotations
- **PyTorch 2.0+** — GPU training backend
- **SpikingJelly 0.0.0.0.14+** — LIF/IF neuron kernels (GPU)
- **Lava-nc 0.9+** — Loihi backend (optional `[loihi]` extra)
- **py-spinnaker2** — SpiNNaker 2 backend (optional `[spinnaker2]` extra)
- **akida** — Akida backend (optional `[akida]` extra)
- **nir** — NIR interop (optional `[nir]` extra)
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
- Optional deps (`[gpu]`, `[loihi]`, `[spinnaker2]`, `[akida]`, `[nir]`) — never require hardware-specific packages by default
- Auto-quantization on `compile()` when `weights_from` is set and target is hardware
