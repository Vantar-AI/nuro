# Nuro: A Universal SDK for Spiking Neural Networks with Multi-Backend Compilation

**Malte Wagenbach**
VantarGroup LLC
malte@vantar.xyz

*Preprint — February 2026*

---

## Abstract

Spiking neural networks (SNNs) hold significant promise for energy-efficient inference on neuromorphic hardware, yet the software ecosystem remains fragmented: training frameworks (SpikingJelly, Norse) and deployment SDKs (Intel Lava, py-spinnaker2) operate with incompatible APIs, no weight transfer story, and separate installation requirements. Researchers and engineers must maintain parallel codebases for a single network. We present **Nuro**, an open-source Python SDK that unifies the SNN development lifecycle through a backend-agnostic intermediate representation (IR). Users define a spiking network once using a clean declarative API, train it with surrogate gradients on GPU using standard PyTorch optimizers, and recompile the same network to Intel Loihi 2 or SpiNNaker 2 with a single argument change and zero code modifications. We describe Nuro's architecture (API → IR → backends), its surrogate gradient training system, and its hardware weight transfer pipeline. On an RTX 4090, batched simulation achieves up to 124× throughput scaling over single-trial execution at 1,000 neurons. The train-to-Loihi deployment workflow reduces a previously multi-framework, multi-step process to three lines of code. Nuro is available at [https://github.com/Vantar-AI/nuro](https://github.com/Vantar-AI/nuro) under the Apache 2.0 license.

---

## 1. Introduction

Spiking neural networks process information through discrete spike events rather than continuous activations, enabling event-driven computation with orders-of-magnitude lower energy consumption on neuromorphic hardware [CITE: Mahowald1991, Davies2021]. As neuromorphic chips become increasingly available — Intel Loihi 2 [CITE: Orchard2021], SpiNNaker 2 [CITE: Furber2014], BrainScaleS-2 [CITE: Schemmel2020], Akida, Xylo — the bottleneck has shifted from hardware to software.

The current SNN software ecosystem is fragmented along two incompatible axes:

**Training frameworks** prioritize GPU throughput and gradient computation. SpikingJelly [CITE: Fang2023] wraps SpikingJelly's neuron kernels around PyTorch's autograd, enabling backpropagation-through-time (BPTT) via surrogate gradients [CITE: Neftci2019]. Norse [CITE: Pehle2021] provides functional-style PyTorch primitives for SNNs. Both are excellent for training but have no path to hardware deployment.

**Deployment SDKs** target specific hardware. Intel Lava [CITE: Lava2022] provides a process-based programming model for Loihi 2 but has no gradient training and a steep learning curve. py-spinnaker2 targets the SpiNNaker 2 architecture with a Brian2-based simulation fallback. PyNN [CITE: Davison2008] offers a simulator-neutral interface but reflects 2008-era design decisions and limited modern hardware support.

The result: a researcher training an SNN in SpikingJelly must separately learn Lava to run it on Loihi, with no automated mechanism for transferring trained weights between the two. This creates duplicated effort, untested deployment paths, and a high barrier to hardware experimentation.

We present **Nuro**, a Python SDK that addresses this fragmentation through a single unifying abstraction: an intermediate representation (IR) that cleanly separates network definition from compilation target. The contributions of this work are:

1. **A declarative SNN API** (Population, Connection, Input, Graph) that is neuron-model-agnostic and backend-agnostic.
2. **An IR-based compilation system** that lowers user-defined graphs to multiple hardware backends without modifying user code.
3. **A surrogate gradient training system** for GPU-based BPTT with three built-in surrogate functions.
4. **Automated weight transfer** from GPU checkpoints to neuromorphic backends (Loihi 2, SpiNNaker 2).
5. **Batched simulation** achieving 32–124× throughput scaling for multi-trial experiments.
6. **A cloud compilation backend** (Vantar Cloud) for hardware access without chip ownership.

---

## 2. Background

### 2.1 Spiking Neural Networks

Unlike artificial neurons that output real-valued activations, spiking neurons integrate inputs over time and emit binary spike events when a threshold is crossed. The leaky integrate-and-fire (LIF) neuron is the most common model:

```
τ dv/dt = -(v - v_rest) + R·I(t)
spike if v ≥ v_thresh; reset v ← v_reset
```

More complex models include Izhikevich neurons [CITE: Izhikevich2003] (capable of replicating 20+ cortical firing patterns) and AdEx neurons [CITE: Brette2005] (adaptive threshold). SNNs encode information in spike timing and rate, enabling temporal codes that are inaccessible to rate-based ANNs.

### 2.2 Surrogate Gradients

The Heaviside firing function has zero gradient almost everywhere, making standard backpropagation impossible. Surrogate gradient methods [CITE: Neftci2019] replace the Heaviside derivative during the backward pass with a smooth approximation:

```
Forward:   s = H(v - v_thresh)   [binary spike]
Backward:  ∂s/∂v ≈ σ'(v - v_thresh)  [smooth surrogate]
```

Common surrogate functions include the arc-tangent (ATan), logistic sigmoid, and triangular window. This enables BPTT through spiking layers using standard autodiff frameworks.

### 2.3 Related Work

**SpikingJelly** [CITE: Fang2023] is the most widely adopted SNN training framework. It provides efficient GPU implementations of LIF neurons with surrogate gradients, built on PyTorch. SpikingJelly has no hardware deployment path and is GPU-only.

**Intel Lava** [CITE: Lava2022] is the official SDK for Intel Loihi 2. It uses a process-based programming model (Process/ProcessModel pairs) distinct from standard PyTorch-style APIs. Lava supports deployment but not gradient training. Its API requires significant restructuring of code designed for SpikingJelly.

**Norse** [CITE: Pehle2021] provides functional PyTorch primitives for spiking networks, with elegant API design. Like SpikingJelly, it has no hardware deployment path.

**PyNN** [CITE: Davison2008] is a simulator-neutral interface predating modern deep learning frameworks. Its API design reflects 2008-era practices; it lacks surrogate gradient training and has limited support for current hardware.

**Brian2** [CITE: Goodman2009] is a flexible Python simulator for SNNs, popular in computational neuroscience for its expressive equation-based neuron definition language. It targets simulation rather than hardware deployment.

Nuro differs from all of the above by spanning both training (GPU) and deployment (Loihi 2, SpiNNaker 2) within a single API, with automated weight transfer between domains.

---

## 3. System Design

### 3.1 Overview

Nuro is organized into three layers:

```
┌──────────────────────────────────────────────────────────┐
│  API Layer                                               │
│  Population · Connection · Input · Graph · compile()     │
├──────────────────────────────────────────────────────────┤
│  Intermediate Representation (IR)                        │
│  DynamicsNode · SynapticEdge · IRGraph                   │
├──────────────────────────────────────────────────────────┤
│  Backends                                                │
│  GPU (SpikingJelly) · Loihi (Lava) · SpiNNaker2 · Cloud  │
└──────────────────────────────────────────────────────────┘
```

The IR is the hard boundary: backends receive only IR objects, never API objects. This separation ensures that adding a new backend requires no changes to user-facing code.

### 3.2 API Layer

The user-facing API consists of five classes:

**Population** — a group of neurons with shared dynamics:
```python
pop = nuro.Population(
    size=100,
    dynamics="lif",          # "lif" | "if" | "izhikevich" | "adex"
    params={"tau": 20e-3},
)
```

**Connection** — a synaptic projection between two populations:
```python
conn = nuro.Connection(
    source=inp_pop,
    target=out_pop,
    pattern="dense",          # "dense" | "random_sparse"
    plasticity="stdp",        # "stdp" | None
)
```

**Input** — external drive to a population:
```python
inp = nuro.Input(population=pop, mode="poisson", rate=100.0)
inp = nuro.Input(population=pop, data=spike_tensor)       # (T, N)
inp = nuro.Input(population=pop, generator=lambda t: ...) # per-step fn
```

**Graph** — the complete network definition:
```python
graph = nuro.Graph([inp_pop, out_pop], [conn], inputs=[inp])
print(graph.is_cyclic)  # NetworkX-based cycle detection
```

**compile()** — dispatches to a backend:
```python
model = nuro.compile(graph, target="gpu")       # training
model = nuro.compile(graph, target="loihi")     # deployment
model = nuro.compile(graph, target="spinnaker2")
model = nuro.compile(graph, target="cloud", hardware="loihi")
```

The `compile()` function lowers the Graph to an IRGraph and dispatches to the selected backend. Keyword arguments specific to a backend (e.g., `requires_grad`, `weights_from`, `hardware`, `api_key`) are passed through to `backend.compile()`.

### 3.3 Intermediate Representation

The IR consists of three dataclasses:

```python
@dataclass
class DynamicsNode:
    id: str
    size: int
    dynamics: str          # "lif" | "if" | "izhikevich" | "adex"
    params: dict

@dataclass
class SynapticEdge:
    id: str
    source_id: str
    target_id: str
    pattern: str
    plasticity: str | None

@dataclass
class IRGraph:
    nodes: list[DynamicsNode]
    edges: list[SynapticEdge]
    inputs: list[InputSpec]
```

`IRGraph.from_api_graph(graph)` performs the lowering step, validating population sizes, checking for unsupported dynamics combinations, and running topological analysis via NetworkX.

### 3.4 Backend Registry

Backends are registered by name with lazy imports to avoid requiring all hardware SDKs at import time:

```python
_REGISTRY = {
    "gpu":        "nuro.backends.gpu.backend:GPUBackend",
    "loihi":      "nuro.backends.loihi.backend:LoihiBackend",
    "spinnaker2": "nuro.backends.spinnaker2.backend:SpiNNaker2Backend",
    "cloud":      "nuro.backends.cloud.backend:CloudBackend",
}
```

A user without `lava-nc` installed can still import and use the GPU backend. Hardware SDKs are installed as optional extras (`pip install nuro[loihi]`).

Each backend implements a two-class interface:

```python
class Backend(ABC):
    @abstractmethod
    def compile(self, ir_graph: IRGraph, **kwargs) -> CompiledModel: ...

class CompiledModel(ABC):
    @abstractmethod
    def run(self, duration: float, dt: float, batch_size: int): ...
    @abstractmethod
    def reset(self) -> None: ...
    @property
    @abstractmethod
    def metrics(self) -> dict[str, Any]: ...
```

---

## 4. Backends

### 4.1 GPU Backend

The GPU backend compiles an IRGraph to a `NuroSNN`, a `torch.nn.Module` that wraps SpikingJelly neuron layers and `nn.Linear` synapse matrices.

**Neuron mapping:**

| Nuro dynamics | GPU implementation |
|--------------|-------------------|
| `"lif"` | `spikingjelly.activation_based.neuron.LIFNode` |
| `"if"` | `spikingjelly.activation_based.neuron.IFNode` |
| `"izhikevich"` | Custom `IzhikevichNode(nn.Module)` |
| `"adex"` | Custom `AdExNode(nn.Module)` |

Custom neurons implement `.v` (membrane potential), `.reset()`, and `.init_state(batch_size)` to remain compatible with SpikingJelly's `functional.reset_net()` utilities.

**Surrogate gradients** are applied via a custom autograd function:

```python
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_shifted, surrogate_fn):
        ctx.save_for_backward(v_shifted)
        ctx.surrogate_fn = surrogate_fn
        return (v_shifted >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v_shifted,) = ctx.saved_tensors
        grad = ctx.surrogate_fn(v_shifted) * grad_output
        return grad, None
```

Three surrogate functions are provided: ATan (`1 / (π(1 + (πv)²))`), sigmoid (`σ'(v)`), and triangular (`max(0, 1 - |v|)`).

**Batch support** is implemented via a leading batch dimension in all internal tensors. When `batch_size=1` (default), no batch dimension is added, preserving full backward compatibility.

**STDP** is implemented via trace-based pre/post-synaptic updates. It is automatically disabled when `requires_grad=True`, as STDP and BPTT are mutually exclusive.

### 4.2 Loihi 2 Backend

The Loihi backend compiles an IRGraph to a Lava process graph using `lava.proc.lif.LIF` and `lava.proc.dense.Dense` processes.

**Neuron mapping:**

| Nuro dynamics | Loihi implementation |
|--------------|---------------------|
| `"lif"` | Native `lava.proc.lif.LIF` |
| `"if"` | `LIF` with tau set to maximum |
| `"izhikevich"` | Custom `IzhikevichProcess` (sim) / NcProcess (v0.6+) |
| `"adex"` | Custom `AdExProcess` (sim) / NcProcess (v0.6+) |

**Weight transfer** from GPU checkpoints is implemented in `nuro.backends.loihi.transfer`:

```python
def load_gpu_weights(checkpoint_path: str, ir_graph: IRGraph) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weights = {}
    for edge in ir_graph.edges:
        key = f"synapses.{edge.id}.weight"
        if key in checkpoint["model_state_dict"]:
            weights[edge.id] = checkpoint["model_state_dict"][key].numpy()
    return weights
```

Weights are then applied to the corresponding `Dense` process weight matrices. Fixed-point quantization (`quantize=True`) is supported via a scale factor parameter.

**Run configuration** supports both simulation (`Loihi2SimCfg`, default) and real hardware (`Loihi2HwCfg`, requires INRC access):

```python
from lava.magma.core.run_configs import Loihi2HwCfg
model.set_run_config(Loihi2HwCfg())
```

### 4.3 SpiNNaker 2 Backend

The SpiNNaker 2 backend compiles to a `py-spinnaker2` `Network` with Brian2 as a simulation fallback.

**Population mapping** converts `DynamicsNode` to `spinnaker2.Population` with equivalent LIF parameters. Dense connections become `spinnaker2.Projection` with `AllToAllConnector`.

**Hardware switching:**
```python
model.set_hardware(True)  # switches from Brian2 to real SpiNNaker 2 chip
```

Real hardware access requires SpiNNcloud credentials (University of Manchester).

### 4.4 Cloud Backend

The cloud backend serializes the IRGraph to a JSON wire format and submits it to the Vantar Cloud API:

```
POST /v1/compile  →  {job_id}
GET  /v1/jobs/{id}  →  {status, progress}
POST /v1/execute/{id}  →  {run_id}
GET  /v1/results/{run_id}  →  {metrics, spike_data}
```

The API server compiles and executes via hardware brokers that install Nuro locally on the cloud server and call the existing Loihi or SpiNNaker 2 backends. No compilation logic is duplicated.

---

## 5. Training Workflow

The standard training workflow with surrogate gradients:

```python
import torch, nuro

# 1. Define network
inp  = nuro.Population(size=784, dynamics="lif", params={"tau": 10e-3})
hid  = nuro.Population(size=256, dynamics="lif", params={"tau": 20e-3})
out  = nuro.Population(size=10,  dynamics="lif", params={"tau": 5e-3})
c1   = nuro.Connection(source=inp, target=hid, pattern="dense")
c2   = nuro.Connection(source=hid, target=out, pattern="dense")
data = nuro.Input(population=inp, mode="poisson", rate=200.0)
graph = nuro.Graph([inp, hid, out], [c1, c2], inputs=[data])

# 2. Compile with surrogate gradients
model = nuro.compile(graph, target="gpu", requires_grad=True, surrogate="atan")
optimizer = torch.optim.Adam(model.snn.parameters(), lr=1e-3)

# 3. Training loop
for epoch in range(100):
    optimizer.zero_grad()
    model.reset()
    output = model.run(duration=0.05, dt=1e-3)   # returns dict[pop_id → Tensor]
    loss = F.cross_entropy(output[out.id].sum(0), labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.snn.parameters(), 1.0)
    optimizer.step()

# 4. Save
model.save("trained.pt")
```

When `requires_grad=True`:
- `model.run()` returns `dict[pop_id → Tensor(T, N)]` rather than `None`
- Surrogate spike function is applied instead of raw Heaviside
- `torch.no_grad()` context is removed from the run loop
- STDP is disabled

---

## 6. Train-to-Hardware Deployment

After training, recompiling to hardware requires changing one argument:

```python
# Deploy to Loihi 2
loihi_model = nuro.compile(graph, target="loihi", weights_from="trained.pt")
loihi_model.run(duration=1.0)
print(loihi_model.metrics["total_spikes"])

# Deploy to SpiNNaker 2
sp2_model = nuro.compile(graph, target="spinnaker2", weights_from="trained.pt")
sp2_model.run(duration=1.0)

# Deploy via Vantar Cloud (no chip ownership required)
cloud_model = nuro.compile(graph, target="cloud", hardware="loihi",
                           weights_from="trained.pt")
cloud_model.run(duration=1.0)
```

The network definition (`graph`) is identical across all targets. The IR enforces this — the same `IRGraph` object is dispatched to different backends.

**Lines of code comparison for a train-then-deploy workflow:**

| Approach | Training LOC | Deployment LOC | Weight transfer |
|----------|-------------|----------------|----------------|
| SpikingJelly + Lava (manual) | ~50 | ~80 | Manual (framework-specific) |
| Norse + Lava (manual) | ~40 | ~80 | Manual |
| **Nuro** | **~15** | **1 argument change** | **Automatic** |

---

## 7. Evaluation

### 7.1 GPU Throughput Benchmarks

We benchmarked the GPU backend on an NVIDIA RTX 4090 using LIF networks of varying sizes. All results are wall-clock time for a 1-second simulation at dt=1ms.

**Table 1: GPU simulation throughput (RTX 4090)**

| Neurons | Batch=1 | Batch=8 | Batch=32 | Batch=128 | Speedup (128×) |
|---------|---------|---------|---------|----------|--------------|
| 100 | 31ms / 0.6M spk/s | 31ms / 5.2M | 35ms / 18M | 35ms / 73M | **114×** |
| 1,000 | 350ms / 5.7M | 358ms / 44M | 347ms / 184M | 362ms / 707M | **124×** |
| 10,000 | 584ms / 34M | 646ms / 248M | 646ms / 991M | 833ms / 3.1B | **90×** |

Key observations:
- **Near-linear batch scaling** up to batch=32: the same wall time yields 32× more trials
- **Sublinear scaling at batch=128** for large networks (10K neurons) due to memory bandwidth limits
- **Overhead is constant**: the difference between batch=1 and batch=32 at 1,000 neurons is 3ms — amortized across 32 trials this is negligible

This scaling makes Nuro practical for hyperparameter sweeps, population variability studies, and few-shot learning evaluations where many trials are needed.

### 7.2 Memory Footprint

| Neurons | Batch=1 | Batch=32 | Batch=128 |
|---------|---------|---------|----------|
| 100 | 8.2 MB | 8.3 MB | 8.5 MB |
| 1,000 | 12.0 MB | 12.8 MB | 15.5 MB |
| 10,000 | 390 MB | 399 MB | 427 MB |

GPU memory scales primarily with network size, not batch size. Batch overhead is modest: batch=128 vs batch=1 at 10K neurons adds only 37MB.

### 7.3 Train-to-Deploy Workflow Validation

We validated the train→deploy workflow using a 3-layer network (50→30→10 LIF neurons) trained for 5 epochs on GPU, then deployed to Loihi 2 simulation. Weights were transferred automatically via `weights_from="trained.pt"`.

| Step | Tool | Time |
|------|------|------|
| Define network | Nuro API | < 1s |
| GPU training (5 epochs) | nuro + PyTorch | ~2s |
| Save checkpoint | torch.save | < 1s |
| Compile to Loihi | nuro + Lava | ~3s |
| Loihi inference (0.1s sim) | Lava | ~1s |
| **Total** | | **~7s** |

Without Nuro, this workflow requires separate SpikingJelly training and Lava deployment code, with manual weight extraction and reshaping.

### 7.4 Test Coverage

Nuro ships with 121 tests across the full stack:

| Module | Tests | Coverage |
|--------|-------|---------|
| API validation | 18 | Population, Connection, Graph, Input |
| IR lowering | 12 | from_api_graph, cycle detection |
| GPU backend | 31 | Training, surrogates, batching, checkpointing |
| Loihi backend | 22 | Compilation, weight transfer, probes |
| SpiNNaker 2 | 14 | Compilation, Brian2 sim |
| Neuron models | 9 | LIF, Izhikevich presets, AdEx |
| Recurrence | 8 | Cyclic graphs, Jacobi iteration |
| Gradients | 7 | BPTT, loss propagation |
| **Total** | **121** | |

---

## 8. Discussion

### 8.1 Design Decisions

**Why Python?** SNN researchers use Python. SpikingJelly, Lava, Brian2, PyNN are all Python. Nuro needs to compose with existing codebases, not replace them. Performance-critical code lives in SpikingJelly's CUDA kernels, not Nuro's Python orchestration layer.

**Why an IR?** The IR ensures clean separation of concerns. Backends never touch user API objects. This enables (a) adding backends without touching user code, (b) serializing networks for cloud compilation, (c) validating networks before compilation, and (d) eventually enabling cross-backend optimizations.

**Why lazy backend imports?** A user without `lava-nc` installed should be able to import `nuro` and use the GPU backend. Optional extras (`[loihi]`, `[spinnaker2]`) install hardware-specific dependencies without polluting the default environment.

**Why not build on Lava?** Lava targets Loihi specifically and has a process-based programming model that doesn't generalize to other backends. Nuro uses Lava as a compilation target — analogous to LLVM IR — not as a user-facing API.

### 8.2 Limitations

- **Loihi custom neurons on hardware**: Izhikevich and AdEx use simulation-only Processes in v0.6. NcProcess support for hardware is planned for v0.7.
- **Recurrent graphs on Loihi**: Cyclic graphs are supported on GPU and SpiNNaker 2 but not yet on Loihi (planned v0.7).
- **Weight quantization**: Fixed-point quantization is partially implemented; full Loihi 2 hardware quantization (8-bit) is in progress.
- **Python ≤3.10 for Loihi**: `lava-nc` requires Python ≤3.10, which limits use on Python 3.11+ environments. A separate virtual environment is recommended.

### 8.3 Future Work

**Vantar Cloud** (v0.7.0): A managed API for remote compilation and hardware access. Researchers without INRC or SpiNNcloud access can submit a Nuro graph and receive results from real neuromorphic hardware. This removes the hardware access barrier — currently the highest friction point for SNN research.

**Model zoo**: Pretrained SNN models for common tasks (image classification, event-based vision, audio processing) will be provided with weights transferable to hardware backends.

**Additional backends**: Analog neuromorphic chips (BrainScaleS-2, Akida, Xylo) and event-based sensor pipelines (DVS cameras).

**Compiler passes**: Automated partitioning for networks that exceed single-chip capacity; layer-to-core mapping optimization for Loihi 2.

---

## 9. Conclusion

We presented Nuro, an open-source SDK that unifies spiking neural network training and neuromorphic hardware deployment through a backend-agnostic intermediate representation. Nuro reduces the train-to-deploy workflow from a multi-framework, multi-step process requiring expertise in both SpikingJelly/Norse and Lava/py-spinnaker2 to a single API with one argument change between training and deployment.

Key results: up to 124× GPU throughput scaling through batched simulation; automated weight transfer from PyTorch checkpoints to Loihi 2 and SpiNNaker 2; and 121 tests covering the full compilation stack. The IR-based architecture enables adding new hardware backends without modifying user-facing API code.

Nuro is available at [https://github.com/Vantar-AI/nuro](https://github.com/Vantar-AI/nuro) under the Apache 2.0 license. We welcome contributions from the neuromorphic research community.

---

## References

[Brette2005] Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. *Journal of Neurophysiology*, 94(5), 3637–3642.

[Davies2021] Davies, M., et al. (2021). Advancing neuromorphic computing with Loihi: A survey of results and outlook. *Proceedings of the IEEE*, 109(5), 911–934.

[Davison2008] Davison, A. P., et al. (2008). PyNN: a common interface for neuronal network simulators. *Frontiers in Neuroinformatics*, 2, 11.

[Fang2023] Fang, W., et al. (2023). SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence. *Science Advances*, 9(16).

[Furber2014] Furber, S. B., et al. (2014). The SpiNNaker project. *Proceedings of the IEEE*, 102(5), 652–665.

[Goodman2009] Goodman, D. F. M., & Brette, R. (2009). The Brian simulator. *Frontiers in Neuroscience*, 3, 26.

[Izhikevich2003] Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569–1572.

[Lava2022] Intel Corporation. (2022). Lava: An open-source software framework for neuromorphic computing. [https://github.com/lava-nc/lava](https://github.com/lava-nc/lava)

[Mahowald1991] Mahowald, M., & Douglas, R. (1991). A silicon neuron. *Nature*, 354(6354), 515–518.

[Neftci2019] Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51–63.

[Orchard2021] Orchard, G., et al. (2021). Efficient neuromorphic signal processing with Loihi 2. *IEEE Workshop on Signal Processing Systems (SiPS)*.

[Pehle2021] Pehle, C., & Pedersen, J. E. (2021). Norse — A deep learning library for spiking neural networks. [https://github.com/norse/norse](https://github.com/norse/norse)

[Schemmel2020] Schemmel, J., et al. (2020). Accelerated analog neuromorphic computing. *Proceedings of the IEEE*, 108(8), 1227–1231.

---

*Nuro is developed by Malte Wagenbach at VantarGroup LLC. Correspondence: malte@vantar.xyz*

*Code: [https://github.com/Vantar-AI/nuro](https://github.com/Vantar-AI/nuro)*
*Docs: [https://github.com/Vantar-AI/nuro/tree/main/docs](https://github.com/Vantar-AI/nuro/tree/main/docs)*
