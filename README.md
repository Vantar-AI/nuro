<div align="center">

# Nuro

**The universal SDK for spiking neural networks.**

*Train on GPU. Deploy to neuromorphic silicon. One API, any backend.*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.4.0-green.svg)](https://github.com/Vantar-AI/nuro/releases/tag/v0.4.0)
[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen.svg)](#testing)
[![Website](https://img.shields.io/badge/website-vantar.xyz-white.svg)](https://vantar.xyz)

</div>

---

## What is Nuro?

Nuro is a Python SDK for building spiking neural networks that compiles to multiple backends. Define your network once — train with surrogate gradients on GPU, then deploy the same network to Intel Loihi, SpiNNaker, or analog neuromorphic chips. No code changes. The GPU is your training workbench. Neuromorphic silicon is the destination.

```python
import nuro

# Define populations
sensory = nuro.Population(size=100, dynamics="izhikevich", params={"preset": "regular_spiking"})
motor = nuro.Population(size=20, dynamics="adex")

# Connect with STDP learning
conn = nuro.Connection(source=sensory, target=motor, pattern="dense", plasticity="stdp")

# Bring your own data
inp = nuro.Input(population=sensory, data=my_spike_tensor)

# Compile and run
graph = nuro.Graph([sensory, motor], [conn], inputs=[inp])
model = nuro.compile(graph, target="gpu")

# Record state during simulation
model.record("voltages", population=motor)
model.record("spikes", population=motor)
model.run(duration=1.0)

# Inspect results
voltages = model.get_state("voltages", population=motor)  # (1000, 20) tensor
spikes = model.get_state("spikes", population=motor)      # (1000, 20) tensor

# Save for later
model.save("my_network.pt")
```

## Why Nuro?

The neuromorphic ecosystem is fragmented. SpikingJelly, Lava, BrainPy, Norse — each framework locks you into one paradigm. Nuro provides a single abstraction layer: define once, train on GPU, deploy to silicon.

- **Train on GPU, deploy to silicon** — surrogate gradients for training, then recompile to Loihi/SpiNNaker with zero code changes
- **Backend-agnostic** — write once, compile to GPU (SpikingJelly), Loihi (Lava), or future targets
- **Gradient training** — surrogate gradients for BPTT with standard PyTorch optimizers
- **Biologically realistic** — LIF, IF, Izhikevich, and AdEx neuron models out of the box
- **Batch support** — run N parallel trials with `model.run(batch_size=32)` for 10-50x speedups
- **State inspection** — record voltages, spikes, and weight dynamics during simulation
- **Checkpointing** — save and load trained networks with `model.save()` / `nuro.load()`
- **Bring your own data** — static tensors, generators, or configurable Poisson inputs

## Installation

```bash
# From source (recommended during early access)
git clone https://github.com/Vantar-AI/nuro.git
cd nuro
pip install -e ".[gpu,dev]"
```

**Requirements:** Python 3.10+, PyTorch 2.0+, SpikingJelly 0.0.0.0.14+

## Quick Start

### Hello Spikes

```python
import nuro

input_pop = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
output_pop = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})

conn = nuro.Connection(source=input_pop, target=output_pop, pattern="dense", plasticity="stdp")
graph = nuro.Graph([input_pop, output_pop], [conn])

model = nuro.compile(graph, target="gpu")
model.run(duration=1.0)

print(f"Total spikes: {model.metrics['total_spikes']}")
```

### Custom Input

```python
import torch
import nuro

pop = nuro.Population(size=50, dynamics="lif")
out = nuro.Population(size=10, dynamics="lif")
conn = nuro.Connection(source=pop, target=out, pattern="dense")

# Option 1: Static tensor (num_steps, pop_size)
inp = nuro.Input(population=pop, data=torch.rand(1000, 50))

# Option 2: Generator function
inp = nuro.Input(population=pop, generator=lambda step: (torch.rand(50) < 0.1).float())

# Option 3: Configurable Poisson rate
inp = nuro.Input(population=pop, mode="poisson", rate=100.0)

graph = nuro.Graph([pop, out], [conn], inputs=[inp])
```

### Biologically Realistic Neurons

```python
import nuro

# Izhikevich with preset firing patterns
exc = nuro.Population(size=800, dynamics="izhikevich", params={"preset": "regular_spiking"})
inh = nuro.Population(size=200, dynamics="izhikevich", params={"preset": "fast_spiking"})

# Adaptive Exponential IF
adex = nuro.Population(size=100, dynamics="adex")

# Or raw Izhikevich parameters
custom = nuro.Population(size=50, dynamics="izhikevich", params={"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0})
```

**Available Izhikevich presets:** `regular_spiking`, `intrinsically_bursting`, `chattering`, `fast_spiking`, `low_threshold_spiking`

### Recurrent Networks

```python
import nuro

exc = nuro.Population(size=400, dynamics="lif", params={"tau": 20e-3})
inh = nuro.Population(size=100, dynamics="lif", params={"tau": 10e-3})

conn_ei = nuro.Connection(source=exc, target=inh, pattern="dense")
conn_ie = nuro.Connection(source=inh, target=exc, pattern="dense")

graph = nuro.Graph([exc, inh], [conn_ei, conn_ie])
print(graph.is_cyclic)  # True — Nuro handles this automatically
```

### State Recording

```python
model = nuro.compile(graph, target="gpu")
model.record("voltages", population=output_pop)
model.record("spikes", population=output_pop)
model.record("weights", connection=conn, interval=100)

model.run(duration=1.0)

v = model.get_state("voltages", population=output_pop)   # (1000, pop_size)
s = model.get_state("spikes", population=output_pop)     # (1000, pop_size)
w = model.get_state("weights", connection=conn)           # (10, out, in)
```

### Batched Simulation

```python
import nuro

sensory = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
motor = nuro.Population(size=20, dynamics="lif", params={"tau": 10e-3})
conn = nuro.Connection(source=sensory, target=motor, pattern="dense")
graph = nuro.Graph([sensory, motor], [conn])

model = nuro.compile(graph, target="gpu")
model.record("spikes", population=motor)

# Run 32 independent trials in parallel
model.run(duration=1.0, batch_size=32)

spikes = model.get_state("spikes", population=motor)  # (1000, 32, 20)
per_trial = spikes.sum(dim=(0, 2))  # spike count per trial
print(f"Mean: {per_trial.mean():.0f}, Std: {per_trial.std():.0f}")
```

### Gradient Training

```python
import torch
import nuro

inp_pop = nuro.Population(size=10, dynamics="lif", params={"tau": 20e-3})
out_pop = nuro.Population(size=5, dynamics="lif", params={"tau": 10e-3})
conn = nuro.Connection(source=inp_pop, target=out_pop, pattern="dense")

data = (torch.rand(50, 10) < 0.3).float()
inp = nuro.Input(population=inp_pop, data=data)
graph = nuro.Graph([inp_pop, out_pop], [conn], inputs=[inp])

# Enable surrogate gradients
model = nuro.compile(graph, target="gpu", requires_grad=True, surrogate="atan")
optimizer = torch.optim.Adam(model.snn.parameters(), lr=1e-3)

optimizer.zero_grad()
output = model.run(duration=0.05, dt=1e-3)  # Returns spike counts dict
loss = output[out_pop.id].sum()              # Your loss here
loss.backward()
optimizer.step()
```

**Available surrogates:** `"atan"` (default), `"sigmoid"`, `"triangular"`

### Checkpointing

```python
# Save after training
model.run(duration=10.0)
model.save("trained_network.pt")

# Load later
loaded = nuro.load("trained_network.pt")
loaded.run(duration=1.0)
```

## Examples

See [`examples/basics/`](examples/basics/) for complete runnable scripts:

| Example | What it shows |
|---------|--------------|
| [`custom_input.py`](examples/basics/custom_input.py) | Static tensors, generators, custom Poisson rates |
| [`state_recording.py`](examples/basics/state_recording.py) | Recording voltages, spikes, and weight snapshots |
| [`izhikevich_network.py`](examples/basics/izhikevich_network.py) | Izh presets, AdEx, mixed-dynamics networks |
| [`recurrent_network.py`](examples/basics/recurrent_network.py) | Mutual inhibition, ring circuits, recurrent Izh |
| [`batched_simulation.py`](examples/basics/batched_simulation.py) | 32 parallel trials, cross-trial variance analysis |
| [`train_xor.py`](examples/training/train_xor.py) | XOR training with surrogate gradients and Adam optimizer |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  API Layer                                          │
│  Population · Connection · Input · Graph · compile  │
├─────────────────────────────────────────────────────┤
│  Intermediate Representation                        │
│  DynamicsNode · SynapticEdge · IRGraph              │
├─────────────────────────────────────────────────────┤
│  Backends                                           │
│  GPU (SpikingJelly) · Loihi (planned) · more        │
└─────────────────────────────────────────────────────┘
```

**Neuron models:** LIF, IF, Izhikevich, AdEx
**Training:** Surrogate gradients (ATan, sigmoid, triangular) + PyTorch optimizers
**Connectivity:** Dense, Random Sparse
**Plasticity:** STDP (trace-based)
**Graph types:** DAG (topological sort) and Cyclic (Jacobi iteration)
**Backends:** GPU (SpikingJelly) — Loihi (Lava, planned) — SpiNNaker (planned)

## Roadmap

| Version | Status | Features |
|---------|--------|----------|
| **v0.1.0** | Done | Core API, IR, GPU backend, LIF/IF, STDP |
| **v0.2.0** | Done | User inputs, state recording, Izh/AdEx, recurrent graphs, checkpointing |
| **v0.3.0** | Done | Batch support, performance benchmarks |
| **v0.4.0** | **Current** | Surrogate gradients, BPTT training, differentiable neurons |
| **v0.5.0** | Next | Intel Loihi 2 backend (Lava), weight transfer GPU→Loihi, train→deploy workflow |
| **v0.6.0** | Planned | SpiNNaker 2 backend, custom neuron dynamics on Loihi NeuroCores |
| **v0.7.0** | Planned | Vantar Cloud — remote compile and deploy to neuromorphic hardware |
| **v1.0.0** | Planned | Stable API, documentation site, model zoo |

## Testing

```bash
pip install -e ".[gpu,dev]"
pytest tests/ -v
```

109 tests covering the full stack: API validation, IR lowering, GPU execution, neuron models, recurrent graphs, state recording, checkpointing, batch support, and gradient training.

## The Vision

```
┌──────────────────────────────────────────┐
│  Nuro SDK (open source, Apache 2.0)      │
│  Define → Train on GPU → Deploy to chip  │
├──────────────────────────────────────────┤
│  Vantar Cloud (coming 2026)              │
│  Remote compile · Deploy · Monitor       │
└──────────────────────────────────────────┘
```

GPU is the training workbench. Neuromorphic silicon is the deployment target. Vantar Cloud is the managed platform. [Learn more at vantar.xyz](https://vantar.xyz)

## Early Access

Nuro is in active development. We're looking for early testers:

- **SNN researchers** who want a cleaner API than raw SpikingJelly/Norse
- **Neuromorphic engineers** with Loihi/SpiNNaker access to help build backends
- **ML engineers** curious about spiking networks for edge deployment

**How to get involved:**
1. Clone the repo and try the [examples](examples/basics/)
2. Open an [issue](https://github.com/Vantar-AI/nuro/issues) with feedback, bugs, or feature requests
3. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup

We especially want to hear: *What would make you switch from your current SNN framework?*

## License

Apache 2.0 — Free to use, modify, and distribute.

---

<div align="center">

**[Vantar AI](https://vantar.ai)** — Building the universal substrate for neural computation.

[GitHub](https://github.com/Vantar-AI/nuro) · [Issues](https://github.com/Vantar-AI/nuro/issues) · [Examples](examples/basics/)

</div>
