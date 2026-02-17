<div align="center">

# Nuro

**The universal programming language for neuromorphic, thermodynamic, and biological computing.**

*Write once. Compile to any substrate.*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## The Problem

NVIDIA didn't win AI because of better transistors. They won because **CUDA** let a million developers write parallel code without understanding graphics pipelines.

The entire neuromorphic, thermodynamic, and organoid intelligence sector is stuck at the **pre-CUDA stage**: powerful hardware, almost no one can program it.

- **SpikingJelly** has a few thousand active users
- **Intel Lava** is mature but locked to Loihi
- **BrainPy** is elegant but academic
- **FinalSpark** gives you a REST API to a dish of neurons
- **Extropic THRML** just shipped v0.1.3 for thermodynamic sampling

None of these talk to each other. A developer who learns one framework is locked into one hardware paradigm.

## The Solution

Nuro is a unified programming abstraction that compiles to spiking silicon, thermodynamic ASICs, and living neurons.

```python
import nuro

# Define populations — the runtime decides the substrate
sensory = nuro.Population(size=1024, dynamics="lif", params={"tau": 20e-3})
motor = nuro.Population(size=64, dynamics="lif", params={"tau": 10e-3})

# Connect them — topology is logical, compilation is physical
conn = nuro.Connection(
    source=sensory, target=motor,
    pattern="random_sparse",
    plasticity="stdp",
)

# Define what the system should optimize — not how
objective = nuro.Objective(
    type="minimize_surprise",
    prediction=motor,
    observation=nuro.EventStream("camera_dvs"),
)

# Compile to any target
graph = nuro.Graph([sensory, motor], [conn], [objective])
runtime = nuro.compile(graph, target="auto")
runtime.run(duration=60.0)
```

**The same code compiles to:**
- 🧠 Intel Loihi 3 (spiking silicon)
- ⚡ SpiNNaker 2 / SpiNNcloud (general-purpose neuromorphic)
- 🌡️ Normal Computing CN101 (thermodynamic)
- 🧬 Extropic TSU (stochastic thermodynamic)
- 🔬 FinalSpark / Cortical Labs (biological neurons)
- 🖥️ GPU fallback (via SpikingJelly, for development)

## Key Design Principles

1. **Dynamics-first, not tensor-first.** The primitive is a dynamical system, not a matrix.
2. **Probabilistic by default.** Variables carry uncertainty. A "value" is a distribution unless you explicitly collapse it.
3. **Topology-aware compilation.** You define logical connectivity. The compiler handles physical mapping.
4. **Target-agnostic source, target-specific compilation.** One language, every backend.
5. **Feedback loops are first-class.** Closed-loop sensing-acting-learning is the default.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  DEVELOPER API  (Python-native, PyTorch-like)           │
│  nuro.Population / nuro.Connection / nuro.Objective     │
├─────────────────────────────────────────────────────────┤
│  GRAPH IR  (NIR-Extended)                               │
│  Dynamics nodes + Probabilistic edges +                 │
│  Temporal annotations + Topology constraints            │
├─────────────────────────────────────────────────────────┤
│  COMPILER                                               │
│  Graph partitioning · Time-domain mapping ·             │
│  Learning rule translation · Hardware negotiation       │
├─────────────────────────────────────────────────────────┤
│  BACKENDS                                               │
│  Loihi 3 · SpiNNaker 2 · CN101 · TSU · biOS · GPU     │
└─────────────────────────────────────────────────────────┘
```

## Roadmap

| Phase | Timeline | Milestone |
|-------|----------|-----------|
| **Phase 1** | 2026 Q1-Q2 | Core API + GPU backend (SpikingJelly) + Loihi backend (Lava) |
| **Phase 2** | 2026 Q3-Q4 | SpiNNaker 2 backend + thermodynamic backend (THRML/thermox) |
| **Phase 3** | 2027 H1 | Biological backend (FinalSpark) + task multiplexing |
| **Phase 4** | 2027 H2+ | Cloud compilation service + model zoo + developer ecosystem |

## Installation

```bash
pip install nuro  # coming soon
```

## Current Status

🚧 **Pre-alpha** — Designing core abstractions and building the GPU backend.

We're actively looking for:
- Neuromorphic computing researchers
- Compiler engineers
- Contributors with access to Loihi / SpiNNaker / thermodynamic hardware

## Research

See our foundational research memo: [The CUDA Moment: Designing the Universal Language for Physics-Native and Biological Computing](https://github.com/Vantar-AI/research)

## License

Apache 2.0 — Free to use, modify, and distribute. The core SDK will always be open.

---

<div align="center">

**Vantar AI** — The CUDA of post-Von Neumann computing.

[Website](https://vantar.ai) · [Research](https://github.com/Vantar-AI/research) · [Examples](https://github.com/Vantar-AI/nuro-examples)

</div>
