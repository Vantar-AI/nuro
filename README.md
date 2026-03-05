<div align="center">

# Nuro

**DevTools for neuromorphic hardware.**

Record. Track. Visualize. Deploy.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.8.0-green.svg)](https://github.com/Vantar-AI/nuro/releases/tag/v0.8.0)
[![Tests](https://img.shields.io/badge/tests-203%20passing-brightgreen.svg)](#testing)
[![Website](https://img.shields.io/badge/website-vantar.xyz-white.svg)](https://vantar.xyz)

</div>

---

The first experiment tracking platform for neuromorphic research. Capture spike trains from any hardware - Loihi, SpiNNaker, DYNAP-SE2, or GPU simulations. Track experiments with metadata and metrics. Visualize with one line. No PyTorch required.

The SNN compiler is still here too: train on GPU with surrogate gradients, deploy to neuromorphic silicon with zero code changes.

## Install

```bash
# DevTools only (no GPU/PyTorch dependency)
pip install nuro[devtools]

# Full stack (DevTools + GPU training + hardware deployment)
pip install nuro[gpu,devtools,dev]
```

## Quick Start: Record & Track

```python
import numpy as np
import nuro

# 1. Record from any source
rec = nuro.Recording(dt=1e-3, source="dynap-se2")
rec.add_probe("spikes", unit="binary")
rec.extend("spikes", np.random.randint(0, 2, (500, 128)))  # (time, neurons)

# 2. Track experiment
exp = nuro.experiment("baseline_v1", project="analog_vision")
exp.set_hardware("dynap-se2", board_id="board_03")
exp.set_params(tau=20e-3, threshold=-50.0)
exp.add_recording("main", rec)
exp.log_metrics({"mean_rate": 42.5, "accuracy": 0.89})

# 3. Save (JSON metadata + HDF5 recordings)
exp.save("./experiments")

# 4. Visualize
nuro.plot.spike_raster(rec.get("spikes"), dt=rec.dt)
nuro.plot.experiment_dashboard(rec, save_path="dashboard.png")
```

## Import from Any Hardware

```python
from nuro.adapters.samna import SamnaAdapter

# Offline event analysis (no Samna SDK needed)
events = [(0.001, 3), (0.002, 7), (0.005, 3)]
rec = SamnaAdapter.from_events(events, num_neurons=128, dt=1e-3)

# Or import from files
rec = nuro.adapters.from_csv("chip_recording.csv", dt=1e-3)
rec = nuro.adapters.from_numpy(spike_array, probe_name="spikes")
rec = nuro.adapters.from_hdf5("recording.h5")
```

## Compare Experiments

```python
# Load two experiments
exp_a = nuro.Experiment.load("./experiments/exp_a_id")
exp_b = nuro.Experiment.load("./experiments/exp_b_id")

# Compare recordings side by side
nuro.plot.compare_recordings({
    "Chip A": exp_a.get_recording("main"),
    "Chip B": exp_b.get_recording("main"),
}, metric="spikes")
```

## GPU Bridge

Already using Nuro's GPU compiler? Bridge to DevTools in one line:

```python
from nuro.adapters.gpu import recording_from_gpu_model

model = nuro.compile(graph, target="gpu")
model.record("spikes", population=out_pop)
model.run(duration=0.5)

# Convert GPU Recorder -> Recording for experiment tracking
rec = recording_from_gpu_model(model.recorder, dt=1e-3)
```

---

## The Compiler (Train & Deploy)

The original SNN compiler is still intact. Define once, train on GPU, deploy to silicon:

```python
import torch, nuro

# Define
sensory = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
motor   = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
conn    = nuro.Connection(source=sensory, target=motor, pattern="dense")
inp     = nuro.Input(population=sensory, mode="poisson", rate=100.0)
graph   = nuro.Graph([sensory, motor], [conn], inputs=[inp])

# Train on GPU
model = nuro.compile(graph, target="gpu", requires_grad=True, surrogate="atan")
optimizer = torch.optim.Adam(model.snn.parameters(), lr=1e-3)
for _ in range(100):
    optimizer.zero_grad()
    out = model.run(duration=0.1)
    loss = -out[motor.id].sum()
    loss.backward()
    optimizer.step()
    model.reset()
model.save("trained.pt")

# Deploy to silicon (one line change)
model = nuro.compile(graph, target="loihi", weights_from="trained.pt")     # Intel Loihi 2
model = nuro.compile(graph, target="spinnaker2", weights_from="trained.pt") # SpiNNaker 2
model = nuro.compile(graph, target="akida", weights_from="trained.pt")      # BrainChip Akida
```

---

## Why Nuro?

**27+ SNN frameworks** exist for simulation and training. **Zero tools** exist for experiment tracking on neuromorphic hardware.

Every lab has ad-hoc Python scripts. No structured way to capture, store, compare, or reproduce experiments across platforms. Nuro fills this gap.

| What | Nuro | Other frameworks |
|------|------|-----------------|
| Experiment tracking | Recording, Experiment, metrics, HDF5 | Ad-hoc scripts |
| Hardware adapters | GPU, Loihi, Samna, CSV, HDF5, NIR | Hardware-specific only |
| Visualization | Raster, traces, rates, dashboard, compare | DIY matplotlib |
| GPU training | Surrogate gradients, BPTT, batch | Some |
| Hardware deploy | Loihi 2, SpiNNaker 2, Akida | One target each |
| PyTorch required? | No (numpy-first for DevTools) | Usually yes |

---

## Supported Hardware

### DevTools (recording + tracking)

| Adapter | Hardware | Requires SDK? |
|---------|----------|---------------|
| `adapters.gpu` | GPU (PyTorch) | Yes |
| `adapters.lava` | Intel Loihi 2 | Yes |
| `adapters.samna` | SynSense (DYNAP-SE2, Xylo) | Live capture: yes. Offline: **no** |
| `adapters.file` | CSV, HDF5, numpy, NIR | No |

### Compiler (training + deployment)

| Target | Backend | Status |
|--------|---------|--------|
| `"gpu"` | SpikingJelly + PyTorch | Stable |
| `"loihi"` | Intel Lava | Stable |
| `"spinnaker2"` | py-spinnaker2 | Stable |
| `"akida"` | BrainChip Akida | Stable |

**Neuron models:** LIF, IF, Izhikevich (5 presets), AdEx
**Connectivity:** Dense, Random Sparse, One-to-One, Conv1D, Distance-Dependent
**Interop:** NIR import/export, ANN-to-SNN conversion

---

## Architecture

```
nuro/
  recording.py          # Hardware-agnostic Recording (numpy, HDF5)
  experiment.py         # Experiment tracking (metadata, metrics, persistence)
  plot.py               # Visualization (raster, traces, dashboard, compare)
  adapters/             # Hardware bridges (GPU, Loihi, Samna, file)
  callbacks.py          # MLOps + ExperimentCallback
  api/                  # User-facing API (Population, Connection, Graph, compile)
  ir/                   # Intermediate Representation (compiler boundary)
  backends/             # GPU, Loihi, SpiNNaker2, Akida, Cloud
  datasets/             # N-MNIST, DVS-CIFAR10, DVS Gesture
  cloud/                # Vantar Cloud experiment storage (v0.9)
```

---

## Roadmap

| Version | Status | Features |
|---------|--------|----------|
| **v0.1-0.7** | Done | Core API, IR, GPU + Loihi + SpiNNaker2 + Akida backends, NIR, ANN-to-SNN, auto-quantization |
| **v0.8.0** | **Current** | Neuromorphic DevTools: Recording, Experiment, Plot, Adapters (203 tests) |
| **v0.9.0** | Next | Vantar Cloud: experiment storage, sharing, remote hardware access |
| **v1.0.0** | Planned | Stable API, documentation site, model zoo |

---

## Testing

```bash
pip install -e ".[gpu,dev]"
pytest tests/ -v
```

203 tests covering: Recording, Experiment, Plot, Adapters, API, IR, GPU, Loihi, SpiNNaker2, Akida, neuron models, gradients, batching, checkpoints, connectivity, quantization, callbacks, datasets.

---

## FAQ

**Q: Do I need PyTorch?**
No. DevTools (Recording, Experiment, Plot, Adapters) are numpy-first. Only the GPU compiler backend needs PyTorch.

**Q: Do I need neuromorphic hardware?**
No. Import data from CSV/HDF5/numpy for offline analysis. GPU backend works on any machine. Hardware backends fall back to simulators.

**Q: Who is this for?**
Analog neuromorphic labs (DYNAP-SE2, BrainScaleS-2), digital neuromorphic teams (Loihi, SpiNNaker), and SNN researchers who want reproducible experiments.

**Q: How is this different from W&B/MLflow?**
Those tools assume dense tensors, epochs, and loss curves. Neuromorphic experiments have spike trains, event-based data, and hardware-specific recording formats. Nuro speaks the language of neuromorphic research.

**Q: Is the API stable?**
Core API (Population, Connection, Graph, compile) is stable from v0.1. DevTools API (Recording, Experiment, plot) is new in v0.8 - we aim for stability but may iterate.

---

## License

Apache 2.0 - free to use, modify, distribute.

---

<div align="center">

**[Vantar AI](https://vantar.xyz)** | [GitHub](https://github.com/Vantar-AI/nuro) | [Issues](https://github.com/Vantar-AI/nuro/issues)

</div>
