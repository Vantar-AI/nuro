# Deployment Guide

Deploy trained Nuro networks to neuromorphic hardware.

---

## Overview

Nuro's train → deploy workflow:

```
1. Define network once (Population, Connection, Graph)
2. Train on GPU (surrogate gradients, PyTorch optimizers)
3. Save weights (model.save("trained.pt"))
4. Recompile to hardware target (weights_from="trained.pt")
5. Run on silicon
```

No code changes between steps 2 and 4. The IR is the contract.

---

## Intel Loihi 2

**Backend:** `lava-nc` (Intel's Lava framework)
**Status:** Stable (v0.5.0)

### Install

```bash
pip install -e ".[loihi]"
# Note: lava-nc requires Python ≤3.10
```

### Simulation (no hardware needed)

```python
import nuro

# ... define graph ...
model = nuro.compile(graph, target="loihi")
model.record("spikes", population=output_pop)
model.run(duration=1.0)
spikes = model.get_state("spikes", population=output_pop)
```

### With pre-trained weights

```python
# Train on GPU first
gpu_model = nuro.compile(graph, target="gpu", requires_grad=True)
# ... training loop ...
gpu_model.save("trained.pt")

# Deploy to Loihi
loihi_model = nuro.compile(graph, target="loihi", weights_from="trained.pt")
loihi_model.run(duration=1.0)
print(loihi_model.metrics)
```

### Real hardware (INRC access required)

```python
from lava.magma.core.run_configs import Loihi2HwCfg

loihi_model = nuro.compile(graph, target="loihi", weights_from="trained.pt")
loihi_model.set_run_config(Loihi2HwCfg())   # Switch to hardware
loihi_model.run(duration=1.0)
```

Apply for Intel Neuromorphic Research Community (INRC) access at [intel-research.net/neuromorphic-computing](https://intel-research.net/neuromorphic-computing).

### Neuron model support on Loihi

| Model | Loihi support |
|-------|--------------|
| LIF | Native (Lava LIF Process) |
| IF | Native (Lava LIF with no leak) |
| Izhikevich | Simulation only (v0.5); NcProcess (v0.6) |
| AdEx | Simulation only (v0.5); NcProcess (v0.6) |

---

## SpiNNaker 2

**Backend:** `py-spinnaker2` + Brian2 (simulator fallback)
**Status:** Stable (v0.6.0)

### Install

```bash
pip install -e ".[spinnaker2]"
```

### Simulation (Brian2)

```python
model = nuro.compile(graph, target="spinnaker2")
model.record("spikes", population=output_pop)
model.run(duration=1.0)
```

### With pre-trained weights

```python
sp2_model = nuro.compile(graph, target="spinnaker2", weights_from="trained.pt")
sp2_model.run(duration=1.0)
print(sp2_model.metrics)
```

### Real hardware

```python
sp2_model.set_hardware(True)   # Switches from Brian2 to real SpiNNaker 2 chip
sp2_model.run(duration=1.0)
```

Access SpiNNaker 2 hardware via [SpiNNcloud](https://spinncloud.com) or the [Human Brain Project](https://www.humanbrainproject.eu).

---

## Vantar Cloud (v0.7.0)

Deploy to neuromorphic hardware without owning the chip:

```python
model = nuro.compile(
    graph,
    target="cloud",
    hardware="loihi",           # or "spinnaker2"
    weights_from="trained.pt",
    api_key="your-key-here",    # or set VANTAR_API_KEY env var
)
model.run(duration=1.0)
```

[Join the waitlist →](https://vantar.xyz)

---

## Weight Transfer

When you pass `weights_from="trained.pt"`, Nuro:

1. Loads the PyTorch checkpoint
2. Extracts weight matrices per `Connection`
3. Quantizes to fixed-point (where hardware requires it)
4. Maps to the target backend's synapse format

Weights are matched by connection identity (source population → target population). The graph structure must be identical between GPU and deployment compilation.

---

## Limitations

- **Recurrent graphs on Loihi**: not yet supported (planned v0.7)
- **Custom dynamics on Loihi hardware**: Izhikevich/AdEx use simulation mode; NcProcess support in v0.6
- **Weight quantization**: fixed-point scale factor stubbed; full implementation in v0.7
- **Batch size**: hardware backends always run `batch_size=1` (parallelism is in the hardware architecture)

---

## Troubleshooting

**ImportError: lava-nc not found**
```bash
pip install -e ".[loihi]"   # Python 3.10 required
```

**Weights don't match**
The graph structure must be identical. Compile the same `graph` object (or equivalent structure) for both GPU and deployment.

**Hardware not responding**
Check INRC access credentials and network connectivity to the Loihi board.
