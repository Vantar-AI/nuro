# Nuro Documentation

**The universal SDK for spiking neural networks.**
Train on GPU. Deploy to neuromorphic silicon. One API, any backend.

---

## Getting Started

| | |
|--|--|
| [Quick Start](quickstart.md) | Install Nuro and run your first SNN in 5 minutes |
| [Training Guide](training.md) | Surrogate gradients, BPTT, optimizers |
| [Deployment Guide](deployment.md) | Hardware deploy: Loihi 2, SpiNNaker 2 |
| [Backend Comparison](backends.md) | Which backend to use and when |
| [API Reference](api-reference.md) | All public classes and functions |
| [Vantar Cloud](cloud.md) | Remote compile and deploy (beta) |

---

## What is Nuro?

Nuro is a Python SDK that compiles spiking neural networks (SNNs) to multiple backends. You define your network once using a clean Python API, train it with surrogate gradients on GPU, then recompile the same network to neuromorphic hardware (Intel Loihi 2, SpiNNaker 2) with zero code changes.

```
Define (Python) → Train (GPU) → Deploy (Loihi / SpiNNaker / Cloud)
                     ↑
             surrogate gradients
             standard PyTorch optimizers
             10-50x batch speedups
```

The intermediate representation (IR) is the hard boundary. Backends only see IR objects, never your API code.

---

## Quick Example

```python
import nuro

inp  = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
out  = nuro.Population(size=10,  dynamics="lif", params={"tau": 10e-3})
conn = nuro.Connection(source=inp, target=out, pattern="dense")
graph = nuro.Graph([inp, out], [conn])

model = nuro.compile(graph, target="gpu")
model.run(duration=1.0)
print(model.metrics["total_spikes"])
```

Change `target="gpu"` to `target="loihi"` or `target="spinnaker2"` — everything else stays the same.

---

## Version: 0.6.0

- Intel Loihi 2 backend (Lava) — stable
- SpiNNaker 2 backend (py-spinnaker2 + Brian2) — stable
- GPU training with surrogate gradients — stable
- 121 tests passing

---

## License

Apache 2.0. Free to use, modify, distribute. [GitHub →](https://github.com/Vantar-AI/nuro)
