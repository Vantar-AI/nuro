# Training Guide

Train spiking neural networks with surrogate gradients and standard PyTorch optimizers.

---

## The Problem with Backpropagation Through Spikes

Spikes are binary events — discontinuous by definition. The Heaviside step function has zero gradient almost everywhere, making standard backpropagation impossible.

**Surrogate gradients** replace the Heaviside derivative with a smooth function during the backward pass. The forward pass still uses real spikes (binary). Only the gradient computation is smoothed.

```
Forward:   spike = Heaviside(v - threshold)   → binary {0, 1}
Backward:  ∂spike/∂v ≈ surrogate(v - threshold)  → smooth gradient
```

---

## Basic Training Loop

```python
import torch
import nuro

# Define network
inp_pop = nuro.Population(size=10,  dynamics="lif", params={"tau": 20e-3})
out_pop = nuro.Population(size=5,   dynamics="lif", params={"tau": 10e-3})
conn    = nuro.Connection(source=inp_pop, target=out_pop, pattern="dense")
data    = (torch.rand(50, 10) < 0.3).float()
inp     = nuro.Input(population=inp_pop, data=data)
graph   = nuro.Graph([inp_pop, out_pop], [conn], inputs=[inp])

# Compile with surrogate gradients
model = nuro.compile(graph, target="gpu", requires_grad=True, surrogate="atan")
optimizer = torch.optim.Adam(model.snn.parameters(), lr=1e-3)

# Training loop
for step in range(100):
    optimizer.zero_grad()
    model.reset()

    # run() returns dict[pop_id → spike_tensor] when requires_grad=True
    output = model.run(duration=0.05, dt=1e-3)

    # Spike count as logits → your loss
    spike_counts = output[out_pop.id].sum(dim=0)   # (5,) total spikes per neuron
    target = torch.tensor([1.0, 0, 0, 0, 0])       # want neuron 0 to spike most
    loss = ((spike_counts / spike_counts.sum()) - target).pow(2).mean()

    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")
```

---

## Surrogate Functions

| Name | Description | Sharpness |
|------|-------------|-----------|
| `"atan"` | Arc-tangent (default) | Medium, stable |
| `"sigmoid"` | Logistic sigmoid | Softer |
| `"triangular"` | Triangle window | Sharp, local |

```python
# Use ATan (recommended)
model = nuro.compile(graph, target="gpu", requires_grad=True, surrogate="atan")

# Softer gradient
model = nuro.compile(graph, target="gpu", requires_grad=True, surrogate="sigmoid")
```

---

## Accessing Parameters

The underlying PyTorch module is at `model.snn`:

```python
# All parameters (weights + biases)
params = list(model.snn.parameters())
print(f"Parameter tensors: {len(params)}")
print(f"Total params: {sum(p.numel() for p in params):,}")

# For Adam with weight decay
optimizer = torch.optim.AdamW(model.snn.parameters(), lr=1e-3, weight_decay=1e-4)
```

---

## Batch Training

Run multiple samples in parallel for significant speedups:

```python
# 32 parallel trials per step
output = model.run(duration=0.05, batch_size=32)

# spike tensor shape: (T, B, N) = (50, 32, 5)
spikes = output[out_pop.id]
batch_spike_counts = spikes.sum(dim=0)  # (32, 5)
```

On RTX 4090: 32 parallel trials takes the same wall time as 1 trial.

---

## Checkpointing

```python
# Save after training
model.save("my_network.pt")

# Load later (same graph definition required)
loaded = nuro.load("my_network.pt")
loaded.run(duration=1.0)
```

---

## Train → Deploy Workflow

After training, recompile to neuromorphic hardware:

```python
# Train on GPU
gpu_model = nuro.compile(graph, target="gpu", requires_grad=True)
# ... training loop ...
gpu_model.save("trained.pt")

# Deploy to Loihi (zero code changes)
loihi_model = nuro.compile(graph, target="loihi", weights_from="trained.pt")
loihi_model.run(duration=1.0)

# Or SpiNNaker 2
sp2_model = nuro.compile(graph, target="spinnaker2", weights_from="trained.pt")
sp2_model.run(duration=1.0)
```

---

## STDP (Spike-Timing Dependent Plasticity)

Unsupervised local learning rule — enabled during simulation, disabled during gradient training:

```python
conn = nuro.Connection(
    source=inp_pop,
    target=out_pop,
    pattern="dense",
    plasticity="stdp",
)
model = nuro.compile(graph, target="gpu")  # STDP active (no requires_grad)
model.run(duration=10.0)
```

Note: STDP is automatically disabled when `requires_grad=True`.

---

## Tips

- **Reset between epochs**: call `model.reset()` before each forward pass to clear membrane potentials
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.snn.parameters(), 1.0)` prevents instability
- **Short simulations**: 20-100 timesteps per training step is common (more = slower, not always better)
- **Learning rate**: 1e-3 (Adam) is a good start; decay with `StepLR` or `CosineAnnealingLR`

---

## See Also

- [`examples/training/train_xor.py`](../examples/training/train_xor.py) — XOR classification
- [`examples/training/mnist_snn.py`](../examples/training/mnist_snn.py) — MNIST with rate coding
- [API Reference](api-reference.md) — `compile()`, `Population`, `Connection`
