# Hyperparameter Tuning Guide

A practical guide to tuning Nuro SNN hyperparameters for different tasks.

---

## Neuron Parameters

### LIF (Leaky Integrate-and-Fire)

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `tau` | 5e-3 - 100e-3 | 20e-3 | Membrane time constant (seconds). Lower = faster response, less temporal integration |
| `v_thresh` | 0.5 - 5.0 | 1.0 | Spike threshold. Higher = fewer spikes, more selective |
| `v_reset` | -80 - -50 | -70.0 | Reset potential after spike |
| `v_rest` | -70 - -55 | -65.0 | Resting potential |

**Guidelines:**
- Classification tasks: `tau=10e-3 - 20e-3` (fast temporal dynamics)
- Temporal pattern recognition: `tau=50e-3 - 100e-3` (long memory)
- Edge detection: `tau=5e-3` (very fast, sensitive to change)

### IF (Integrate-and-Fire)

Same as LIF but no leak (`tau = infinity`). Good for rate-coding tasks where temporal dynamics don't matter.

### Izhikevich

Use presets for common firing patterns:
- `"regular_spiking"` — cortical pyramidal neurons
- `"fast_spiking"` — cortical interneurons, best for classification
- `"intrinsically_bursting"` — temporal coding
- `"chattering"` — fast burst responses
- `"low_threshold_spiking"` — inhibitory neurons

---

## Training Parameters

### Learning Rate

| Task | Recommended LR | Scheduler |
|------|----------------|-----------|
| MNIST classification | 1e-3 | StepLR(step=10, gamma=0.5) |
| Temporal patterns | 5e-4 | CosineAnnealingLR |
| Large networks (>1k neurons) | 1e-4 | ReduceLROnPlateau |

### Surrogate Gradient

| Function | Best For | Smoothness |
|----------|----------|------------|
| `"atan"` | General purpose (default) | Smooth |
| `"sigmoid"` | When atan has gradient issues | Very smooth |
| `"triangular"` | Sparse spiking networks | Sharp |

### Batch Size

- Start with `batch_size=32`
- Increase to 64-128 for larger datasets
- GPU memory is the main constraint

---

## Simulation Parameters

### Timestep (dt)

| Value | Use Case |
|-------|----------|
| 1e-3 (1ms) | Default. Good for most tasks |
| 1e-4 (0.1ms) | High temporal precision, temporal coding |
| 5e-3 (5ms) | Fast simulation, rate coding |

### Duration

- Classification: 50-200ms (50-200 steps at dt=1ms)
- Temporal tasks: 500ms-2s
- STDP learning: 1-10s

### Number of Timesteps for ANN-to-SNN

Higher `num_steps` = better accuracy but slower inference:
- Quick test: 50 steps
- Production: 100-200 steps
- High accuracy: 500+ steps

---

## Weight Initialization

Nuro auto-scales weights based on population sizes. Override with `params`:

```python
conn = nuro.Connection(
    source=pop1, target=pop2,
    params={"weight_scale": 0.1}  # Manual scale
)
```

### For hardware deployment

After GPU training, weights are quantized:
- **Loihi:** 8-bit signed integers [-256, 254]
- **SpiNNaker 2:** 4-bit [-15, 15]
- **Akida:** 4-bit default, supports 1/2/4/8-bit

Use `normalize_weights()` before deployment to maximize quantization resolution.

---

## Network Architecture

### Small classification (MNIST)

```python
inp = nuro.Population(size=784, dynamics="if")
hidden = nuro.Population(size=256, dynamics="lif", params={"tau": 10e-3})
out = nuro.Population(size=10, dynamics="lif", params={"tau": 20e-3})
```

### Temporal processing

```python
inp = nuro.Population(size=64, dynamics="lif", params={"tau": 50e-3})
rec = nuro.Population(size=128, dynamics="lif", params={"tau": 100e-3})
# Add recurrent connection:
nuro.Connection(source=rec, target=rec, pattern="random_sparse", params={"sparsity": 0.9})
```

### Balanced excitation/inhibition

```python
exc = nuro.Population(size=800, dynamics="lif")
inh = nuro.Population(size=200, dynamics="lif")
# Inhibitory connections use negative initial weights
```
