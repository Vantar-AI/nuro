# Neuron Models

Mathematical descriptions of all neuron models supported by Nuro.

---

## Leaky Integrate-and-Fire (LIF)

The workhorse of SNN research. Simple, efficient, hardware-friendly.

### Dynamics

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)$$

When $V \geq V_{thresh}$: spike, then $V \leftarrow V_{reset}$

### Discrete-time (simulation)

$$V[t+1] = \beta \cdot V[t] + (1 - \beta) \cdot I[t]$$

where $\beta = e^{-dt/\tau_m}$ is the decay factor.

### Parameters

| Symbol | Parameter | Default | Unit |
|--------|-----------|---------|------|
| $\tau_m$ | `tau` | 20e-3 | seconds |
| $V_{thresh}$ | `v_thresh` | 1.0 | normalized |
| $V_{reset}$ | `v_reset` | 0.0 | normalized |
| $V_{rest}$ | `v_rest` | 0.0 | normalized |

### Backend mapping

- **GPU:** SpikingJelly `LIFNode` with configurable surrogate gradient
- **Loihi:** Native Lava `LIF` Process (runs on neuromorphic cores)
- **SpiNNaker 2:** `snn.Population(neuron_model="LIF")`
- **Akida:** Dense/Convolutional spiking layer

---

## Integrate-and-Fire (IF)

LIF without leak. Pure accumulator. Best for rate coding.

### Dynamics

$$\frac{dV}{dt} = I(t)$$

When $V \geq V_{thresh}$: spike, then $V \leftarrow V_{reset}$

### Discrete-time

$$V[t+1] = V[t] + I[t]$$

### Backend mapping

- **GPU:** SpikingJelly `IFNode`
- **Loihi:** Lava `LIF` with `du=0` (no decay)
- **SpiNNaker 2:** `snn.Population(neuron_model="IF")`
- **Akida:** Dense spiking layer

---

## Izhikevich

Rich dynamics with 5 presets covering diverse cortical neuron types.

### Dynamics

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$

$$\frac{du}{dt} = a(bv - u)$$

When $v \geq 30$: spike, then $v \leftarrow c$, $u \leftarrow u + d$

### Presets

| Preset | a | b | c | d | Behavior |
|--------|---|---|---|---|----------|
| `regular_spiking` | 0.02 | 0.2 | -65 | 8 | Cortical pyramidal |
| `intrinsically_bursting` | 0.02 | 0.2 | -55 | 4 | Bursting |
| `chattering` | 0.02 | 0.2 | -50 | 2 | Fast bursts |
| `fast_spiking` | 0.1 | 0.2 | -65 | 2 | Interneurons |
| `low_threshold_spiking` | 0.02 | 0.25 | -65 | 2 | Inhibitory |

### Backend mapping

- **GPU:** Custom `IzhikevichNode` with surrogate gradients
- **Loihi:** Simulation only (custom ProcessModel)
- **SpiNNaker 2:** Simulation only
- **Akida:** Not supported

---

## Adaptive Exponential IF (AdEx)

Captures subthreshold oscillations and adaptation.

### Dynamics

$$C \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{(V-V_T)/\Delta_T} - w + I$$

$$\tau_w \frac{dw}{dt} = a(V - E_L) - w$$

When $V \geq V_{cutoff}$: spike, then $V \leftarrow V_{reset}$, $w \leftarrow w + b$

### Parameters

| Symbol | Parameter | Default | Unit |
|--------|-----------|---------|------|
| $C$ | `C` | 281e-12 | F |
| $g_L$ | `g_L` | 30e-9 | S |
| $E_L$ | `E_L` | -70.6e-3 | V |
| $V_T$ | `V_T` | -50.4e-3 | V |
| $\Delta_T$ | `delta_T` | 2e-3 | V |
| $a$ | `a` | 4e-9 | S |
| $b$ | `b` | 0.0805e-9 | A |
| $\tau_w$ | `tau_w` | 144e-3 | s |

### Backend mapping

- **GPU:** Custom `AdExNode` with surrogate gradients
- **Loihi:** Simulation only
- **SpiNNaker 2:** Simulation only
- **Akida:** Not supported

---

## Surrogate Gradients

For gradient-based training, non-differentiable spikes are replaced with smooth approximations:

### Atan (default)

$$\sigma(x) = \frac{1}{\pi} \arctan(\pi x) + \frac{1}{2}$$

Gradient: $\sigma'(x) = \frac{1}{1 + (\pi x)^2}$

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-\alpha x}}$$

Gradient: $\sigma'(x) = \alpha \sigma(x)(1 - \sigma(x))$

### Triangular

$$\sigma'(x) = \max(0, 1 - |x|)$$

Sharpest surrogate — best for sparse networks.
