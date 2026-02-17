"""Custom Input — loading external spike data into a Nuro network.

Demonstrates:
- Static tensor input (pre-computed spike trains)
- Generator-based input (per-timestep callable)
- Custom Poisson rate
"""

import torch

import nuro

# --- 1. Static tensor input ---
print("=== Static tensor input ===")

input_pop = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
output_pop = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
conn = nuro.Connection(source=input_pop, target=output_pop, pattern="dense")

# Pre-compute 100 steps of spike data (50 neurons, ~10% firing rate)
num_steps = 100
spike_data = (torch.rand(num_steps, 50) < 0.1).float()
inp = nuro.Input(population=input_pop, data=spike_data)

graph = nuro.Graph([input_pop, output_pop], [conn], inputs=[inp])
model = nuro.compile(graph, target="gpu")
model.run(duration=0.1, dt=1e-3)

print(f"  Total spikes: {model.metrics['total_spikes']}")
print(f"  Input spikes: {model.metrics['spike_counts'][input_pop.id]}")
print(f"  Output spikes: {model.metrics['spike_counts'][output_pop.id]}")

# --- 2. Generator input ---
print("\n=== Generator input ===")

input_pop2 = nuro.Population(size=30, dynamics="lif", params={"tau": 20e-3})
output_pop2 = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
conn2 = nuro.Connection(source=input_pop2, target=output_pop2, pattern="dense")


def burst_generator(step: int) -> torch.Tensor:
    """Generate bursts: high activity every 20 steps, quiet otherwise."""
    if step % 20 < 5:
        return (torch.rand(30) < 0.3).float()  # 30% firing during burst
    return torch.zeros(30)


inp2 = nuro.Input(population=input_pop2, generator=burst_generator)
graph2 = nuro.Graph([input_pop2, output_pop2], [conn2], inputs=[inp2])
model2 = nuro.compile(graph2, target="gpu")
model2.run(duration=0.1, dt=1e-3)

print(f"  Total spikes: {model2.metrics['total_spikes']}")

# --- 3. Custom Poisson rate ---
print("\n=== Custom Poisson rate (200 Hz) ===")

input_pop3 = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
output_pop3 = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
conn3 = nuro.Connection(source=input_pop3, target=output_pop3, pattern="dense")
inp3 = nuro.Input(population=input_pop3, mode="poisson", rate=200.0)

graph3 = nuro.Graph([input_pop3, output_pop3], [conn3], inputs=[inp3])
model3 = nuro.compile(graph3, target="gpu")
model3.run(duration=0.1, dt=1e-3)

print(f"  Total spikes: {model3.metrics['total_spikes']}")
print(f"  Input spikes: {model3.metrics['spike_counts'][input_pop3.id]}")
print(f"  (Compare to default 50 Hz — higher rate means more activity)")
