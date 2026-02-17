"""Recurrent Network — spiking circuits with feedback connections.

Demonstrates:
- Bidirectional (recurrent) connections
- Cycle detection
- Three-population ring circuit
- Jacobi iteration for cyclic graphs
"""

import nuro

# --- 1. Mutual inhibition with external drive ---
print("=== Mutual Inhibition Circuit ===")

# External input drives excitatory population; inhibitory feeds back
ext = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
exc = nuro.Population(size=40, dynamics="lif", params={"tau": 20e-3})
inh = nuro.Population(size=20, dynamics="lif", params={"tau": 10e-3})

conn_ext = nuro.Connection(source=ext, target=exc, pattern="dense")
conn_ei = nuro.Connection(source=exc, target=inh, pattern="dense")
conn_ie = nuro.Connection(source=inh, target=exc, pattern="dense")

graph = nuro.Graph([ext, exc, inh], [conn_ext, conn_ei, conn_ie])
print(f"  Is cyclic: {graph.is_cyclic}")

model = nuro.compile(graph, target="gpu")
model.record("spikes", population=exc)
model.record("spikes", population=inh)
model.run(duration=0.5, dt=1e-3)

exc_spikes = model.get_state("spikes", population=exc)
inh_spikes = model.get_state("spikes", population=inh)
print(f"  Excitatory spikes: {int(exc_spikes.sum().item())}")
print(f"  Inhibitory spikes: {int(inh_spikes.sum().item())}")

# --- 2. Ring circuit (A → B → C → A) ---
print("\n=== Ring Circuit (3 populations) ===")

pA = nuro.Population(size=30, dynamics="lif", params={"tau": 20e-3})
pB = nuro.Population(size=30, dynamics="lif", params={"tau": 20e-3})
pC = nuro.Population(size=30, dynamics="lif", params={"tau": 20e-3})

cAB = nuro.Connection(source=pA, target=pB, pattern="dense")
cBC = nuro.Connection(source=pB, target=pC, pattern="dense")
cCA = nuro.Connection(source=pC, target=pA, pattern="dense")

graph2 = nuro.Graph([pA, pB, pC], [cAB, cBC, cCA])
print(f"  Is cyclic: {graph2.is_cyclic}")

# Provide external input to kickstart activity
inp = nuro.Input(population=pA, mode="poisson", rate=100.0)
graph2_with_input = nuro.Graph([pA, pB, pC], [cAB, cBC, cCA], inputs=[inp])

model2 = nuro.compile(graph2_with_input, target="gpu")
model2.run(duration=0.5, dt=1e-3)

print(f"  Total spikes: {model2.metrics['total_spikes']}")
for name, pop in [("A", pA), ("B", pB), ("C", pC)]:
    count = model2.metrics["spike_counts"][pop.id]
    print(f"  Population {name}: {count} spikes")

# --- 3. Recurrent Izhikevich network ---
print("\n=== Recurrent Izhikevich Network ===")

izh1 = nuro.Population(size=50, dynamics="izhikevich")
izh2 = nuro.Population(size=50, dynamics="izhikevich", params={"preset": "fast_spiking"})

c_fwd = nuro.Connection(source=izh1, target=izh2, pattern="dense")
c_back = nuro.Connection(source=izh2, target=izh1, pattern="dense")

inp3 = nuro.Input(population=izh1, mode="poisson", rate=80.0)
graph3 = nuro.Graph([izh1, izh2], [c_fwd, c_back], inputs=[inp3])

model3 = nuro.compile(graph3, target="gpu")
model3.run(duration=0.5, dt=1e-3)

print(f"  Total spikes: {model3.metrics['total_spikes']}")
print(f"  Regular spiking: {model3.metrics['spike_counts'][izh1.id]}")
print(f"  Fast spiking:    {model3.metrics['spike_counts'][izh2.id]}")
