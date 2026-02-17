"""Deploy to Loihi — train on GPU, deploy to neuromorphic hardware.

This example demonstrates the train-on-GPU → deploy-to-Loihi workflow:
1. Define a network with Nuro's API
2. Train with surrogate gradients on GPU
3. Save the trained weights
4. Recompile to Loihi backend with weight transfer
5. Run inference on Loihi (simulator or hardware)

Requirements:
    pip install nuro[gpu,loihi]
"""

import numpy as np

import nuro

# ── 1. Define the network ───────────────────────────────────────────
sensory = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
hidden = nuro.Population(size=30, dynamics="lif", params={"tau": 15e-3})
motor = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})

conn1 = nuro.Connection(source=sensory, target=hidden, pattern="dense")
conn2 = nuro.Connection(source=hidden, target=motor, pattern="dense")

# Static input data
data = (np.random.rand(100, 50) < 0.1).astype(np.float32)
inp = nuro.Input(population=sensory, data=data)

graph = nuro.Graph(
    [sensory, hidden, motor],
    [conn1, conn2],
    inputs=[inp],
)

# ── 2. Train on GPU with surrogate gradients ────────────────────────
print("Training on GPU...")

try:
    import torch

    gpu_model = nuro.compile(graph, target="gpu", requires_grad=True)
    optimizer = torch.optim.Adam(gpu_model.snn.parameters(), lr=1e-3)

    for epoch in range(5):
        optimizer.zero_grad()
        output = gpu_model.run(duration=0.1, dt=1e-3)
        # Simple loss: encourage motor population to spike
        loss = -output[motor.id].sum()
        loss.backward()
        optimizer.step()
        gpu_model.reset()
        print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

    # Save trained weights
    gpu_model.save("trained_network.pt")
    print("Saved GPU checkpoint: trained_network.pt")

except ImportError:
    print("PyTorch/SpikingJelly not available — skipping GPU training.")
    print("Creating a dummy checkpoint for demonstration...")
    # For demo without GPU deps, compile to Loihi directly
    loihi_model = nuro.compile(graph, target="loihi")
    loihi_model.run(duration=0.1)
    print(f"Loihi sim metrics: {loihi_model.metrics}")
    exit()

# ── 3. Deploy to Loihi ──────────────────────────────────────────────
print("\nDeploying to Loihi (simulation mode)...")

loihi_model = nuro.compile(
    graph,
    target="loihi",
    weights_from="trained_network.pt",
)

# Record spikes on motor population
loihi_model.record("spikes", population=motor)

# Run inference
loihi_model.run(duration=0.1)

print(f"Total spikes: {loihi_model.metrics['total_spikes']}")
print(f"Simulation steps: {loihi_model.metrics['num_steps']}")

# Retrieve recorded spikes
spikes = loihi_model.get_state("spikes", population=motor)
print(f"Motor spike shape: {spikes.shape}")  # (100, 10)
print(f"Motor spike count: {int(np.sum(spikes))}")

# ── 4. Switch to hardware (requires INRC access) ────────────────────
# Uncomment for real Loihi 2 hardware:
#
# from lava.magma.core.run_configs import Loihi2HwCfg
# loihi_model.set_run_config(Loihi2HwCfg())
# loihi_model.run(duration=1.0)

print("\nDone! Train on GPU → Deploy to Loihi in 3 lines of code.")
