"""Example: Convert a trained PyTorch MLP to a Nuro SNN.

Demonstrates the ANN-to-SNN conversion pipeline:
1. Train a simple MLP on a classification task
2. Convert to SNN using rate-based coding
3. Compile and run on GPU
4. (Optional) Deploy to neuromorphic hardware
"""

import torch
import torch.nn as nn

import nuro


def main():
    # 1. Define and "train" a simple MLP
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    # In practice, you'd train this on MNIST first
    print(f"ANN model: {sum(p.numel() for p in model.parameters())} parameters")

    # 2. Convert ANN to SNN
    graph = nuro.convert_ann(model, input_shape=(784,), num_steps=100)
    print(f"SNN graph: {graph.num_populations} populations, {graph.num_connections} connections")

    # 3. Normalize weights for better spiking behavior
    from nuro.conversion import normalize_weights
    graph = normalize_weights(graph, method="robust")

    # 4. Compile to GPU and run
    compiled = nuro.compile(graph, target="gpu")
    compiled.run(duration=0.1, dt=1e-3)
    print(f"Total spikes: {compiled.metrics['total_spikes']}")
    print(f"Steps: {compiled.metrics['num_steps']}")

    # 5. Deploy to Loihi (uncomment if Lava SDK available)
    # loihi_model = nuro.compile(graph, target="loihi", weights_from="trained.pt")
    # loihi_model.run(duration=0.1)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
