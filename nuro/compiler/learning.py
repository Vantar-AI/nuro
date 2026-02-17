"""Learning rule compiler — translates abstract plasticity rules to target-specific implementations.

Currently a pass-through. The GPU backend handles STDP directly via
nuro.backends.gpu.plasticity.STDPUpdater. This module will grow when
additional backends or custom learning rules are added.
"""
