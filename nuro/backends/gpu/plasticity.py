"""GPU STDP plasticity — trace-based spike-timing-dependent plasticity."""

from __future__ import annotations

import torch
import torch.nn as nn


class STDPUpdater:
    """Classical trace-based STDP learning rule.

    For each synapse connecting pre → post:
        dW = A_plus * pre_trace * post_spike - A_minus * post_trace * pre_spike

    Traces decay exponentially each timestep.

    Parameters
    ----------
    synapse : nn.Linear
        The synapse layer whose weights are updated.
    tau_pre : float
        Pre-synaptic trace time constant (unitless, in timesteps).
    tau_post : float
        Post-synaptic trace time constant (unitless, in timesteps).
    a_plus : float
        LTP learning rate.
    a_minus : float
        LTD learning rate.
    w_min : float
        Minimum weight.
    w_max : float
        Maximum weight.
    """

    def __init__(
        self,
        synapse: nn.Linear,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ) -> None:
        self.synapse = synapse
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self.pre_trace: torch.Tensor | None = None
        self.post_trace: torch.Tensor | None = None

    def reset(self) -> None:
        self.pre_trace = None
        self.post_trace = None

    def step(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """Update traces and weights for one timestep.

        Parameters
        ----------
        pre_spikes : Tensor of shape (batch, pre_size) or (pre_size,)
            Binary spike tensor from the pre-synaptic population.
        post_spikes : Tensor of shape (batch, post_size) or (post_size,)
            Binary spike tensor from the post-synaptic population.
        """
        pre = pre_spikes.detach().float()
        post = post_spikes.detach().float()

        if pre.dim() > 1:
            pre = pre.mean(dim=0)
        if post.dim() > 1:
            post = post.mean(dim=0)

        device = self.synapse.weight.device

        if self.pre_trace is None:
            self.pre_trace = torch.zeros(pre.shape[-1], device=device)
        if self.post_trace is None:
            self.post_trace = torch.zeros(post.shape[-1], device=device)

        # Decay traces
        self.pre_trace *= 1.0 - 1.0 / self.tau_pre
        self.post_trace *= 1.0 - 1.0 / self.tau_post

        # Update traces with new spikes
        self.pre_trace += pre
        self.post_trace += post

        # Weight update: outer product of traces and spikes
        # dW[post, pre] = A+ * post_spike * pre_trace - A- * pre_spike * post_trace
        dw = (
            self.a_plus * torch.outer(post, self.pre_trace)
            - self.a_minus * torch.outer(self.post_trace, pre)
        )

        with torch.no_grad():
            self.synapse.weight.add_(dw)
            self.synapse.weight.clamp_(self.w_min, self.w_max)
