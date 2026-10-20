"""Experimental innovation-gated associative memory alongside the liquid core.

This is a local combination of continuous decay and delta-rule memory, not a
claim of a new class of neural networks. State is explicit, causal and independent
of sequence length; no token cache or attention softmax is used.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import RMSNorm


class InnovationMemory(nn.Module):
    """Read old associations, then correct only their unexplained value component.

    With unit key k, bounded value v and decayed state A:
        error = v - A k
        novelty = mean(error**2) / (mean(error**2) + positive_threshold)
        M_next = A + beta * novelty * error k^T

    beta and each row's decay are continuous-time, input-dependent rates.
    The read uses A (before this token's write), and never feeds the persistent
    matrix through normalization or dropout. The surrounding LTC state is unchanged.
    """

    def __init__(self, hidden_size: int, memory_dim: int) -> None:
        super().__init__()
        if memory_dim < 1:
            raise ValueError("memory_dim must be >= 1")
        self.memory_dim = memory_dim
        self.norm = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * memory_dim, bias=False)
        self.control = nn.Linear(hidden_size, memory_dim + 2)
        self.output = nn.Linear(memory_dim, hidden_size, bias=False)
        # Diverse initial e-folding horizons, not three banks averaged to one.
        rates = torch.logspace(math.log10(1 / 1024), math.log10(1 / 32), memory_dim)
        self.log_rate = nn.Parameter(torch.log(torch.expm1(rates)))
        self.log_threshold = nn.Parameter(torch.tensor(math.log(math.expm1(0.25))))
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.output.weight, gain=0.25)
        nn.init.zeros_(self.control.weight)
        nn.init.zeros_(self.control.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        dtype = torch.float64 if self.qkv.weight.dtype == torch.float64 else torch.float32
        return torch.zeros(batch_size, self.memory_dim**2, device=device, dtype=dtype)

    def _decay_rate(self, controls: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.log_rate.to(controls.dtype)) * (
            2 * torch.sigmoid(controls[:, :self.memory_dim])
        )

    def _write_novelty(self, error: torch.Tensor) -> torch.Tensor:
        energy = error.square().mean(dim=-1, keepdim=True)
        return energy / (energy + F.softplus(self.log_threshold.to(error.dtype)) + 1e-6)

    def forward(self, x: torch.Tensor, state: torch.Tensor, dt: float = 1.0):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("innovation memory dt must be finite and >= 0")
        normalized = self.norm(x)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        controls = self.control(normalized)
        # Autocast would otherwise lower even explicitly FP32 bmm operands.
        # Accumulate the persistent memory in FP32 (FP64 for gradcheck).
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            controls = controls.to(dtype)
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
            v = torch.tanh(v)
            memory = state.to(dtype).reshape(x.size(0), self.memory_dim, self.memory_dim)
            rate = self._decay_rate(controls)
            decayed = memory * torch.exp(-dt * rate).unsqueeze(-1)
            read = torch.bmm(decayed, q.unsqueeze(-1)).squeeze(-1)
            error = v - torch.bmm(decayed, k.unsqueeze(-1)).squeeze(-1)
            novelty = self._write_novelty(error)
            write = -torch.expm1(-dt * F.softplus(controls[:, -2:-1]))
            updated = decayed + (write * novelty * error).unsqueeze(-1) * k.unsqueeze(-2)
            read = torch.sigmoid(controls[:, -1:]) * read
        return x + self.output(read.to(x.dtype)), updated.flatten(1)
