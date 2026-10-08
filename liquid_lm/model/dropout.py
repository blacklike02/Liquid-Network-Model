"""Dropout with a mask shared across time steps."""

from __future__ import annotations

import torch
import torch.nn as nn


class LockedDropout(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)
