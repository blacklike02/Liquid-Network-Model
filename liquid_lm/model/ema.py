"""Exponential moving average of weights."""

from __future__ import annotations

import torch

from .utils import get_base_model


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            k: v.detach().clone()
            for k, v in get_base_model(model).state_dict().items()
            if v.dtype.is_floating_point
        }

    @classmethod
    def from_state(cls, state, decay):
        obj = cls.__new__(cls)
        obj.decay = float(decay)
        obj.updates = int(state.get("updates", 0))
        obj.shadow = state["shadow"]
        return obj

    def state_dict(self):
        return {"updates": self.updates, "shadow": self.shadow}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = get_base_model(model).state_dict()
        keys = list(self.shadow.keys())
        shadow = [self.shadow[k] for k in keys]
        params = [msd[k] for k in keys]
        torch._foreach_mul_(shadow, d)
        torch._foreach_add_(shadow, params, alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model):
        msd = get_base_model(model).state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)
