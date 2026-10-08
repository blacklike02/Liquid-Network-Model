"""Optimizers (Muon, AdamW) and lr schedulers."""

from __future__ import annotations

import math

import torch

from ..model.utils import get_base_model


def _zeropower_via_newton_schulz5(G, steps):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            weight_decay, nesterov = group["weight_decay"], group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError("Muon supports only 2D parameters.")
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update_src = g.add(buf, alpha=momentum) if nesterov else buf
                u = _zeropower_via_newton_schulz5(update_src, steps=ns_steps)
                scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
                p.add_(u, alpha=-lr * scale)
        return loss


class MuonAdamWHybrid:
    """
    Behaves like torch.optim.Optimizer (param_groups/step/zero_grad/
    state_dict/load_state_dict) via duck typing, but does NOT inherit from
    torch.optim.Optimizer directly: it has no parameter list of its own
    (it is assembled from two REAL nested optimizers), and there is nothing
    to call Optimizer.__init__() with — without it LRScheduler would fail on
    the nonexistent _step_count. GradScaler and clip_grad_norm_ work with
    this object purely via duck typing.

    LambdaLR in newer PyTorch versions checks isinstance(optimizer, Optimizer),
    so the scheduler for this object is created separately for Muon and AdamW
    (see MultiScheduler/make_scheduler).
    """

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW):
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.muon.step()
        return self.adamw.step(closure)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


class MultiScheduler:
    """Several LambdaLR schedulers behind a single interface (step/state_dict/...)."""

    def __init__(self, schedulers):
        self.schedulers = list(schedulers)

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def get_last_lr(self):
        return [lr for sch in self.schedulers for lr in sch.get_last_lr()]

    def state_dict(self):
        return {"schedulers": [sch.state_dict() for sch in self.schedulers]}

    def load_state_dict(self, state):
        states = state.get("schedulers") if isinstance(state, dict) else None
        if not states or len(states) != len(self.schedulers):
            raise ValueError("scheduler state format was not accepted")
        for sch, st in zip(self.schedulers, states):
            sch.load_state_dict(st)


def make_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    total_steps = max(total_steps, 1)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step + 1, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    if isinstance(optimizer, MuonAdamWHybrid):
        return MultiScheduler([
            torch.optim.lr_scheduler.LambdaLR(optimizer.muon, lr_lambda),
            torch.optim.lr_scheduler.LambdaLR(optimizer.adamw, lr_lambda),
        ])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_optimizer(model, args, device):
    if args.optimizer == "muon":
        base = get_base_model(model)
        out_linear = base.readout[-1] if base.mlp_head else base.readout
        excluded_ids = {id(base.embedding.weight), id(out_linear.weight)}
        if out_linear.bias is not None:
            excluded_ids.add(id(out_linear.bias))
        muon_params, adamw_params = [], []
        for p in base.parameters():
            if not p.requires_grad:
                continue
            if p.ndim != 2 or id(p) in excluded_ids:
                adamw_params.append(p)
            else:
                muon_params.append(p)
        print(
            f"[OPT] Muon: {sum(p.numel() for p in muon_params):,} параметров "
            f"(hidden-матрицы) | AdamW: {sum(p.numel() for p in adamw_params):,}"
        )
        muon = Muon(muon_params, lr=args.muon_lr, momentum=args.muon_momentum,
                    weight_decay=args.weight_decay, ns_steps=args.muon_ns_steps)
        adamw = torch.optim.AdamW(adamw_params, lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
        return MuonAdamWHybrid(muon, adamw)
    kwargs = dict(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)
