"""Mixed precision: dtype selection, GradScaler, autocast."""

from __future__ import annotations

import torch


def resolve_amp_dtype(device, requested):
    if requested == "off" or device.type != "cuda":
        return None
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_grad_scaler(enabled, device):
    if not enabled or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device, amp_dtype):
    if amp_dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
