"""Validation loss evaluation."""

from __future__ import annotations

import torch

from ..data.batching import get_batch
from ..model.chunk_memory import (
    chunk_memory_read,
    chunk_memory_reset,
    chunk_memory_write,
)
from ..model.utils import detach_hidden, get_base_model
from .amp import autocast_context
from .losses import masked_cross_entropy


@torch.inference_mode()
def evaluate(
    model, data, batch_size, seq_len, state_carry, device,
    split_start, split_end, batches, amp_dtype,
    train_mask=None, anchors=None, anchor_ratio=0.0,
):
    model.eval()
    base_model = get_base_model(model)
    gen = torch.Generator(device=data.device)
    gen.manual_seed(1234)
    loss_sum = torch.zeros((), device=data.device)
    token_sum = torch.zeros((), device=data.device)
    span = seq_len * state_carry
    for _ in range(batches):
        x_span, y_span = get_batch(
            data, batch_size, span, split_start, split_end, gen=gen,
            anchors=anchors, anchor_ratio=anchor_ratio, train_mask=train_mask,
        )
        hidden = None
        memory = chunk_memory_reset(base_model, x_span.shape[0], device)
        for k in range(state_carry):
            xk = x_span[:, k * seq_len : (k + 1) * seq_len]
            yk = y_span[:, k * seq_len : (k + 1) * seq_len]
            hidden = chunk_memory_read(base_model, hidden, memory)
            with autocast_context(device, amp_dtype):
                logits, hidden = model(xk, hidden=hidden)
                _loss, sum_loss, valid_count = masked_cross_entropy(
                    logits.reshape(-1, model.vocab_size), yk.reshape(-1),
                )
            loss_sum += sum_loss.detach().float()
            token_sum += valid_count.float()
            if k < state_carry - 1:
                memory = chunk_memory_write(base_model, hidden, memory)
                if memory is not None:
                    memory = memory.detach()
            hidden = detach_hidden(hidden)
    return (loss_sum / torch.clamp(token_sum, min=1.0)).item()
