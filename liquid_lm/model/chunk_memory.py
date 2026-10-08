"""Sliding memory on top of --state-carry and helpers for working with it."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChunkMemory(nn.Module):
    """
    Sliding memory on top of --state-carry: a separate, slower long-term
    memory channel that is not subject to the exp-decay dynamics of the
    liquid cell itself. The idea is "explicit persistent memory on top of the
    ODE state for long sequences" (inspired by M-CfC; the formula from the
    paper is NOT reproduced), mechanically closest to the Transformer-XL
    cache memory.

    At the end of each --state-carry subchunk, the final hidden state of the
    last liquid layer is pushed into a FIFO buffer (the last --memory-slots
    summaries; overflow evicts the oldest). At the start of the NEXT
    subchunk this memory is read via dot-product attention and mixed (through
    a learnable gate) into the last layer's hidden state that is carried
    across chunks.

    Between subchunks the hidden state and the buffer are detached (to save
    GPU memory) — there is no direct gradient across a chunk boundary. The
    read/write weights are learned from the effect within each individual
    subchunk.

    The buffer stores RAW (detached) final states, and summarize is applied
    lazily in read() — so summarize actually receives gradients. The memory
    only works with --state-carry >= 2.

    Wired into training and evaluate(); generate()/chat do NOT use this
    memory (a deliberate limitation).
    """

    def __init__(self, hidden_size: int, memory_slots: int) -> None:
        super().__init__()
        if memory_slots < 1:
            raise ValueError("--memory-slots must be >= 1")
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.summarize = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.init_slot = nn.Parameter(torch.zeros(hidden_size))
        self._scale = hidden_size ** 0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.summarize.weight)
        nn.init.zeros_(self.summarize.bias)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.read_gate.weight)
        nn.init.zeros_(self.read_gate.bias)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fresh buffer at the start of a new span — the analogue of initial_hidden().
        init_slot is NOT detached: it is a learnable parameter, and the gradient
        from read() in the first subchunk of the span must reach it."""
        return (
            self.init_slot
            .to(device=device)
            .view(1, 1, -1)
            .expand(batch_size, self.memory_slots, -1)
            .contiguous()
        )

    def read(self, h0: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mem = torch.tanh(self.summarize(memory))                       # (B,slots,H)
        q = self.query_proj(h0).unsqueeze(1)                           # (B,1,H)
        k = self.key_proj(mem)                                         # (B,slots,H)
        attn = torch.softmax((q * k).sum(dim=-1) / self._scale, dim=-1)  # (B,slots)
        retrieved = (attn.unsqueeze(-1) * mem).sum(dim=1)              # (B,H)
        gate = torch.sigmoid(self.read_gate(torch.cat([h0, retrieved], dim=-1)))
        return h0 + gate * retrieved

    def write(self, h_final: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        raw = h_final.detach().to(memory.dtype).unsqueeze(1)           # (B,1,H)
        return torch.cat([memory[:, 1:, :], raw], dim=1)               # FIFO shift


def chunk_memory_reset(base_model, batch_size: int, device: torch.device):
    """Fresh ChunkMemory buffer at the start of a new span — call it where hidden
    is reset. None if --memory-slots is disabled (0)."""
    if base_model.chunk_memory is None:
        return None
    return base_model.chunk_memory.init_memory(batch_size, device)


def chunk_memory_read(base_model, hidden, memory):
    """Mixes memory into the last layer's state BEFORE processing the next
    --state-carry subchunk. If hidden is not yet materialized (the very first
    subchunk of a span), takes the learnable h0 first — if chunk_memory is
    None, simply returns hidden as is."""
    if base_model.chunk_memory is None:
        return hidden
    if hidden is None:
        hidden = base_model.initial_hidden(memory.shape[0], memory.device)
    hidden = list(hidden)
    last = base_model.num_layers - 1
    hidden[last] = base_model.chunk_memory.read(hidden[last], memory)
    return hidden


def chunk_memory_write(base_model, hidden, memory):
    """Updates the buffer AFTER a subchunk is processed with the final hidden[-1]."""
    if base_model.chunk_memory is None:
        return memory
    return base_model.chunk_memory.write(hidden[base_model.num_layers - 1], memory)
