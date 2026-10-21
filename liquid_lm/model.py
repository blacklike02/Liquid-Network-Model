"""The liquid language model, architecture config and checkpoint I/O."""

from __future__ import annotations

import math
import os
import tempfile
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .associative import InnovationMemory
from .cuda_graphs import CudaGraphDecode, CudaGraphUnavailable, decode_signature, serialize_graph_generation
from .layers import (
    ChunkMemory,
    CompactChunkMemory,
    validate_retired_architecture,
    LiquidCell,
    LockedDropout,
    ResidualFFN,
    _TRITON_AVAILABLE,
    chunk_memory_read,
    chunk_memory_reset,
    chunk_memory_write,
    make_norm,
)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def get_base_model(model):
    return getattr(model, "_orig_mod", model)


def detach_hidden(hidden):
    return [h.detach() for h in hidden]


def clone_float_state(model):
    return {
        k: v.detach().clone()
        for k, v in get_base_model(model).state_dict().items()
        if v.dtype.is_floating_point
    }


def restore_float_state(model, backup):
    msd = get_base_model(model).state_dict()
    for k, v in backup.items():
        msd[k].copy_(v)


def validate_generation_options(
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
) -> None:
    """Fail early on invalid generation settings instead of silently ignoring them."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not math.isfinite(float(top_p)) or not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be finite and between 0 and 1")
    if not math.isfinite(float(repetition_penalty)) or repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be finite and > 0")
    if repetition_window < 0:
        raise ValueError("repetition_window must be >= 0")


def _trim_recent_tokens(tokens: list[int], window: int) -> list[int]:
    """A zero window disables repetition tracking; negative windows are invalid."""
    if window < 0:
        raise ValueError("repetition_window must be >= 0")
    return list(tokens[-window:]) if window > 0 else []


# ---------------------------------------------------------------------------
# Liquid language model
# ---------------------------------------------------------------------------


class LiquidLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 384,
        hidden_size: int = 384,
        num_layers: int = 2,
        ode_unfolds: int = 1,
        dropout: float = 0.05,
        zoneout: float = 0.1,
        tie_weights: bool = True,
        mlp_head: bool = False,
        residual: bool = True,
        multi_tau: bool = True,
        local_conv: bool = True,
        use_checkpoint: bool = False,
        gate_mode: str = "ltc",
        ode_solver: str = "euler",
        wiring: str = "dense",
        memory_slots: int = 0,
        ffn_mult: float = 0.0,
        norm_type: str = "rms",
        innovation_memory_dim: int = 0,
        memory_dim: int = 0,
        recurrent_passes: int = 1,
    ) -> None:
        super().__init__()
        validate_retired_architecture(ode_unfolds, ode_solver, wiring)
        if memory_dim < 0 or memory_slots < 0:
            raise ValueError("memory_dim and memory_slots must be >= 0")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if isinstance(recurrent_passes, bool) or not isinstance(recurrent_passes, int) or recurrent_passes < 1:
            raise ValueError("recurrent_passes must be a positive integer")
        if recurrent_passes > 1 and embedding_dim != hidden_size:
            raise ValueError("recurrent_passes > 1 requires embedding_dim == hidden_size")
        if ffn_mult < 0:
            raise ValueError("ffn_mult must be >= 0 (0 disables the FFN blocks)")
        if innovation_memory_dim < 0:
            raise ValueError("innovation_memory_dim must be >= 0")
        if tie_weights and embedding_dim != hidden_size:
            raise ValueError(
                "tie_weights=True requires embedding_dim == hidden_size "
                f"(сейчас {embedding_dim} != {hidden_size})."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_passes = recurrent_passes
        self.recurrent_state_count = num_layers * recurrent_passes
        self.dropout_rate = dropout
        self.zoneout = zoneout
        self.tie_weights = tie_weights
        self.mlp_head = mlp_head
        self.residual = residual
        self.multi_tau = multi_tau
        self.local_conv = local_conv
        self.use_checkpoint = use_checkpoint
        self.gate_mode = gate_mode
        self.memory_slots = memory_slots
        self.ffn_mult = float(ffn_mult)
        self.norm_type = norm_type
        self.innovation_memory_dim = innovation_memory_dim
        self.memory_dim = min(memory_dim, hidden_size) if memory_dim > 0 else 0
        self.chunk_memory = None
        if memory_slots > 0:
            self.chunk_memory = (CompactChunkMemory(hidden_size, memory_slots, self.memory_dim)
                                 if self.memory_dim else ChunkMemory(hidden_size, memory_slots))

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emb_dropout = LockedDropout(dropout)

        if local_conv:
            # CAUSAL convolution: padding on the left only
            # (kernel_size-1 zeros/buffer), 0 on the right. Symmetric padding
            # in Conv1d sees FUTURE tokens at position t — a direct leak
            # of the answer into train/val loss (see forward()).
            self._conv_kernel_size = 5
            self._conv_left_pad = self._conv_kernel_size - 1
            self.local_conv_layer = nn.Conv1d(
                embedding_dim, embedding_dim,
                kernel_size=self._conv_kernel_size,
                groups=embedding_dim, padding=0,
            )
        else:
            self.local_conv_layer = None

        cells = []
        for layer in range(num_layers):
            input_size = embedding_dim if layer == 0 else hidden_size
            cells.append(
                LiquidCell(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    zoneout=zoneout,
                    multi_tau=multi_tau,
                    gate_mode=gate_mode,
                    norm_type=norm_type,
                )
            )
        self.cells = nn.ModuleList(cells)

        if ffn_mult > 0:
            self.ffns = nn.ModuleList(
                [
                    ResidualFFN(
                        hidden_size, ffn_mult, norm_type=norm_type,
                        dropout=dropout, num_layers=num_layers,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.ffns = None

        self.h0 = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.final_norm = make_norm(norm_type, hidden_size)
        if mlp_head:
            self.readout = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, vocab_size),
            )
        else:
            self.readout = nn.Linear(hidden_size, vocab_size)

        if residual:
            self._do_residual = tuple(
                [embedding_dim == hidden_size] + [True] * (num_layers - 1)
            )
        else:
            self._do_residual = tuple([False] * num_layers)

        self._reset_parameters()
        # Construct after the legacy modules so a disabled feature also preserves
        # initialization/RNG behavior and old state_dict keys exactly.
        self.innovation_memory = (
            InnovationMemory(hidden_size, innovation_memory_dim)
            if innovation_memory_dim > 0 else None
        )
        # Store only an optional compiled override. A bound method here creates
        # a self-cycle and delays releasing model tensors until cyclic GC runs.
        self._step_fn = None
        self._sequence_batching = False
        self._kernel = "torch"
        self._init_cuda_graph_runtime()

    def _init_cuda_graph_runtime(self, enabled=False):
        self._cuda_graphs_enabled = bool(enabled)
        self._cuda_graph_runner = None
        self._cuda_graph_signature = None
        self._cuda_graph_failed_signature = None
        self._cuda_graph_last_error = None
        self._cuda_graph_lock = threading.RLock()
        self._cuda_graph_captures = 0
        self._cuda_graph_replays = 0
        self._cuda_graph_fallbacks = 0

    def __getstate__(self):
        state = super().__getstate__()
        # Captured graphs/locks are runtime resources, not portable model state.
        state = {**state, "_cuda_graph_runner": None, "_cuda_graph_signature": None,
                 "_cuda_graph_failed_signature": None, "_cuda_graph_lock": None}
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._init_cuda_graph_runtime(state.get("_cuda_graphs_enabled", False))

    def clear_cuda_graph_cache(self):
        """Release captures; also call after untracked parameter .data edits."""
        with self._cuda_graph_lock:
            self._cuda_graph_runner = None
            self._cuda_graph_signature = None
            self._cuda_graph_failed_signature = None
            self._cuda_graph_last_error = None

    def set_cuda_graphs(self, enabled: bool):
        """Opt into cached whole-token CUDA inference; never used by training."""
        with self._cuda_graph_lock:
            if enabled and self._step_fn is not None:
                raise ValueError("CUDA Graphs and a compiled step are separate inference backends")
            self.clear_cuda_graph_cache()
            self._cuda_graphs_enabled = bool(enabled)

    def cuda_graph_stats(self):
        return dict(enabled=self._cuda_graphs_enabled, captures=self._cuda_graph_captures,
                    replays=self._cuda_graph_replays, fallbacks=self._cuda_graph_fallbacks,
                    cached_graphs=self._cuda_graph_runner.capture_count if self._cuda_graph_runner else 0,
                    last_error=self._cuda_graph_last_error)

    def _cuda_graph_fallback(self, reason, signature):
        if reason != self._cuda_graph_last_error:
            warnings.warn(f"CUDA Graphs unavailable; using eager decode: {reason}", RuntimeWarning, stacklevel=3)
        self._cuda_graph_runner = None
        self._cuda_graph_signature = None
        self._cuda_graph_failed_signature = signature
        self._cuda_graph_last_error = reason
        self._cuda_graph_fallbacks += 1

    def _get_cuda_graph_runner(self, hidden, memory):
        if hidden[0].device.type != "cuda":
            self._cuda_graph_fallback("CUDA device required", None)
            return None
        if self._step_fn is not None:
            self._cuda_graph_fallback("compiled step is active", None)
            return None
        generator = torch.cuda.default_generators[hidden[0].device.index]
        if not all(hasattr(generator, name) for name in ("graphsafe_get_state", "graphsafe_set_state", "clone_state")):
            self._cuda_graph_fallback("this PyTorch build lacks graph-safe RNG state management", None)
            return None
        hooks = torch.nn.modules.module
        if (any(module._forward_hooks or module._forward_pre_hooks for module in self.modules())
                or getattr(hooks, "_global_forward_hooks", None) or getattr(hooks, "_global_forward_pre_hooks", None)):
            self._cuda_graph_fallback("Python forward hooks are active", None)
            return None
        signature = decode_signature(self, hidden, memory)
        if signature == self._cuda_graph_failed_signature:
            self._cuda_graph_fallbacks += 1
            return None
        try:
            if signature != self._cuda_graph_signature:
                self.clear_cuda_graph_cache()
                self._cuda_graph_runner = CudaGraphDecode(self, hidden, memory)
                self._cuda_graph_signature = signature
            else:
                self._cuda_graph_runner.reset(hidden, memory)
            return self._cuda_graph_runner
        except (RuntimeError, NotImplementedError) as exc:
            self._cuda_graph_fallback(str(exc), signature)
            return None

    def train(self, mode=True):
        lock = getattr(self, "_cuda_graph_lock", None)
        with lock if lock is not None else nullcontext():
            if mode and lock is not None:
                self.clear_cuda_graph_cache()
            return super().train(mode)

    def _apply(self, fn, recurse=True):
        lock = getattr(self, "_cuda_graph_lock", None)
        with lock if lock is not None else nullcontext():
            if lock is not None:
                self.clear_cuda_graph_cache()
            return super()._apply(fn, recurse=recurse)

    def set_sequence_batching(self, enabled: bool) -> None:
        """Opt into batching input projections/FFNs over time between layers.

        This is an execution choice, not an architecture/checkpoint change.
        Stochastic training and compiled-step execution retain the step schedule.
        """
        self._sequence_batching = bool(enabled)
        self.clear_cuda_graph_cache()

    def _can_batch_sequence(self, seq_len: int) -> bool:
        if not self._sequence_batching or seq_len <= 1:
            return False
        if self._step_fn is not None and self._kernel != "triton":
            return False
        if self.training:
            if any(cell.dropout.p > 0 or cell.zoneout > 0 for cell in self.cells):
                return False
            if self.ffns is not None and any(ffn.dropout.p > 0 for ffn in self.ffns):
                return False
        return True

    def set_kernel(self, kernel: str) -> str:
        """Select torch or experimental Triton backend for the LTC substep.

        Triton supports inference and first-order training backward for LTC + Euler.
        If unavailable, prints the reason and keeps the PyTorch backend.
        Returns the backend actually selected.
        """
        if kernel not in ("torch", "triton"):
            raise ValueError(f"Неизвестный kernel: {kernel!r} (ожидался 'torch' или 'triton')")
        actual = kernel
        if kernel == "triton":
            reason = None
            if not _TRITON_AVAILABLE:
                reason = "the triton package is not installed"
            elif not torch.cuda.is_available():
                reason = "CUDA is unavailable"
            elif not next(self.parameters()).is_cuda:
                reason = "the model is not on CUDA"
            elif self.gate_mode != "ltc":
                reason = "only --gate-mode ltc is supported"
            if reason is not None:
                print(f"[KERNEL] triton недоступен ({reason}) — используется torch.")
                actual = "torch"
        for cell in self.cells:
            cell.kernel = actual
        self._kernel = actual
        self.clear_cuda_graph_cache()
        return actual

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        if self.local_conv_layer is not None:
            nn.init.kaiming_normal_(
                self.local_conv_layer.weight, mode="fan_out", nonlinearity="relu"
            )
            if self.local_conv_layer.bias is not None:
                nn.init.zeros_(self.local_conv_layer.bias)

        out_linear = self.readout[-1] if self.mlp_head else self.readout
        if self.tie_weights:
            out_linear.weight = self.embedding.weight
        else:
            nn.init.xavier_uniform_(out_linear.weight)
        if out_linear.bias is not None:
            nn.init.zeros_(out_linear.bias)
        if self.mlp_head:
            nn.init.xavier_uniform_(self.readout[0].weight)
            nn.init.zeros_(self.readout[0].bias)

    def initial_hidden(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        hidden = [
            self.h0[i].unsqueeze(0).expand(batch_size, -1).contiguous()
            for _ in range(self.recurrent_passes) for i in range(self.num_layers)
        ]
        if self.recurrent_passes > 1:
            # Pass-major layout: each depth pass has its own temporal state,
            # while cell weights and learned initial states are shared.
            hidden = [state.clone() for state in hidden]
        if self.innovation_memory is not None:
            hidden.append(self.innovation_memory.initial_state(batch_size, device))
        return hidden

    def enable_compiled_step(
        self, mode: str = "default", batch_size: int = 2, inference: bool = False,
    ) -> tuple[bool, str | None]:
        """Compiles the recurrent step. inference=False — warm-up in train mode
        with backward (for training); inference=True — warm-up in eval mode under
        inference_mode (for generation/chat: otherwise the very first generate()
        would trigger a recompilation due to the change of training flag and
        gradient mode)."""
        if not hasattr(torch, "compile"):
            return False, None
        if self._cuda_graphs_enabled:
            raise ValueError("Disable CUDA Graphs before compiling the recurrent step")
        device = next(self.parameters()).device
        b = max(1, int(batch_size))
        x_t = torch.zeros(b, self.embedding_dim, device=device)
        xp_dim = self.hidden_size * 2 if self.gate_mode == "cfc" else self.hidden_size
        xp_t = torch.zeros(b, xp_dim, device=device)
        tx_t = torch.zeros(b, xp_dim, device=device)
        was_training = self.training
        last_error: str | None = None
        try:
            if inference:
                self.eval()
            else:
                self.train()
            modes = [mode] if mode == "default" else [mode, "default"]
            for try_mode in modes:
                try:
                    compiled = torch.compile(
                        self._step_impl, dynamic=False, mode=try_mode
                    )
                    for _ in range(3):
                        hidden = self.initial_hidden(b, device)
                        if inference:
                            with torch.inference_mode():
                                fused = [cell.fused_h_weights() for cell in self.cells]
                                compiled(x_t, xp_t, tx_t, hidden, 1.0, fused, self._tau_mix_weights(),
                                         [cell.input_projection_weights() for cell in self.cells])
                        else:
                            # fused is recomputed on every iteration: the
                            # torch.cat graph must not be reused after backward.
                            fused = [cell.fused_h_weights() for cell in self.cells]
                            out, _ = compiled(x_t, xp_t, tx_t, hidden, 1.0, fused, self._tau_mix_weights(),
                                              [cell.input_projection_weights() for cell in self.cells])
                            self.final_norm(out).sum().backward()
                    if not inference:
                        self.zero_grad(set_to_none=True)
                    self._step_fn = compiled
                    return True, try_mode
                except Exception as exc:
                    self._step_fn = None
                    last_error = f"[{try_mode}] {type(exc).__name__}: {exc}"
                    continue
            if last_error:
                print(f"[COMPILE] Причина: {last_error}")
            return False, None
        finally:
            if was_training:
                self.train()
            else:
                self.eval()

    def _tau_mix_weights(self):
        # Parameters are constant within one forward graph. Sharing this node
        # retains autograd while avoiding a tiny softmax for every token/substep.
        return [F.softmax(cell.tau_mix, dim=0) if cell.multi_tau else None for cell in self.cells]

    def _step_impl(self, x_t, xp_t, tx_t, hidden, dt, fused, mix_weights=None, input_projections=None):
        hidden = list(hidden)
        do_residual = self._do_residual
        use_ckpt = self.use_checkpoint and self.training

        for state_index in range(self.recurrent_state_count):
            i = state_index % self.num_layers
            cell = self.cells[i]
            w, b = fused[i]
            if use_ckpt:
                h_new = cp.checkpoint(
                    cell,
                    hidden[state_index], x_t,
                    dt=dt,
                    xp=xp_t if state_index == 0 else None,
                    tx=tx_t if state_index == 0 else None,
                    fused_w=w,
                    fused_b=b,
                    mix_w=mix_weights[i] if mix_weights is not None else None,
                    input_projection=input_projections[i] if input_projections is not None else None,
                    use_reentrant=False,
                )
            else:
                if state_index == 0:
                    h_new = cell(hidden[state_index], x_t, dt=dt, xp=xp_t, tx=tx_t, fused_w=w, fused_b=b,
                                 mix_w=mix_weights[i] if mix_weights is not None else None,
                                 input_projection=input_projections[i] if input_projections is not None else None)
                else:
                    h_new = cell(hidden[state_index], x_t, dt=dt, fused_w=w, fused_b=b,
                                 mix_w=mix_weights[i] if mix_weights is not None else None,
                                 input_projection=input_projections[i] if input_projections is not None else None)

            hidden[state_index] = h_new
            if do_residual[i]:
                x_t = x_t + h_new
            else:
                x_t = h_new
            # Repeat the liquid mixers; execute each FFN on the final pass.
            if self.ffns is not None and state_index >= self.recurrent_state_count - self.num_layers:
                x_t = self.ffns[i](x_t)

        if self.innovation_memory is not None:
            x_t, hidden[self.recurrent_state_count] = self.innovation_memory(
                x_t, hidden[self.recurrent_state_count], dt=dt,
            )
        # Final normalization has no recurrent-state dependency. Batch it after
        # stacking time, or normalize only the final token during prompt prefill.
        return x_t, hidden

    def forward(
        self,
        tokens: torch.Tensor,
        hidden: list[torch.Tensor] | None = None,
        dt: float = 1.0,
        fused: list | None = None,
        last_token_only: bool = False,
    ):
        batch_size, seq_len = tokens.shape
        conv_buf = None
        if hidden is None:
            hidden = self.initial_hidden(batch_size, tokens.device)
        else:
            hidden = list(hidden)
            state_count = self.recurrent_state_count + int(self.innovation_memory is not None)
            if len(hidden) > state_count:
                conv_buf = hidden[state_count]
                hidden = hidden[:state_count]
        new_conv_buf = None

        x = self.emb_dropout(self.embedding(tokens))

        if self.local_conv_layer is not None:
            if conv_buf is None:
                conv_buf = x.new_zeros(batch_size, self._conv_left_pad, x.size(-1))
            x_ext = torch.cat([conv_buf.to(x.dtype), x], dim=1)
            # A detached view would still own the complete chunk's storage.
            # Clone only the small carried tail; preserve its autograd edge.
            new_conv_buf = x_ext[:, -self._conv_left_pad:, :].clone()
            x_conv = self.local_conv_layer(x_ext.transpose(1, 2)).transpose(1, 2)
            x = x + x_conv

        if fused is None:
            fused = [cell.fused_h_weights() for cell in self.cells]
        mix_weights = self._tau_mix_weights()
        input_projections = [cell.input_projection_weights() for cell in self.cells]

        if self._can_batch_sequence(seq_len):
            features, hidden = self._forward_sequence_layers(
                x, hidden, dt, fused, mix_weights, input_projections, last_token_only,
            )
        else:
            cell0 = self.cells[0]
            xp_seq, tx_seq = cell0.precompute_x(x, input_projections[0])
            outputs = []
            # Compiled steps have no Triton branch.
            step_fn = self._step_impl if self._kernel == "triton" else (self._step_fn or self._step_impl)
            for t in range(seq_len):
                out_t, hidden = step_fn(
                    x[:, t, :], xp_seq[:, t, :], tx_seq[:, t, :], hidden, dt, fused, mix_weights, input_projections
                )
                if not last_token_only:
                    outputs.append(out_t)
            features = out_t.unsqueeze(1) if last_token_only else torch.stack(outputs, dim=1)

        # Generation needs the final prediction only. Keep all recurrent
        # updates, but avoid projecting the whole prompt into vocabulary space.
        logits = self.readout(self.final_norm(features))
        if new_conv_buf is not None:
            hidden = list(hidden) + [new_conv_buf]
        return logits, hidden

    def _forward_sequence_layers(self, x, hidden, dt, fused, mixes, projections, last_token_only):
        """Finish each layer before batching the next layer's independent work.

        Recurrent states remain sequential. Residuals/FFNs only feed the next
        layer, so their sequence dimension can be evaluated in parallel.
        """
        hidden = list(hidden)
        length = x.size(1)
        for index in range(self.recurrent_state_count):
            layer = index % self.num_layers
            cell = self.cells[layer]
            xp, tx = cell.precompute_x(x, projections[layer])
            state = hidden[index]
            outputs = []
            keep_sequence = index + 1 < self.recurrent_state_count or not last_token_only or self.innovation_memory is not None
            for t in range(length):
                if self.use_checkpoint and self.training:
                    state = cp.checkpoint(
                        cell, state, x[:, t], dt=dt, xp=xp[:, t], tx=tx[:, t],
                        fused_w=fused[layer][0], fused_b=fused[layer][1], mix_w=mixes[layer],
                        input_projection=projections[layer], use_reentrant=False,
                    )
                else:
                    state = cell(
                        state, x[:, t], dt=dt, xp=xp[:, t], tx=tx[:, t],
                        fused_w=fused[layer][0], fused_b=fused[layer][1], mix_w=mixes[layer],
                        input_projection=projections[layer],
                    )
                if keep_sequence:
                    outputs.append(state)
            hidden[index] = state
            output = torch.stack(outputs, dim=1) if keep_sequence else state.unsqueeze(1)
            if self._do_residual[layer]:
                output = output + (x if keep_sequence else x[:, -1:])
            if self.ffns is not None and index >= self.recurrent_state_count - self.num_layers:
                output = self.ffns[layer](output)
            x = output
        if self.innovation_memory is not None:
            outputs = []
            for t in range(length):
                output, hidden[self.recurrent_state_count] = self.innovation_memory(x[:, t], hidden[self.recurrent_state_count], dt=dt)
                if not last_token_only:
                    outputs.append(output)
            x = output.unsqueeze(1) if last_token_only else torch.stack(outputs, dim=1)
        return x, hidden

    def _forward_with_chunk_memory(self, tokens, hidden, memory, fused, chunk_size):
        """Process a prompt in chunks, consulting and updating ChunkMemory."""
        logits_all = None
        processed = 0
        for start in range(0, tokens.size(1), chunk_size):
            hidden = chunk_memory_read(self, hidden, memory)
            chunk = tokens[:, start:start + chunk_size]
            logits_all, hidden = self.forward(chunk, hidden=hidden, fused=fused, last_token_only=True)
            processed += chunk.size(1)
            if processed % chunk_size == 0:
                memory = chunk_memory_write(self, hidden, memory)
        if logits_all is None:
            raise ValueError("Cannot run generation on an empty token sequence.")
        return logits_all, hidden, memory, processed

    @staticmethod
    def _apply_repetition_penalty(logits, recent_tokens, penalty):
        if penalty <= 1.0 or not recent_tokens:
            return logits
        logits = logits.clone()
        uniq = torch.tensor(
            sorted(set(recent_tokens)), dtype=torch.long, device=logits.device
        )
        vals = logits.index_select(1, uniq)
        logits.index_copy_(
            1, uniq, torch.where(vals < 0, vals * penalty, vals / penalty)
        )
        return logits

    @staticmethod
    def _top_k_top_p_filter(logits, top_k=0, top_p=1.0):
        logits = logits.clone()
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, k=k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
            sorted_remove[:, 0] = False
            remove_mask = torch.zeros_like(logits, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove_mask, float("-inf"))
        return logits

    @torch.inference_mode()
    @serialize_graph_generation
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 500,
        temperature: float = 0.85,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        repetition_window: int = 128,
        device: torch.device | str = "cpu",
        stop_token_ids: list[int] | None = None,
        only_new: bool = False,
        memory_chunk_size: int = 128,
    ) -> str:
        validate_generation_options(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        self.eval()
        if not prompt:
            prompt = " "
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = tokenizer.encode(" ")
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        stop_ids = set(stop_token_ids or ())

        fused = [cell.fused_h_weights() for cell in self.cells]
        use_chunk_memory = self.chunk_memory is not None
        memory = None
        if use_chunk_memory:
            if memory_chunk_size < 1:
                raise ValueError("memory_chunk_size must be >= 1 when ChunkMemory is enabled")
            memory = chunk_memory_reset(self, tokens.size(0), tokens.device)
            logits_all, hidden, memory, processed_tokens = self._forward_with_chunk_memory(
                tokens, None, memory, fused, int(memory_chunk_size),
            )
        else:
            logits_all, hidden = self.forward(tokens, fused=fused, last_token_only=True)
            processed_tokens = tokens.size(1)
        next_logits = logits_all[:, -1, :].float()

        generated_ids = list(prompt_ids)
        recent_tokens = _trim_recent_tokens(generated_ids, repetition_window)
        temperature = max(float(temperature), 1e-4)
        runner = None
        graph_attempted = False

        for step_i in range(max_new_tokens):
            logits = self._apply_repetition_penalty(
                next_logits, recent_tokens, repetition_penalty
            )
            logits = logits / temperature
            logits = self._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            if token_id in stop_ids:
                break
            generated_ids.append(token_id)
            recent_tokens = _trim_recent_tokens(recent_tokens + [token_id], repetition_window)
            if step_i + 1 < max_new_tokens:
                if self._cuda_graphs_enabled and not graph_attempted:
                    runner = self._get_cuda_graph_runner(hidden, memory)
                    graph_attempted = True
                if runner is not None:
                    previous_captures = runner.capture_count
                    try:
                        next_logits = runner.step(
                            next_token,
                            read_memory=use_chunk_memory and processed_tokens % memory_chunk_size == 0,
                            write_memory=use_chunk_memory and (processed_tokens + 1) % memory_chunk_size == 0,
                        )
                    except CudaGraphUnavailable as exc:
                        hidden, memory = runner.hidden, runner.memory
                        self._cuda_graph_fallback(str(exc), self._cuda_graph_signature)
                        runner = None
                    else:
                        self._cuda_graph_captures += runner.capture_count - previous_captures
                        self._cuda_graph_replays += 1
                        processed_tokens += 1
                        continue
                if use_chunk_memory:
                    if processed_tokens % memory_chunk_size == 0:
                        hidden = chunk_memory_read(self, hidden, memory)
                    logits_all, hidden = self.forward(next_token, hidden=hidden, fused=fused)
                    processed_tokens += 1
                    if processed_tokens % memory_chunk_size == 0:
                        memory = chunk_memory_write(self, hidden, memory)
                else:
                    logits_all, hidden = self.forward(next_token, hidden=hidden, fused=fused)
                next_logits = logits_all[:, -1, :].float()

        if only_new:
            return tokenizer.decode(generated_ids[len(prompt_ids):])
        return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# Architecture config
# ---------------------------------------------------------------------------


ARCH_KEYS = (
    "embedding_dim", "hidden_size", "num_layers",
    "dropout", "zoneout", "tie_weights", "mlp_head", "residual",
    "multi_tau", "local_conv", "gate_mode",
    "memory_slots", "memory_dim",
    "ffn_mult", "norm_type", "innovation_memory_dim", "recurrent_passes",
)


LEGACY_ARCH_DEFAULTS = {
    "zoneout": 0.0, "tie_weights": False, "mlp_head": False,
    "residual": False, "multi_tau": False, "local_conv": False,
    "gate_mode": "ltc",
    "memory_slots": 0, "memory_dim": 0,
    "ffn_mult": 0.0, "norm_type": "layer", "innovation_memory_dim": 0, "recurrent_passes": 1,
}


def _arch_value(config, key):
    if key in config:
        return config[key]
    return LEGACY_ARCH_DEFAULTS.get(key)


def validate_retired_config(config):
    validate_retired_architecture(config.get("ode_unfolds", 1),
                                  config.get("ode_solver", "euler"),
                                  config.get("wiring", "dense"))
    # Missing metadata historically meant LayerNorm, never RMSNorm.
    if config.get("norm_type") != "rms":
        raise ValueError(
            "Checkpoint normalization is not explicitly RMSNorm (norm_type='rms'). "
            "LayerNorm checkpoints, including older checkpoints without norm_type, "
            "are retired. Load them with an earlier revision or train a new RMSNorm model; "
            "changing the metadata does not convert trained weights."
        )


def create_model_from_config(vocab_size, config, device):
    validate_retired_config(config)
    return LiquidLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=int(config["embedding_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config.get("dropout", 0.05)),
        zoneout=float(_arch_value(config, "zoneout") or 0.0),
        tie_weights=bool(_arch_value(config, "tie_weights")),
        mlp_head=bool(_arch_value(config, "mlp_head")),
        residual=bool(_arch_value(config, "residual")),
        multi_tau=bool(_arch_value(config, "multi_tau")),
        local_conv=bool(_arch_value(config, "local_conv")),
        use_checkpoint=False,
        gate_mode=str(_arch_value(config, "gate_mode") or "ltc"),
        memory_slots=int(_arch_value(config, "memory_slots") or 0),
        memory_dim=int(_arch_value(config, "memory_dim") or 0),
        ffn_mult=float(_arch_value(config, "ffn_mult") or 0.0),
        norm_type="rms",
        innovation_memory_dim=int(_arch_value(config, "innovation_memory_dim") or 0),
        recurrent_passes=_arch_value(config, "recurrent_passes"),
    ).to(device)


def load_model_state_compat(model, state):
    if isinstance(get_base_model(model), LiquidLanguageModel) and any(
        key.endswith("norm.bias") for key in state
    ):
        raise RuntimeError("LayerNorm state cannot be loaded into an RMSNorm model. "
                           "Use an earlier revision for LayerNorm checkpoints.")
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Неожиданные ключи в checkpoint: {unexpected}")
    allowed_missing = {f"h0.{i}" for i in range(model.num_layers)}
    if model.multi_tau:
        allowed_missing.add("tau_mix")
    if model.local_conv_layer is not None:
        allowed_missing.add("local_conv_layer.weight")
        allowed_missing.add("local_conv_layer.bias")
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"Checkpoint is missing weights: {bad_missing}")
    print("[CHECKPOINT] Compatible weights loaded (new features initialized).")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    path, model, optimizer, scheduler, scaler, epoch, global_step,
    train_loss, val_loss, best_val, tokenizer, args,
    ema_state=None, corpus_sha256=None,
):
    base_model = get_base_model(model)
    payload = {
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "ema_state": ema_state,
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "tokenizer": tokenizer.to_config(),
        "corpus_sha256": corpus_sha256,
        "config": {**vars(args), "norm_type": base_model.norm_type,
                   "recurrent_passes": base_model.recurrent_passes}
                  if isinstance(base_model, LiquidLanguageModel) else dict(vars(args)),
    }
    # Replace only after a complete write, so an interrupted/failed save keeps
    # the previous checkpoint intact. Use the same directory/filesystem.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(payload, stream)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


_TRUST_CHECKPOINTS = False


def set_trust_checkpoints(value: bool) -> None:
    """Allow unsafe checkpoint loading (weights_only=False) — the --trust-checkpoint flag."""
    global _TRUST_CHECKPOINTS
    _TRUST_CHECKPOINTS = bool(value)


def load_checkpoint_raw(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        if not _TRUST_CHECKPOINTS:
            raise RuntimeError(
                f"Не удалось безопасно загрузить {path} (weights_only=True): {exc}\n"
                "If the checkpoint is yours and trusted, add --trust-checkpoint "
                "(pickle can execute arbitrary code)."
            ) from exc
        print("[CHECKPOINT] WARNING: unsafe loading (--trust-checkpoint).")
        return torch.load(path, map_location=device, weights_only=False)


def load_inference_checkpoint(path):
    """Load on CPU and discard training-only tensors before building a model."""
    payload = load_checkpoint_raw(path, torch.device("cpu"))
    for key in ("optimizer_state", "scheduler_state", "scaler_state", "ema_state"):
        payload.pop(key, None)
    return payload


def print_checkpoint_info(checkpoint):
    tokenizer_config = checkpoint.get("tokenizer", {})
    kind = tokenizer_config.get("kind", "char" if "char_to_idx" in checkpoint else "?")
    train_loss = checkpoint.get("train_loss")
    val_loss = checkpoint.get("val_loss")
    print("[CHECKPOINT]")
    print(f"  epoch       : {checkpoint.get('epoch')}")
    print(f"  global_step : {checkpoint.get('global_step')}")
    print(f"  train_loss  : {train_loss if train_loss is None else round(float(train_loss), 6)}")
    print(f"  val_loss    : {val_loss if val_loss is None else round(float(val_loss), 6)}")
    print(f"  tokenizer   : {kind}")
    print(f"  config      : {checkpoint.get('config', {})}")
