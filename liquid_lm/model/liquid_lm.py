"""Language model built on liquid cells (training, generation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .chunk_memory import ChunkMemory
from .dropout import LockedDropout
from .liquid_cell import LiquidCell
from .triton_kernel import _TRITON_AVAILABLE


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
        ncp_command_frac: float = 0.4,
        ncp_motor_frac: float = 0.2,
        memory_slots: int = 0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if tie_weights and embedding_dim != hidden_size:
            raise ValueError(
                "tie_weights=True requires embedding_dim == hidden_size "
                f"(сейчас {embedding_dim} != {hidden_size})."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ode_unfolds = ode_unfolds
        self.dropout_rate = dropout
        self.zoneout = zoneout
        self.tie_weights = tie_weights
        self.mlp_head = mlp_head
        self.residual = residual
        self.multi_tau = multi_tau
        self.local_conv = local_conv
        self.use_checkpoint = use_checkpoint
        self.gate_mode = gate_mode
        self.ode_solver = ode_solver
        self.wiring = wiring
        self.ncp_command_frac = ncp_command_frac
        self.ncp_motor_frac = ncp_motor_frac
        self.memory_slots = memory_slots
        self.chunk_memory = ChunkMemory(hidden_size, memory_slots) if memory_slots > 0 else None

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
                    ode_unfolds=ode_unfolds,
                    dropout=dropout,
                    zoneout=zoneout,
                    multi_tau=multi_tau,
                    gate_mode=gate_mode,
                    ode_solver=ode_solver,
                    wiring=wiring,
                    ncp_command_frac=ncp_command_frac,
                    ncp_motor_frac=ncp_motor_frac,
                )
            )
        self.cells = nn.ModuleList(cells)

        self.h0 = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(hidden_size)
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
        self._step_fn = self._step_impl
        self._kernel = "torch"

    def set_kernel(self, kernel: str) -> str:
        """Select the backend for the tail of the ltc substep: "torch" (default) or
        "triton" (inference only, see _ltc_tail_kernel). If triton cannot be
        enabled, prints the reason and stays on torch without failing.
        Returns the backend actually selected."""
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
            elif self.gate_mode != "ltc" or self.ode_solver != "euler":
                reason = "only --gate-mode ltc with --ode-solver euler is supported"
            if reason is not None:
                print(f"[KERNEL] triton недоступен ({reason}) — используется torch.")
                actual = "torch"
        for cell in self.cells:
            cell.kernel = actual
        self._kernel = actual
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
        return [
            self.h0[i].unsqueeze(0).expand(batch_size, -1).contiguous()
            for i in range(self.num_layers)
        ]

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
                                compiled(x_t, xp_t, tx_t, hidden, 1.0, fused)
                        else:
                            # fused is recomputed on every iteration: the
                            # torch.cat graph must not be reused after backward.
                            fused = [cell.fused_h_weights() for cell in self.cells]
                            out, _ = compiled(x_t, xp_t, tx_t, hidden, 1.0, fused)
                            out.sum().backward()
                    if not inference:
                        self.zero_grad(set_to_none=True)
                    self._step_fn = compiled
                    return True, try_mode
                except Exception as exc:
                    self._step_fn = self._step_impl
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

    def _step_impl(self, x_t, xp_t, tx_t, hidden, dt, fused):
        hidden = list(hidden)
        do_residual = self._do_residual
        use_ckpt = self.use_checkpoint and self.training

        for i, cell in enumerate(self.cells):
            w, b = fused[i]
            if use_ckpt:
                h_new = cp.checkpoint(
                    cell,
                    hidden[i], x_t,
                    dt=dt,
                    xp=xp_t if i == 0 else None,
                    tx=tx_t if i == 0 else None,
                    fused_w=w,
                    fused_b=b,
                    use_reentrant=False,
                )
            else:
                if i == 0:
                    h_new = cell(hidden[i], x_t, dt=dt, xp=xp_t, tx=tx_t, fused_w=w, fused_b=b)
                else:
                    h_new = cell(hidden[i], x_t, dt=dt, fused_w=w, fused_b=b)

            hidden[i] = h_new
            if do_residual[i]:
                x_t = x_t + h_new
            else:
                x_t = h_new

        return self.final_norm(x_t), hidden

    def forward(
        self,
        tokens: torch.Tensor,
        hidden: list[torch.Tensor] | None = None,
        dt: float = 1.0,
        fused: list | None = None,
    ):
        batch_size, seq_len = tokens.shape
        conv_buf = None
        if hidden is None:
            hidden = self.initial_hidden(batch_size, tokens.device)
        else:
            hidden = list(hidden)
            if len(hidden) > self.num_layers:
                conv_buf = hidden[self.num_layers]
                hidden = hidden[: self.num_layers]
        new_conv_buf = None

        x = self.emb_dropout(self.embedding(tokens))

        if self.local_conv_layer is not None:
            if conv_buf is None:
                conv_buf = x.new_zeros(batch_size, self._conv_left_pad, x.size(-1))
            x_ext = torch.cat([conv_buf.to(x.dtype), x], dim=1)
            new_conv_buf = x_ext[:, -self._conv_left_pad:, :]
            x_conv = self.local_conv_layer(x_ext.transpose(1, 2)).transpose(1, 2)
            x = x + x_conv

        if fused is None:
            fused = [cell.fused_h_weights() for cell in self.cells]

        cell0 = self.cells[0]
        xp_seq, tx_seq = cell0.precompute_x(x)

        outputs = []
        # The compiled step has no triton branch — so with the triton
        # kernel active, inference goes through eager _step_impl.
        if self._kernel == "triton" and not self.training:
            step_fn = self._step_impl
        else:
            step_fn = self._step_fn
        for t in range(seq_len):
            out_t, hidden = step_fn(
                x[:, t, :], xp_seq[:, t, :], tx_seq[:, t, :], hidden, dt, fused
            )
            outputs.append(out_t)

        features = torch.stack(outputs, dim=1)
        logits = self.readout(features)
        if new_conv_buf is not None:
            hidden = list(hidden) + [new_conv_buf]
        return logits, hidden

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
    ) -> str:
        self.eval()
        if not prompt:
            prompt = " "
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = tokenizer.encode(" ")
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        stop_ids = set(stop_token_ids or ())

        fused = [cell.fused_h_weights() for cell in self.cells]
        logits_all, hidden = self.forward(tokens, fused=fused)
        next_logits = logits_all[:, -1, :].float()

        generated_ids = list(prompt_ids)
        recent_tokens = generated_ids[-repetition_window:]
        temperature = max(float(temperature), 1e-4)

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
            recent_tokens.append(token_id)
            recent_tokens = recent_tokens[-repetition_window:]
            if step_i + 1 < max_new_tokens:
                logits_all, hidden = self.forward(next_token, hidden=hidden, fused=fused)
                next_logits = logits_all[:, -1, :].float()

        if only_new:
            return tokenizer.decode(generated_ids[len(prompt_ids):])
        return tokenizer.decode(generated_ids)
