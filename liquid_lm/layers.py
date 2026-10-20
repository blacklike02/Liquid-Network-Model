"""Liquid model layers: dropout, Triton kernel, liquid cell and chunk memory."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Locked dropout
# ---------------------------------------------------------------------------


class LockedDropout(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)


# ---------------------------------------------------------------------------
# Normalization and channel-mixing block
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm without mean subtraction or bias.

    FP16/BF16 use FP32 statistics; explicit FP64 inputs retain FP64 precision.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prefer the native operation; keep FP32 statistics for AMP.
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        xf, weight = x.to(compute_dtype), self.weight.to(compute_dtype)
        if hasattr(F, "rms_norm"):
            out = F.rms_norm(xf, (self.weight.numel(),), weight, self.eps)
        else:  # PyTorch 2.0-2.3 compatibility.
            out = xf * (xf.square().mean(-1, keepdim=True) + self.eps).rsqrt() * weight
        return out.to(x.dtype)


def make_norm(kind: str, dim: int) -> nn.Module:
    # Keep the keyword for Python callers/checkpoint metadata, not as a choice.
    if kind != "rms":
        raise ValueError("Only RMSNorm is supported (norm_type='rms'). "
                         "LayerNorm has been retired; use an earlier revision for its checkpoints.")
    return RMSNorm(dim)


class ResidualFFN(nn.Module):
    """Pre-norm SwiGLU channel-mixing block with its own residual connection:

        x + Dropout(W_out(silu(W_gate(norm(x))) * W_up(norm(x))))

    It is applied per token between liquid layers (inside the recurrent step),
    so it adds only token-wise computation and no state. The cell itself mixes
    channels only through its recurrent/gate matrices; this block is the
    standard "MLP" half of a Transformer-style layer (the LFM2 backbone also
    uses SwiGLU MLPs).

    The output projection is scaled down by 1/sqrt(2 * num_layers) at init so
    that stacking blocks does not inflate the residual stream at the start.
    """

    def __init__(
        self,
        dim: int,
        mult: float,
        norm_type: str = "rms",
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if mult <= 0:
            raise ValueError("ffn_mult must be > 0")
        inner = max(8, int(round(dim * mult / 8.0)) * 8)
        self.inner = inner
        self.norm = make_norm(norm_type, dim)
        self.w_in = nn.Linear(dim, 2 * inner, bias=False)
        self.w_out = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        with torch.no_grad():
            self.w_out.weight.mul_(1.0 / (2.0 * max(1, num_layers)) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_in(self.norm(x)).chunk(2, dim=-1)
        return x + self.dropout(self.w_out(F.silu(gate) * up))


# ---------------------------------------------------------------------------
# Triton kernel (experimental: inference + first-order training backward)
# ---------------------------------------------------------------------------


if _TRITON_AVAILABLE:

    @triton.jit
    def _ltc_activations_kernel(
        fused_ptr, xp_ptr, tx_ptr, candidate_ptr, tau_hidden_ptr,
        n_elements, H: tl.constexpr,
        F0: tl.constexpr, F1: tl.constexpr,
        X0: tl.constexpr, X1: tl.constexpr,
        T0: tl.constexpr, T1: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        row, col = offs // H, offs % H
        f0 = tl.load(fused_ptr + row * F0 + col * F1, mask, other=0).to(tl.float32)
        f1 = tl.load(fused_ptr + row * F0 + (col + H) * F1, mask, other=0).to(tl.float32)
        xp = tl.load(xp_ptr + row * X0 + col * X1, mask, other=0).to(tl.float32)
        tx = tl.load(tx_ptr + row * T0 + col * T1, mask, other=0).to(tl.float32)
        # PyTorch rounds each addition before tanh under FP16/BF16 autocast.
        candidate_sum = (xp + f0).to(candidate_ptr.dtype.element_ty).to(tl.float32)
        tau_sum = (f1 + tx).to(tau_hidden_ptr.dtype.element_ty).to(tl.float32)
        tl.store(candidate_ptr + offs, libdevice.tanh(candidate_sum), mask)
        tl.store(tau_hidden_ptr + offs, libdevice.tanh(tau_sum), mask)

    @triton.jit
    def _ltc_activations_backward_kernel(
        candidate_ptr, tau_hidden_ptr, grad_candidate_ptr, grad_tau_hidden_ptr,
        grad_fused_ptr, grad_xp_ptr, grad_tx_ptr,
        n_elements, H: tl.constexpr,
        C0: tl.constexpr, C1: tl.constexpr,
        T0: tl.constexpr, T1: tl.constexpr,
        HAS_CANDIDATE_GRAD: tl.constexpr, HAS_TAU_GRAD: tl.constexpr,
        SHARED_GRADS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        row, col = offs // H, offs % H
        candidate = tl.load(candidate_ptr + offs, mask, other=0).to(tl.float32)
        tau_hidden = tl.load(tau_hidden_ptr + offs, mask, other=0).to(tl.float32)
        dc = tl.full((BLOCK,), 0, tl.float32)
        dt = tl.full((BLOCK,), 0, tl.float32)
        if HAS_CANDIDATE_GRAD:
            dc = tl.load(grad_candidate_ptr + row * C0 + col * C1, mask, other=0).to(tl.float32)
        if HAS_TAU_GRAD:
            dt = tl.load(grad_tau_hidden_ptr + row * T0 + col * T1, mask, other=0).to(tl.float32)
        # CUDA's native tanh backward uses scalar_t arithmetic: FP16/BF16
        # round the square and subtraction as well as the final product.
        # Keeping only FP32 intermediates here changes AMP training gradients.
        c_square = (candidate * candidate).to(candidate_ptr.dtype.element_ty).to(tl.float32)
        t_square = (tau_hidden * tau_hidden).to(tau_hidden_ptr.dtype.element_ty).to(tl.float32)
        c_slope = (1.0 - c_square).to(candidate_ptr.dtype.element_ty).to(tl.float32)
        t_slope = (1.0 - t_square).to(tau_hidden_ptr.dtype.element_ty).to(tl.float32)
        if candidate_ptr.dtype.element_ty == tl.float32:
            c_slope = libdevice.fma(-candidate, candidate, 1.0)
        if tau_hidden_ptr.dtype.element_ty == tl.float32:
            t_slope = libdevice.fma(-tau_hidden, tau_hidden, 1.0)
        dc = dc * c_slope
        dt = dt * t_slope
        tl.store(grad_fused_ptr + row * (2 * H) + col, dc, mask)
        tl.store(grad_fused_ptr + row * (2 * H) + H + col, dt, mask)
        if not SHARED_GRADS:
            tl.store(grad_xp_ptr + offs, dc, mask)
            tl.store(grad_tx_ptr + offs, dt, mask)

    @triton.jit
    def _ltc_tail_kernel(
        h_ptr, cand_ptr, tau_raw_ptr, w_ptr, out_ptr,
        sub_dt, min_tau, n_elements, H,
        MULTI_TAU: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        h = tl.load(h_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        cand = tl.load(cand_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if MULTI_TAU:
            row = offs // H
            col = offs - row * H
            base = row * (3 * H) + col
            r0 = tl.load(tau_raw_ptr + base, mask=mask, other=0.0).to(tl.float32) - 2.0
            r1 = tl.load(tau_raw_ptr + base + H, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(tau_raw_ptr + base + 2 * H, mask=mask, other=0.0).to(tl.float32) + 2.0
            sp0 = tl.maximum(r0, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r0)))
            sp1 = tl.maximum(r1, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r1)))
            sp2 = tl.maximum(r2, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r2)))
            w0 = tl.load(w_ptr).to(tl.float32)
            w1 = tl.load(w_ptr + 1).to(tl.float32)
            w2 = tl.load(w_ptr + 2).to(tl.float32)
            tau = w0 * (sp0 + 0.05) + w1 * (sp1 + 0.01) + w2 * (sp2 + 0.001)
        else:
            r = tl.load(tau_raw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            tau = tl.maximum(r, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r))) + min_tau
        decay = tl.exp(-sub_dt / tau)
        res = decay * h + (1.0 - decay) * cand
        tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _ltc_tail_backward_kernel(
        h_ptr, cand_ptr, tau_raw_ptr, mix_ptr, grad_ptr,
        grad_h_ptr, grad_cand_ptr, grad_tau_ptr, grad_mix_ptr,
        sub_dt, min_tau, n_elements, H,
        MULTI_TAU: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Fused first-order backward for the exact LTC Euler tail update."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        h = tl.load(h_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        cand = tl.load(cand_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        row = offs // H
        col = offs - row * H
        base = row * (3 * H) + col

        if MULTI_TAU:
            r0 = tl.load(tau_raw_ptr + base, mask=mask, other=0.0).to(tl.float32) - 2.0
            r1 = tl.load(tau_raw_ptr + base + H, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(tau_raw_ptr + base + 2 * H, mask=mask, other=0.0).to(tl.float32) + 2.0
            s0 = 1.0 / (1.0 + tl.exp(-r0))
            s1 = 1.0 / (1.0 + tl.exp(-r1))
            s2 = 1.0 / (1.0 + tl.exp(-r2))
            t0 = tl.maximum(r0, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r0))) + 0.05
            t1 = tl.maximum(r1, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r1))) + 0.01
            t2 = tl.maximum(r2, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r2))) + 0.001
            w0 = tl.load(mix_ptr).to(tl.float32)
            w1 = tl.load(mix_ptr + 1).to(tl.float32)
            w2 = tl.load(mix_ptr + 2).to(tl.float32)
            tau = w0 * t0 + w1 * t1 + w2 * t2
        else:
            raw = tl.load(tau_raw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            sig = 1.0 / (1.0 + tl.exp(-raw))
            tau = tl.maximum(raw, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(raw))) + min_tau

        decay = tl.exp(-sub_dt / tau)
        # d update / d tau = (h - candidate) * exp(-dt/tau) * dt / tau^2
        grad_tau = grad * (h - cand) * decay * sub_dt / (tau * tau)
        tl.store(grad_h_ptr + offs, grad * decay, mask=mask)
        tl.store(grad_cand_ptr + offs, grad * (1.0 - decay), mask=mask)

        if MULTI_TAU:
            tl.store(grad_tau_ptr + base, grad_tau * w0 * s0, mask=mask)
            tl.store(grad_tau_ptr + base + H, grad_tau * w1 * s1, mask=mask)
            tl.store(grad_tau_ptr + base + 2 * H, grad_tau * w2 * s2, mask=mask)
            # Sum within the CTA: global scratch is O(ceil(B*H/BLOCK)), not O(B*H).
            mix0 = tl.sum(tl.where(mask, grad_tau * t0, 0.0), axis=0)
            mix1 = tl.sum(tl.where(mask, grad_tau * t1, 0.0), axis=0)
            mix2 = tl.sum(tl.where(mask, grad_tau * t2, 0.0), axis=0)
            tl.store(grad_mix_ptr + pid * 3, mix0)
            tl.store(grad_mix_ptr + pid * 3 + 1, mix1)
            tl.store(grad_mix_ptr + pid * 3 + 2, mix2)
        else:
            tl.store(grad_tau_ptr + offs, grad_tau * sig, mask=mask)

def _ltc_activations_triton(fused, xp, tx):
    """Two add/tanh operations without intermediate sums or slice backward."""
    batch, hidden = xp.shape
    candidate = torch.empty((batch, hidden), device=xp.device,
                            dtype=torch.promote_types(fused.dtype, xp.dtype))
    tau_hidden = torch.empty((batch, hidden), device=tx.device,
                             dtype=torch.promote_types(fused.dtype, tx.dtype))
    _ltc_activations_kernel[(triton.cdiv(batch * hidden, 1024),)](
        fused, xp, tx, candidate, tau_hidden, batch * hidden, hidden,
        *fused.stride(), *xp.stride(), *tx.stride(), BLOCK=1024,
    )
    return candidate, tau_hidden


class _LTCActivationsAutograd(torch.autograd.Function):
    """First-order backward also combines the two recurrent projection slices."""

    @staticmethod
    def forward(ctx, fused, xp, tx):
        candidate, tau_hidden = _ltc_activations_triton(fused, xp, tx)
        ctx.save_for_backward(candidate, tau_hidden)
        ctx.input_dtypes = fused.dtype, xp.dtype, tx.dtype
        ctx.set_materialize_grads(False)
        return candidate, tau_hidden

    @staticmethod
    def backward(ctx, grad_candidate, grad_tau_hidden):
        candidate, tau_hidden = ctx.saved_tensors
        batch, hidden = candidate.shape
        fused_dtype, xp_dtype, tx_dtype = ctx.input_dtypes
        grad_fused = torch.empty((batch, 2 * hidden), device=candidate.device, dtype=fused_dtype)
        shared = fused_dtype == xp_dtype == tx_dtype
        # Views of the combined gradient avoid both slice-scatter kernels and
        # separate buffers when all input dtypes match (including normal AMP).
        grad_xp = grad_fused[:, :hidden] if shared else torch.empty_like(candidate, dtype=xp_dtype)
        grad_tx = grad_fused[:, hidden:] if shared else torch.empty_like(tau_hidden, dtype=tx_dtype)
        gc = grad_candidate if grad_candidate is not None else candidate
        gt = grad_tau_hidden if grad_tau_hidden is not None else tau_hidden
        _ltc_activations_backward_kernel[(triton.cdiv(batch * hidden, 1024),)](
            candidate, tau_hidden, gc, gt, grad_fused, grad_xp, grad_tx,
            batch * hidden, hidden, *gc.stride(), *gt.stride(),
            HAS_CANDIDATE_GRAD=grad_candidate is not None,
            HAS_TAU_GRAD=grad_tau_hidden is not None,
            SHARED_GRADS=shared, BLOCK=1024, enable_fp_fusion=False,
        )
        return grad_fused, grad_xp, grad_tx


def _ltc_activations_triton_autograd(fused, xp, tx):
    if torch.is_grad_enabled():
        return _LTCActivationsAutograd.apply(fused, xp, tx)
    return _ltc_activations_triton(fused, xp, tx)


def _ltc_tail_triton(h, candidate, tau_raw, mix_w, sub_dt, min_tau, multi_tau):
    """Run the fused forward tail; _LTCTailAutograd supplies its first-order backward."""
    h = h.contiguous()
    candidate = candidate.contiguous()
    tau_raw = tau_raw.contiguous()
    B, H = h.shape
    out = torch.empty_like(h)
    n = B * H
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _ltc_tail_kernel[grid](
        h, candidate, tau_raw, mix_w if mix_w is not None else h, out,
        float(sub_dt), float(min_tau), n, H,
        MULTI_TAU=bool(multi_tau), BLOCK=BLOCK,
    )
    return out


class _LTCTailAutograd(torch.autograd.Function):
    """Triton LTC tail with a first-order custom backward."""

    @staticmethod
    def forward(ctx, h, candidate, tau_raw, mix_w, sub_dt, min_tau, multi_tau):
        h_c = h.contiguous()
        cand_c = candidate.contiguous()
        tau_c = tau_raw.contiguous()
        has_mix = mix_w is not None
        mix_c = mix_w.contiguous() if has_mix else h.new_empty((0,))
        ctx.save_for_backward(h_c, cand_c, tau_c, mix_c)
        ctx.sub_dt = float(sub_dt)
        ctx.min_tau = float(min_tau)
        ctx.multi_tau = bool(multi_tau)
        return _ltc_tail_triton(
            h_c, cand_c, tau_c, mix_c if has_mix else None,
            ctx.sub_dt, ctx.min_tau, ctx.multi_tau,
        )

    @staticmethod
    def backward(ctx, grad_output):
        h, candidate, tau_raw, mix_w = ctx.saved_tensors
        batch, hidden = h.shape
        n_elements = batch * hidden
        grad_output = grad_output.contiguous()
        grad_h = torch.empty_like(h)
        grad_candidate = torch.empty_like(candidate)
        grad_tau_raw = torch.empty_like(tau_raw)
        block = 1024
        grid_size = triton.cdiv(n_elements, block)
        grad_mix_parts = (
            torch.empty((grid_size, 3), device=h.device, dtype=torch.float32)
            if ctx.multi_tau else grad_h
        )
        _ltc_tail_backward_kernel[(grid_size,)](
            h, candidate, tau_raw, mix_w, grad_output,
            grad_h, grad_candidate, grad_tau_raw, grad_mix_parts,
            ctx.sub_dt, ctx.min_tau, n_elements, hidden,
            MULTI_TAU=ctx.multi_tau, BLOCK=block,
        )
        grad_mix = grad_mix_parts.sum(dim=0) if ctx.multi_tau else None
        return grad_h, grad_candidate, grad_tau_raw, grad_mix, None, None, None


def _ltc_tail_triton_autograd(h, candidate, tau_raw, mix_w, sub_dt, min_tau, multi_tau):
    # Custom autograd is needed only while recording a backward graph.
    if torch.is_grad_enabled():
        return _LTCTailAutograd.apply(
            h, candidate, tau_raw, mix_w, sub_dt, min_tau, multi_tau
        )
    return _ltc_tail_triton(h, candidate, tau_raw, mix_w, sub_dt, min_tau, multi_tau)

# ---------------------------------------------------------------------------
# Liquid cell
# ---------------------------------------------------------------------------


def validate_retired_architecture(ode_unfolds=1, ode_solver="euler", wiring="dense"):
    """Reject retired equations instead of silently changing old checkpoints."""
    for name, value, supported in (("ode_unfolds", ode_unfolds, 1),
                                   ("ode_solver", ode_solver, "euler"),
                                   ("wiring", wiring, "dense")):
        if value != supported:
            raise ValueError(f"{name}={value!r} is retired; only {supported!r} is supported. "
                             "Use the earlier repository revision for this checkpoint.")


class LiquidCell(nn.Module):
    """Dense recurrent LTC exponential update or simplified closed-form CfC.

    The LTC candidate/time constant are evaluated once per token. Retired
    architecture arguments accept only their former baseline values so old
    baseline callers retain their exact equations and weight layout.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ode_unfolds: int = 1,
        min_tau: float = 0.05,
        dropout: float = 0.0,
        zoneout: float = 0.0,
        multi_tau: bool = True,
        gate_mode: str = "ltc",
        ode_solver: str = "euler",
        wiring: str = "dense",
        norm_type: str = "rms",
    ) -> None:
        super().__init__()
        validate_retired_architecture(ode_unfolds, ode_solver, wiring)
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be > 0")
        if gate_mode not in ("ltc", "cfc"):
            raise ValueError(f"Неизвестный gate_mode: {gate_mode!r} (ожидался 'ltc' или 'cfc')")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.min_tau = min_tau
        self.zoneout = zoneout
        self.gate_mode = gate_mode
        self.kernel = "torch"
        self.multi_tau = bool(multi_tau) and gate_mode == "ltc"

        if gate_mode == "cfc":
            self.ff1_x = nn.Linear(input_size, hidden_size)
            self.ff2_x = nn.Linear(input_size, hidden_size)
            self.ff1_h = nn.Linear(hidden_size, hidden_size, bias=False)
            self.ff2_h = nn.Linear(hidden_size, hidden_size, bias=False)
            self.time_net = nn.Linear(input_size + hidden_size, hidden_size * 2)
        else:
            self.x_proj = nn.Linear(input_size, hidden_size)
            self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            tau_out = hidden_size * 3 if self.multi_tau else hidden_size
            self.tau_net = nn.Sequential(
                nn.Linear(input_size + hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, tau_out),
            )
            if self.multi_tau:
                self.tau_mix = nn.Parameter(torch.zeros(3))

        self.norm = make_norm(norm_type, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.gate_mode == "cfc":
            for lin in (self.ff1_x, self.ff2_x):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            for lin in (self.ff1_h, self.ff2_h):
                nn.init.orthogonal_(lin.weight, gain=0.99)
            nn.init.xavier_uniform_(self.time_net.weight)
            nn.init.zeros_(self.time_net.bias)
            return
        nn.init.orthogonal_(self.h_proj.weight, gain=0.99)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)
        nn.init.xavier_uniform_(self.tau_net[0].weight)
        nn.init.zeros_(self.tau_net[0].bias)
        nn.init.xavier_uniform_(self.tau_net[2].weight)
        nn.init.constant_(self.tau_net[2].bias, 0.1)

    def fused_h_weights(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Pack recurrent projections once for a single linear operation."""
        H = self.hidden_size
        if self.gate_mode == "cfc":
            fused_w = torch.cat(
                [self.ff1_h.weight, self.ff2_h.weight, self.time_net.weight[:, self.input_size:]],
                dim=0,
            )
            fused_b = torch.cat([fused_w.new_zeros(2 * H), self.time_net.bias])
            return fused_w, fused_b
        fused_w = torch.cat(
            [self.h_proj.weight, self.tau_net[0].weight[:, self.input_size:]],
            dim=0,
        )
        fused_b = torch.cat([fused_w.new_zeros(H), self.tau_net[0].bias])
        return fused_w, fused_b

    def _fused_from_h(
        self, h: torch.Tensor,
        fused_w: torch.Tensor | None, fused_b: torch.Tensor | None,
    ) -> torch.Tensor:
        return F.linear(h, fused_w, fused_b)

    def input_projection_weights(self):
        """Pack input-only affine projections once per sequence, retaining grads.

        In particular, slicing tau/time weights inside every recurrent step
        creates repeated full-weight scatter gradients. Packing here shares
        that graph and lets cuBLAS evaluate FP32 input heads in one GEMM.
        AMP keeps separate GEMMs: a joint low-precision input gradient rounds
        after summing heads instead of rounding each head before their FP32 sum.
        """
        parameter = self.ff1_x.weight if self.gate_mode == "cfc" else self.x_proj.weight
        if torch.is_autocast_enabled(parameter.device.type) or parameter.dtype in (torch.float16, torch.bfloat16):
            if self.gate_mode == "cfc":
                return (self.ff1_x.weight, self.ff1_x.bias, self.ff2_x.weight, self.ff2_x.bias,
                        self.time_net.weight[:, :self.input_size])
            return self.x_proj.weight, self.x_proj.bias, self.tau_net[0].weight[:, :self.input_size]
        if self.gate_mode == "cfc":
            weight = torch.cat((self.ff1_x.weight, self.ff2_x.weight,
                                self.time_net.weight[:, :self.input_size]), dim=0)
            bias = torch.cat((self.ff1_x.bias, self.ff2_x.bias,
                              self.time_net.bias.new_zeros(2 * self.hidden_size)))
        else:
            weight = torch.cat((self.x_proj.weight, self.tau_net[0].weight[:, :self.input_size]), dim=0)
            bias = torch.cat((self.x_proj.bias, self.x_proj.bias.new_zeros(self.hidden_size)))
        return weight, bias

    def precompute_x(self, x: torch.Tensor, input_projection=None) -> tuple[torch.Tensor, torch.Tensor]:
        """The part of the projections that depends only on x (not on h) — computed
        once for the whole sequence up front (see
        LiquidLanguageModel.forward) rather than anew at every step."""
        if input_projection is not None:
            if len(input_projection) == 3:
                weight, bias, time_weight = input_projection
                return F.linear(x, weight, bias), F.linear(x, time_weight)
            if len(input_projection) == 5:
                w1, b1, w2, b2, time_weight = input_projection
                return torch.cat((F.linear(x, w1, b1), F.linear(x, w2, b2)), dim=-1), F.linear(x, time_weight)
            projected = F.linear(x, *input_projection)
            width = self.hidden_size * (2 if self.gate_mode == "cfc" else 1)
            return projected.split(width, dim=-1)
        if self.gate_mode == "cfc":
            xp = torch.cat([self.ff1_x(x), self.ff2_x(x)], dim=-1)
            tx = F.linear(x, self.time_net.weight[:, :self.input_size])
        else:
            xp = self.x_proj(x)
            tx = F.linear(x, self.tau_net[0].weight[:, :self.input_size])
        return xp, tx

    def _use_triton_tail(self, h: torch.Tensor) -> bool:
        """Use the opt-in Triton LTC tail for CUDA inference/training.

        The experimental backward supports first-order gradients for LTC + Euler.
        """
        return (
            self.kernel == "triton"
            and _TRITON_AVAILABLE
            and h.is_cuda
            and self.gate_mode == "ltc"
        )
    def _ltc_step(
        self, h: torch.Tensor, xp: torch.Tensor, tx: torch.Tensor,
        sub_dt: float, fused_w: torch.Tensor, fused_b: torch.Tensor,
        mix_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One explicit substep of the ltc gate: candidate/tau are evaluated at the
        point h (the start of the substep), then the ODE dh/dt=(candidate-h)/tau
        is solved exactly over this substep with "frozen" candidate/tau.
        Returns the updated state and diagnostic candidate/tau tensors."""
        H = self.hidden_size
        fused = self._fused_from_h(h, fused_w, fused_b)
        use_triton = self._use_triton_tail(h)
        if use_triton:
            candidate, tau_hidden = _ltc_activations_triton_autograd(fused, xp, tx)
        else:
            candidate = torch.tanh(xp + fused[:, :H])
            tau_hidden = torch.tanh(fused[:, H:] + tx)
        tau_raw = self.tau_net[2](tau_hidden)
        if use_triton:
            if self.multi_tau and mix_w is None:
                mix_w = F.softmax(self.tau_mix, dim=0)
            h_new = _ltc_tail_triton_autograd(
                h, candidate, tau_raw, mix_w, sub_dt, self.min_tau, self.multi_tau
            )
            return h_new, candidate, None
        if self.multi_tau:
            tau_raw = tau_raw.view(-1, 3, H)
            tau_fast = F.softplus(tau_raw[:, 0, :] - 2.0) + 0.05
            tau_med = F.softplus(tau_raw[:, 1, :]) + 0.01
            tau_slow = F.softplus(tau_raw[:, 2, :] + 2.0) + 0.001
            w = F.softmax(self.tau_mix, dim=0) if mix_w is None else mix_w
            tau = w[0] * tau_fast + w[1] * tau_med + w[2] * tau_slow
        else:
            tau = F.softplus(tau_raw) + self.min_tau
        decay = torch.exp(-sub_dt / tau)
        h_new = decay * h + (1.0 - decay) * candidate
        return h_new, candidate, tau

    def _cfc_step(
        self, h: torch.Tensor, xp: torch.Tensor, tx: torch.Tensor,
        sub_dt: float, fused_w: torch.Tensor, fused_b: torch.Tensor,
    ) -> torch.Tensor:
        """Closed form of CfC: two candidates ff1/ff2 plus a learnable sigmoid
        interpolation t_interp(x,h,dt) between them — no ODE."""
        H = self.hidden_size
        fused = self._fused_from_h(h, fused_w, fused_b)
        ff1 = torch.tanh(xp[:, :H] + fused[:, :H])
        ff2 = torch.tanh(xp[:, H:] + fused[:, H:2 * H])
        t_a = fused[:, 2 * H:3 * H] + tx[:, :H]
        t_b = fused[:, 3 * H:] + tx[:, H:]
        t_interp = torch.sigmoid(t_a * sub_dt + t_b)
        return ff1 * (1.0 - t_interp) + t_interp * ff2

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        dt: float = 1.0,
        xp: torch.Tensor | None = None,
        tx: torch.Tensor | None = None,
        fused_w: torch.Tensor | None = None,
        fused_b: torch.Tensor | None = None,
        mix_w: torch.Tensor | None = None,
        input_projection=None,
    ) -> torch.Tensor:
        h_in = h

        if xp is None or tx is None:
            xp, tx = self.precompute_x(x, input_projection)
        if fused_w is None or fused_b is None:
            fused_w, fused_b = self.fused_h_weights()

        if self.gate_mode == "cfc":
            h = self._cfc_step(h, xp, tx, dt, fused_w, fused_b)
        else:
            h, _, _ = self._ltc_step(h, xp, tx, dt, fused_w, fused_b, mix_w)

        out = self.norm(self.dropout(h))

        if self.training and self.zoneout > 0.0:
            prev = h_in
            keep = torch.rand_like(out) < self.zoneout
            out = torch.where(keep, prev, out)

        return out


# ---------------------------------------------------------------------------
# Chunk memory
# ---------------------------------------------------------------------------


class ChunkMemory(nn.Module):
    """Historical full-width memory retained for existing checkpoint weights.

    Raw detached final states are compressed lazily during reads, so read
    projections receive gradients within the current chunk. Initial slots are
    learned. New training uses CompactChunkMemory with explicit occupancy.
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
        # CfC can return an autocast FP16/BF16 state. Reads between chunks run
        # outside autocast, so match the projection's parameter dtype here.
        h0 = h0.to(dtype=self.query_proj.weight.dtype)
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


class CompactChunkMemory(nn.Module):
    """Low-rank FIFO reads with explicit occupancy and detached raw-state writes.

    Slots retain raw states so the compression weights receive gradients at
    read time. The last channel records occupancy; empty slots contribute
    exactly zero, including at the start of a new conversation. No mutable
    memory lives on the module itself.
    """

    def __init__(self, hidden_size: int, memory_slots: int, memory_dim: int):
        super().__init__()
        if min(hidden_size, memory_slots, memory_dim) < 1:
            raise ValueError("hidden_size, memory_slots and memory_dim must be positive")
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.memory_dim = min(memory_dim, hidden_size)
        dim = self.memory_dim
        self.summarize = nn.Linear(hidden_size, dim)
        self.query_proj = nn.Linear(hidden_size, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.read_gate = nn.Linear(hidden_size + dim, dim)
        self.value_proj = nn.Linear(dim, hidden_size, bias=False)
        self._scale = dim ** -0.5
        for module in (self.summarize, self.query_proj, self.key_proj,
                       self.read_gate, self.value_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.read_gate.bias, -2.0)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.memory_slots, self.hidden_size + 1,
                           device=device, dtype=self.query_proj.weight.dtype)

    def read(self, h0: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h0 = h0.to(self.query_proj.weight.dtype)
        valid = memory[..., -1] > 0
        # Mask before projection too: invalid payloads cannot influence a read.
        raw = torch.where(valid.unsqueeze(-1), memory[..., :-1], 0).to(h0.dtype)
        summaries = torch.tanh(self.summarize(raw))
        query = self.query_proj(h0).unsqueeze(1)
        keys = self.key_proj(summaries)
        scores = (query.float() * keys.float()).sum(-1) * self._scale
        # A finite sentinel keeps an all-empty row well defined. Re-normalizing
        # after masking yields zero weights when no slot has been written.
        weights = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        retrieved = (weights.unsqueeze(-1) * summaries.float()).sum(1).to(h0.dtype)
        gate = torch.sigmoid(self.read_gate(torch.cat((h0, retrieved), -1)))
        return h0 + self.value_proj(gate * retrieved)

    def write(self, h_final: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        raw = h_final.detach().to(memory.dtype).unsqueeze(1)
        valid = raw.new_ones(raw.shape[0], 1, 1)
        return torch.cat((memory[:, 1:], torch.cat((raw, valid), -1)), 1)


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
