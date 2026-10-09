"""Liquid model layers: dropout, Triton kernel, NCP, liquid cell and chunk memory."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
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


NORM_TYPES = ("layer", "rms")


class RMSNorm(nn.Module):
    """RMSNorm (no mean subtraction, no bias). Statistics are computed in
    float32 regardless of the autocast dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "rms":
        return RMSNorm(dim)
    raise ValueError(f"Неизвестный norm_type: {kind!r} (ожидался 'layer' или 'rms')")


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
        norm_type: str = "layer",
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
# Triton kernel (inference only)
# ---------------------------------------------------------------------------


if _TRITON_AVAILABLE:

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


def _ltc_tail_triton(h, candidate, tau_raw, mix_w, sub_dt, min_tau, multi_tau):
    """Wrapper around _ltc_tail_kernel. INFERENCE ONLY: the kernel has no backward
    (see LiquidCell._use_triton_tail — it is never called when autograd is
    enabled)."""
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


# ---------------------------------------------------------------------------
# NCP recurrent matrix
# ---------------------------------------------------------------------------


class NCPRecurrent(nn.Module):
    """
    A block-structured replacement for ONE dense (H,H) recurrent matrix
    (h_proj in the ltc gate, ff1_h/ff2_h in cfc) — in the spirit of Neural
    Circuit Policies (Lechner et al., "Neural circuit policies enabling
    auditable autonomy", Nature MI 2020): neurons are split into
    inter/command/motor, and connections are allowed only along the paths

        inter   -> inter    (recurrent)
        inter   -> command
        command -> command  (recurrent)
        command -> motor
        motor   -> command  (feedback, closes the loop)

    Each allowed path has its own SEPARATE, physically SMALLER matrix
    (nn.Linear over inter_size/command_size/motor_size, not hidden_size).
    A mask over a dense matrix in PyTorch gives no speedup — an ordinary
    dense matmul still computes all H*H products. Here the nonexistent
    connections are simply NOT there, neither in the weights nor in the
    computation — a real, not "on paper", reduction of both parameters and
    FLOPs.

    Within each allowed path the connections are still FULLY CONNECTED (not
    pruned element-wise).

    NOTE (a deliberate simplification): synapse polarity (excitatory/
    inhibitory via the reversal potential in the sigmoidal conductance from
    the original LTC biophysics) is not reproduced — the sign and magnitude
    of the weights on the allowed connections are still learned freely by
    gradient descent.
    """

    def __init__(
        self,
        hidden_size: int,
        command_frac: float = 0.4,
        motor_frac: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_size < 3:
            raise ValueError("--wiring ncp requires hidden_size >= 3")
        if not (0.0 < command_frac < 1.0) or not (0.0 < motor_frac < 1.0):
            raise ValueError("--ncp-command-frac and --ncp-motor-frac must be in (0, 1)")
        if command_frac + motor_frac >= 1.0:
            raise ValueError("--ncp-command-frac + --ncp-motor-frac must be < 1")

        motor_size = max(1, round(hidden_size * motor_frac))
        command_size = max(1, round(hidden_size * command_frac))
        inter_size = hidden_size - motor_size - command_size
        if inter_size < 1:
            raise ValueError(
                f"hidden_size={hidden_size} слишком мал для "
                f"ncp_motor_frac={motor_frac}, ncp_command_frac={command_frac}"
            )

        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.command_size = command_size
        self.motor_size = motor_size

        self.w_ii = nn.Linear(inter_size, inter_size, bias=False)
        self.w_ic = nn.Linear(inter_size, command_size, bias=False)
        self.w_cc = nn.Linear(command_size, command_size, bias=False)
        self.w_cm = nn.Linear(command_size, motor_size, bias=False)
        self.w_mc = nn.Linear(motor_size, command_size, bias=False)  # feedback
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.w_ii, self.w_ic, self.w_cc, self.w_cm, self.w_mc):
            nn.init.orthogonal_(lin.weight, gain=0.99)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_inter, h_command, h_motor = torch.split(
            h, [self.inter_size, self.command_size, self.motor_size], dim=-1
        )
        inter_out = self.w_ii(h_inter)
        command_out = self.w_ic(h_inter) + self.w_cc(h_command) + self.w_mc(h_motor)
        motor_out = self.w_cm(h_command)
        return torch.cat([inter_out, command_out, motor_out], dim=-1)

    @property
    def param_fraction(self) -> float:
        """Fraction of parameters/FLOPs relative to the full dense H x H matrix."""
        H, Hi, Hc, Hm = self.hidden_size, self.inter_size, self.command_size, self.motor_size
        used = Hi * Hi + Hi * Hc + Hc * Hc + 2 * Hc * Hm
        return used / (H * H)


# ---------------------------------------------------------------------------
# Liquid cell
# ---------------------------------------------------------------------------


class LiquidCell(nn.Module):
    """
    Two gate modes (gate_mode):

    - "ltc" (default): a physically motivated linear ODE
      dh/dt = (candidate - h) / tau, with candidate and tau "frozen" over the
      substep and solved exactly (exponential integrator). Within a substep,
      candidate/tau are computed either once at the start of the substep
      (ode_solver="euler") or twice with averaging (ode_solver="heun" —
      predictor-corrector, second-order accuracy; twice as expensive per
      substep).

    - "cfc": the closed-form formula from Hasani et al., "Closed-form
      Continuous-time Neural Networks" (Nature Machine Intelligence, 2022):
      two state candidates (ff1, ff2) and a learnable sigmoid interpolation
      over time between them. With gate_mode="cfc" the --multi-tau parameter
      is meaningless (there is no tau) and is ignored.

    wiring="dense" (default) — an ordinary fully connected recurrent
    matrix. wiring="ncp" — the same matrix is replaced by NCPRecurrent.
    In cfc mode, ff1_h and ff2_h are two independent NCPRecurrent instances.
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
        ncp_command_frac: float = 0.4,
        ncp_motor_frac: float = 0.2,
        norm_type: str = "layer",
    ) -> None:
        super().__init__()
        if ode_unfolds < 1:
            raise ValueError("ode_unfolds must be >= 1")
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be > 0")
        if gate_mode not in ("ltc", "cfc"):
            raise ValueError(f"Неизвестный gate_mode: {gate_mode!r} (ожидался 'ltc' или 'cfc')")
        if ode_solver not in ("euler", "heun"):
            raise ValueError(f"Неизвестный ode_solver: {ode_solver!r} (ожидался 'euler' или 'heun')")
        if wiring not in ("dense", "ncp"):
            raise ValueError(f"Неизвестный wiring: {wiring!r} (ожидался 'dense' или 'ncp')")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        self.min_tau = min_tau
        self.zoneout = zoneout
        self.gate_mode = gate_mode
        self.ode_solver = ode_solver
        self.wiring = wiring
        self.kernel = "torch"
        self.multi_tau = bool(multi_tau) and gate_mode == "ltc"
        self._inv_unfolds = 1.0 / ode_unfolds

        if gate_mode == "cfc":
            self.ff1_x = nn.Linear(input_size, hidden_size)
            self.ff2_x = nn.Linear(input_size, hidden_size)
            if wiring == "ncp":
                self.ff1_h_ncp = NCPRecurrent(hidden_size, ncp_command_frac, ncp_motor_frac)
                self.ff2_h_ncp = NCPRecurrent(hidden_size, ncp_command_frac, ncp_motor_frac)
            else:
                self.ff1_h = nn.Linear(hidden_size, hidden_size, bias=False)
                self.ff2_h = nn.Linear(hidden_size, hidden_size, bias=False)
            self.time_net = nn.Linear(input_size + hidden_size, hidden_size * 2)
        else:
            self.x_proj = nn.Linear(input_size, hidden_size)
            if wiring == "ncp":
                self.h_proj_ncp = NCPRecurrent(hidden_size, ncp_command_frac, ncp_motor_frac)
            else:
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

    @property
    def recurrent_param_fraction(self) -> float | None:
        """Fraction of parameters/FLOPs of the recurrent matrix relative to the full
        dense H x H (only for wiring="ncp", otherwise None)."""
        if self.wiring != "ncp":
            return None
        ncp = self.ff1_h_ncp if self.gate_mode == "cfc" else self.h_proj_ncp
        return ncp.param_fraction

    def reset_parameters(self) -> None:
        if self.gate_mode == "cfc":
            for lin in (self.ff1_x, self.ff2_x):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            if self.wiring != "ncp":
                for lin in (self.ff1_h, self.ff2_h):
                    nn.init.orthogonal_(lin.weight, gain=0.99)
            nn.init.xavier_uniform_(self.time_net.weight)
            nn.init.zeros_(self.time_net.bias)
            return
        if self.wiring != "ncp":
            nn.init.orthogonal_(self.h_proj.weight, gain=0.99)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)
        nn.init.xavier_uniform_(self.tau_net[0].weight)
        nn.init.zeros_(self.tau_net[0].bias)
        nn.init.xavier_uniform_(self.tau_net[2].weight)
        nn.init.constant_(self.tau_net[2].bias, 0.1)

    def fused_h_weights(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepares one large (weight, bias) for a SINGLE F.linear call — only for
        wiring="dense". With wiring="ncp" the recurrent part is already several
        physically smaller matmuls with nothing to "glue" them into —
        _fused_from_h() is used instead, and this method returns (None, None)."""
        if self.wiring == "ncp":
            return None, None
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
        """Computes the "fused" projection of the current h: with wiring="dense" — a
        single F.linear; with wiring="ncp" — via the block-structured
        NCPRecurrent plus a separate dense matmul for tau/time gating, with the
        results concatenated into the same (B, 2H) / (B, 4H) layout that
        _ltc_step/_cfc_step expect."""
        if self.wiring != "ncp":
            return F.linear(h, fused_w, fused_b)
        if self.gate_mode == "cfc":
            ff1h = self.ff1_h_ncp(h)
            ff2h = self.ff2_h_ncp(h)
            time_h = F.linear(h, self.time_net.weight[:, self.input_size:], self.time_net.bias)
            return torch.cat([ff1h, ff2h, time_h], dim=-1)
        h_proj_out = self.h_proj_ncp(h)
        tau_h = F.linear(h, self.tau_net[0].weight[:, self.input_size:], self.tau_net[0].bias)
        return torch.cat([h_proj_out, tau_h], dim=-1)

    def precompute_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The part of the projections that depends only on x (not on h) — computed
        once for the whole sequence up front (see
        LiquidLanguageModel.forward) rather than anew at every step."""
        if self.gate_mode == "cfc":
            xp = torch.cat([self.ff1_x(x), self.ff2_x(x)], dim=-1)
            tx = F.linear(x, self.time_net.weight[:, :self.input_size])
        else:
            xp = self.x_proj(x)
            tx = F.linear(x, self.tau_net[0].weight[:, :self.input_size])
        return xp, tx

    def _use_triton_tail(self, h: torch.Tensor) -> bool:
        """The Triton tail is enabled ONLY when it is safe: explicitly requested
        (--kernel triton), the package is present, the tensor is on CUDA, it is
        ltc+euler, the cell is in eval mode and autograd is disabled (the
        kernel has no backward)."""
        return (
            self.kernel == "triton"
            and _TRITON_AVAILABLE
            and h.is_cuda
            and not self.training
            and not torch.is_grad_enabled()
            and self.gate_mode == "ltc"
            and self.ode_solver == "euler"
        )

    def _ltc_step(
        self, h: torch.Tensor, xp: torch.Tensor, tx: torch.Tensor,
        sub_dt: float, fused_w: torch.Tensor, fused_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One explicit substep of the ltc gate: candidate/tau are evaluated at the
        point h (the start of the substep), then the ODE dh/dt=(candidate-h)/tau
        is solved exactly over this substep with "frozen" candidate/tau.
        Returns (h_new, candidate, tau) — candidate and tau are needed outside
        for the heun corrector (see forward)."""
        H = self.hidden_size
        fused = self._fused_from_h(h, fused_w, fused_b)
        candidate = torch.tanh(xp + fused[:, :H])
        tau_raw = self.tau_net[2](torch.tanh(fused[:, H:] + tx))
        if self._use_triton_tail(h):
            mix_w = F.softmax(self.tau_mix, dim=0) if self.multi_tau else None
            h_new = _ltc_tail_triton(
                h, candidate, tau_raw, mix_w, sub_dt, self.min_tau, self.multi_tau
            )
            return h_new, candidate, None
        if self.multi_tau:
            tau_raw = tau_raw.view(-1, 3, H)
            tau_fast = F.softplus(tau_raw[:, 0, :] - 2.0) + 0.05
            tau_med = F.softplus(tau_raw[:, 1, :]) + 0.01
            tau_slow = F.softplus(tau_raw[:, 2, :] + 2.0) + 0.001
            w = F.softmax(self.tau_mix, dim=0)
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
    ) -> torch.Tensor:
        sub_dt = dt * self._inv_unfolds
        h_in = h

        if xp is None or tx is None:
            xp, tx = self.precompute_x(x)
        if self.wiring != "ncp" and (fused_w is None or fused_b is None):
            fused_w, fused_b = self.fused_h_weights()

        for _ in range(self.ode_unfolds):
            if self.gate_mode == "cfc":
                h = self._cfc_step(h, xp, tx, sub_dt, fused_w, fused_b)
            elif self.ode_solver == "heun":
                h_pred, cand0, tau0 = self._ltc_step(h, xp, tx, sub_dt, fused_w, fused_b)
                _, cand1, tau1 = self._ltc_step(h_pred, xp, tx, sub_dt, fused_w, fused_b)
                candidate = 0.5 * (cand0 + cand1)
                tau = 0.5 * (tau0 + tau1)
                decay = torch.exp(-sub_dt / tau)
                h = decay * h + (1.0 - decay) * candidate
            else:
                h, _, _ = self._ltc_step(h, xp, tx, sub_dt, fused_w, fused_b)

        out = self.norm(self.dropout(h))

        if self.training and self.zoneout > 0.0:
            prev = self.norm(h_in)
            keep = torch.rand_like(out) < self.zoneout
            out = torch.where(keep, prev, out)

        return out


# ---------------------------------------------------------------------------
# Chunk memory
# ---------------------------------------------------------------------------


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
