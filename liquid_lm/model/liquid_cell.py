"""Liquid cell: ltc and cfc gates, dense/ncp wiring."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ncp import NCPRecurrent
from .triton_kernel import _TRITON_AVAILABLE, _ltc_tail_triton


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

        self.norm = nn.LayerNorm(hidden_size)
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
