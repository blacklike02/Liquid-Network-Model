"""Block-structured recurrent matrix (Neural Circuit Policies)."""

from __future__ import annotations

import torch
import torch.nn as nn


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
