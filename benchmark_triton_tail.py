"""Microbenchmark the experimental Triton LTC tail against its PyTorch reference.

Run from the repository root after applying liquid_ltc_triton_training_v2.patch:
    python benchmark_triton_tail.py --batch-size 64 --hidden-size 384 --dtype bf16

This isolates the LTC state-update tail. It is NOT an end-to-end model benchmark.
"""
from __future__ import annotations

import argparse
import math
from typing import Callable

import torch
import torch.nn.functional as F

try:
    from liquid_lm.layers import _TRITON_AVAILABLE, _ltc_tail_triton_autograd
except Exception as exc:  # pragma: no cover - environment/import dependent
    raise SystemExit(
        "Cannot import the patched Triton LTC path. Apply the patch from the "
        "repository root and run this script there. Import error: " + str(exc)
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_reference(sub_dt: float, min_tau: float, multi_tau: bool) -> Callable:
    def reference(h: torch.Tensor, candidate: torch.Tensor,
                  tau_raw: torch.Tensor, mix_w: torch.Tensor | None) -> torch.Tensor:
        hidden = h.shape[-1]
        if multi_tau:
            banks = tau_raw.view(h.shape[0], 3, hidden)
            # Match the production PyTorch order of operations and constants.
            tau_fast = F.softplus(banks[:, 0, :] - 2.0) + 0.05
            tau_med = F.softplus(banks[:, 1, :]) + 0.01
            tau_slow = F.softplus(banks[:, 2, :] + 2.0) + 0.001
            assert mix_w is not None
            tau = mix_w[0] * tau_fast + mix_w[1] * tau_med + mix_w[2] * tau_slow
        else:
            tau = F.softplus(tau_raw) + min_tau
        decay = torch.exp(-sub_dt / tau)
        return decay * h + (1.0 - decay) * candidate
    return reference


def clone_leaves(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(v.detach().clone().requires_grad_(True) for v in values)


def evaluate_once(
    fn: Callable,
    leaves: tuple[torch.Tensor, ...],
    probe: torch.Tensor,
    *,
    capture: bool = True,
) -> tuple[torch.Tensor | None, list[torch.Tensor | None]]:
    for value in leaves:
        value.grad = None
    h, candidate, tau_raw, mix_w = leaves
    out = fn(h, candidate, tau_raw, mix_w)
    (out.float() * probe).sum().backward()
    if not capture:
        # Avoid extra clone kernels inside the timed region.
        return None, []
    grads = [value.grad.detach().clone() if value.grad is not None else None for value in leaves]
    return out.detach().clone(), grads


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def time_forward_backward(
    fn: Callable,
    leaves: tuple[torch.Tensor, ...],
    probe: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        evaluate_once(fn, leaves, probe, capture=False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        evaluate_once(fn, leaves, probe, capture=False)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations  # microseconds/iteration


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.hidden_size < 1:
        raise SystemExit("--batch-size and --hidden-size must be positive")
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit("--iterations must be positive and --warmup non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this PyTorch environment")
    if not torch.cuda.is_available() or not _TRITON_AVAILABLE:
        raise SystemExit("This benchmark requires CUDA and the optional Triton package")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("This benchmark requires a CUDA device")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("The selected CUDA device does not report BF16 support; try --dtype fp16")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    batch, hidden = args.batch_size, args.hidden_size
    sub_dt, min_tau = 1.0, 0.05
    probe = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; dtype={args.dtype}; B={batch}; H={hidden}")
    print(f"Runs: warmup={args.warmup}; measured={args.iterations}; scope=tail forward + backward")

    for multi_tau in (False, True):
        tau_width = hidden * (3 if multi_tau else 1)
        mix_w = torch.softmax(torch.randn(3, device=device, dtype=torch.float32), dim=0) if multi_tau else None
        values: tuple[torch.Tensor, ...] = (
            torch.randn(batch, hidden, device=device, dtype=dtype),
            torch.tanh(torch.randn(batch, hidden, device=device, dtype=dtype)),
            torch.randn(batch, tau_width, device=device, dtype=dtype) * 0.7,
            mix_w if mix_w is not None else torch.empty(0, device=device),
        )
        # Keep a fixed four-argument signature; the single-tau case ignores the empty mix tensor.
        torch_values = clone_leaves(values)
        triton_values = clone_leaves(values)
        if multi_tau:
            def torch_fn(h, cand, raw, mix):
                return make_reference(sub_dt, min_tau, True)(h, cand, raw, mix)
            def triton_fn(h, cand, raw, mix):
                return _ltc_tail_triton_autograd(h, cand, raw, mix, sub_dt, min_tau, True)
        else:
            def torch_fn(h, cand, raw, mix):
                return make_reference(sub_dt, min_tau, False)(h, cand, raw, None)
            def triton_fn(h, cand, raw, mix):
                return _ltc_tail_triton_autograd(h, cand, raw, None, sub_dt, min_tau, False)

        expected, expected_grads = evaluate_once(torch_fn, torch_values, probe)
        actual, actual_grads = evaluate_once(triton_fn, triton_values, probe)
        out_error = max_abs_diff(actual, expected)
        missing_grads = [
            name for name, actual_grad, expected_grad in zip(
                ("h", "candidate", "tau_raw", "mix_w"), actual_grads, expected_grads,
            ) if (actual_grad is None) != (expected_grad is None)
        ]
        if missing_grads:
            print(f"CHECK FAILED: gradient presence differs for {missing_grads}")
            return 2
        grad_errors = [
            max_abs_diff(a, b)
            for a, b in zip(actual_grads, expected_grads)
            if a is not None and b is not None
        ]
        max_grad_error = max(grad_errors, default=0.0)

        # Tolerances are intentionally looser for low-precision state tensors.
        tol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 5e-4
        if not math.isfinite(out_error) or not math.isfinite(max_grad_error):
            raise SystemExit("Non-finite difference detected")
        if out_error > tol or max_grad_error > tol * 4:
            print(f"  {'multi-tau' if multi_tau else 'single-tau'}: CHECK FAILED "
                  f"(max_output_err={out_error:.3g}, max_grad_err={max_grad_error:.3g})")
            return 2

        torch_us = time_forward_backward(
            torch_fn, torch_values, probe, args.warmup, args.iterations
        )
        triton_us = time_forward_backward(
            triton_fn, triton_values, probe, args.warmup, args.iterations
        )
        speedup = torch_us / triton_us if triton_us > 0 else float("inf")
        print(
            f"  {'multi-tau' if multi_tau else 'single-tau'}: PASS | "
            f"max output diff={out_error:.3g}; max grad diff={max_grad_error:.3g} | "
            f"PyTorch={torch_us:.1f} us; Triton={triton_us:.1f} us; "
            f"speedup={speedup:.2f}x"
        )

    print("NOTE: this measures only the LTC tail kernel, not full-model tokens/sec.")
    print("For the actual decision, also compare identical one-epoch runs with and without --kernel triton.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
