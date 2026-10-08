"""Self-test of the triton kernel against the torch path."""

from __future__ import annotations

import time

import torch

from .model.liquid_cell import LiquidCell
from .model.triton_kernel import _TRITON_AVAILABLE


def selftest_triton_tail(device) -> None:
    """Compares the triton tail of the ltc substep with the regular torch path on
    random data (single-tau and multi-tau, including uneven bank weights and
    "hard" inputs for softplus) and measures timing. Run manually:
        python liquid_text_model.py --selftest-triton
    Requires CUDA and the triton package; otherwise honestly reports that
    there is nothing to check."""
    if not _TRITON_AVAILABLE:
        print("[SELFTEST] the triton package is not installed — nothing to check.")
        return
    if device.type != "cuda":
        print("[SELFTEST] A CUDA GPU is required — nothing to check.")
        return
    torch.manual_seed(0)
    B, H, IN = 32, 384, 384
    all_ok = True
    for multi_tau in (False, True):
        cell = LiquidCell(IN, H, ode_unfolds=1, multi_tau=multi_tau).to(device).eval()
        if multi_tau:
            cell.tau_mix.data = torch.randn(3, device=device)
        x = torch.randn(B, IN, device=device) * 3.0
        h = torch.randn(B, H, device=device)
        with torch.no_grad():
            xp, tx = cell.precompute_x(x)
            fw, fb = cell.fused_h_weights()
            cell.kernel = "torch"
            ref, _, _ = cell._ltc_step(h, xp, tx, 1.0, fw, fb)
            cell.kernel = "triton"
            if not cell._use_triton_tail(h):
                print("[SELFTEST] The triton path was not activated — check the conditions.")
                return
            got, _, _ = cell._ltc_step(h, xp, tx, 1.0, fw, fb)
            diff = (ref - got).abs().max().item()
            ok = diff < 1e-4
            all_ok &= ok

            def bench(kernel_name: str, iters: int = 300) -> float:
                cell.kernel = kernel_name
                for _ in range(20):
                    cell._ltc_step(h, xp, tx, 1.0, fw, fb)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    cell._ltc_step(h, xp, tx, 1.0, fw, fb)
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / iters * 1e3

            t_torch = bench("torch")
            t_triton = bench("triton")
        print(
            f"[SELFTEST] multi_tau={multi_tau}: max|diff|={diff:.2e} "
            f"{'OK' if ok else 'FAIL'} | torch={t_torch:.3f} ms/step, "
            f"triton={t_triton:.3f} ms/step"
        )
    print("[SELFTEST] RESULT:", "all checks passed" if all_ok else "DIFFERENCES FOUND — do not use --kernel triton")
