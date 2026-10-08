"""Triton kernel for the tail of the ltc substep (inference only)."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


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
