"""GPU correctness audit for Triton tails and AMP recurrent model gradients.

Run only while other GPU audit workloads are idle. This script measures
correctness, not speed. It does not modify production code or model architecture.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# The audit may run in a sandbox where the profile's default .triton cache is
# read-only. An explicit cache choice from the caller still takes precedence.
os.environ.setdefault("TRITON_CACHE_DIR", str(REPO_ROOT / "runs" / "triton_cache"))

import torch
import torch.nn.functional as F

import liquid_lm.layers as layers
from liquid_lm.model import LiquidLanguageModel


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
# These are explicitly AMP approximation tolerances, not an equivalence claim.
# max_abs <= atol + rtol * max(abs(reference)); relative L2 must also meet
# its threshold unless the whole difference already meets the absolute floor.
TOLERANCES = {
    "tail": {
        "fp32": {"atol": 2e-5, "rtol": 2e-4, "relative_l2": 5e-4},
        "bf16": {"atol": 3e-3, "rtol": 4e-2, "relative_l2": 6e-2},
        "fp16": {"atol": 5e-4, "rtol": 8e-3, "relative_l2": 1.5e-2},
    },
    "model": {
        "fp32": {"atol": 2e-5, "rtol": 1e-3, "relative_l2": 3e-3},
        "bf16": {"atol": 2e-3, "rtol": 1e-1, "relative_l2": 1.5e-1},
        "fp16": {"atol": 3e-4, "rtol": 2.5e-2, "relative_l2": 4e-2},
    },
}


def compare(actual, reference, tolerance):
    if actual is None or reference is None:
        # Every listed input/parameter is expected to participate in the graph.
        return {"passed": False, "actual_missing": actual is None,
                "reference_missing": reference is None}
    if actual.shape != reference.shape:
        return {"passed": False, "actual_shape": list(actual.shape),
                "reference_shape": list(reference.shape)}
    actual_cpu = actual.detach().cpu().double()
    reference_cpu = reference.detach().cpu().double()
    finite = bool(torch.isfinite(actual_cpu).all() and torch.isfinite(reference_cpu).all())
    result = {"shape": list(reference.shape), "actual_dtype": str(actual.dtype),
              "reference_dtype": str(reference.dtype), "finite": finite}
    if not finite:
        return {**result, "passed": False}
    difference = actual_cpu - reference_cpu
    max_abs = difference.abs().max().item() if difference.numel() else 0.0
    reference_max = reference_cpu.abs().max().item() if reference_cpu.numel() else 0.0
    ref_norm = torch.linalg.vector_norm(reference_cpu).item()
    diff_norm = torch.linalg.vector_norm(difference).item()
    relative_l2 = diff_norm / max(ref_norm, 1e-30)
    allowed_max = tolerance["atol"] + tolerance["rtol"] * reference_max
    passed = max_abs <= allowed_max and (
        relative_l2 <= tolerance["relative_l2"] or max_abs <= tolerance["atol"]
    )
    return {**result, "max_abs_diff": max_abs, "relative_l2_diff": relative_l2,
            "reference_max_abs": reference_max, "reference_l2": ref_norm,
            "actual_l2": torch.linalg.vector_norm(actual_cpu).item(),
            "allowed_max_abs": allowed_max, "passed": passed}


def reference_tail(h, candidate, raw, mix, sub_dt, min_tau, multi_tau):
    if multi_tau:
        banks = raw.view(h.shape[0], 3, h.shape[-1])
        fast = F.softplus(banks[:, 0, :] - 2.0) + 0.05
        medium = F.softplus(banks[:, 1, :]) + 0.01
        slow = F.softplus(banks[:, 2, :] + 2.0) + 0.001
        tau = mix[0] * fast + mix[1] * medium + mix[2] * slow
    else:
        tau = F.softplus(raw) + min_tau
    decay = torch.exp(-sub_dt / tau)
    return decay * h + (1.0 - decay) * candidate


def run_tail_case(dtype_name, multi_tau, extreme, shape, noncontiguous):
    batch, hidden = shape
    dtype = DTYPES[dtype_name]
    width = hidden * (3 if multi_tau else 1)
    torch.manual_seed(120)
    # AMP uses FP32 hidden state and FP32 mix weights, but low precision linear
    # outputs. In FP32 mode all inputs are FP32.
    h = torch.randn(hidden, batch, device="cuda", dtype=torch.float32).T
    candidate = torch.randn(hidden, batch, device="cuda", dtype=dtype).T.tanh()
    raw = torch.randn(width, batch, device="cuda", dtype=dtype).T * 0.7
    if extreme:
        extremes = torch.tensor([-80., -25., -20., -4., -2., 0., 2., 20., 25., 80.],
                                device="cuda", dtype=dtype)
        raw = extremes[torch.arange(batch * width, device="cuda") % len(extremes)]
        raw = raw.reshape(batch, width)
    if noncontiguous:
        # Ensure all three inputs have real noncontiguous strides even after
        # pointwise ops / indexing above.
        h = h.T.contiguous().T
        candidate = candidate.T.contiguous().T
        raw = raw.T.contiguous().T
    else:
        h, candidate, raw = (v.contiguous() for v in (h, candidate, raw))
    mix = torch.softmax(torch.tensor([-3.0, 0.2, 2.0], device="cuda"), dim=0)
    values = (h, candidate, raw, mix) if multi_tau else (h, candidate, raw)
    leaves_reference = [v.detach().clone().requires_grad_(True) for v in values]
    leaves_actual = [v.detach().clone().requires_grad_(True) for v in values]
    probe = torch.randn(batch, hidden, device="cuda", dtype=torch.float32)
    sub_dt = 0.25 if extreme else 1.0

    def evaluate(fn, leaves):
        mix_arg = leaves[3] if multi_tau else None
        output = fn(*leaves[:3], mix_arg, sub_dt, 0.05, multi_tau)
        (output.float() * probe).sum().backward()
        return output, [value.grad for value in leaves]

    expected, expected_grads = evaluate(reference_tail, leaves_reference)
    actual, actual_grads = evaluate(layers._ltc_tail_triton_autograd, leaves_actual)
    tolerance = TOLERANCES["tail"][dtype_name]
    metrics = {"output": compare(actual, expected, tolerance)}
    for name, actual_grad, reference_grad in zip(
        ("h", "candidate", "tau_raw", "mix_w"), actual_grads, expected_grads
    ):
        metrics["grad_" + name] = compare(actual_grad, reference_grad, tolerance)
    return {
        "case": f"tail_{dtype_name}_multi{int(multi_tau)}_extreme{int(extreme)}_{batch}x{hidden}",
        "dtype": dtype_name, "multi_tau": multi_tau, "extreme_softplus": extreme,
        "sub_dt": sub_dt, "shape": list(shape), "noncontiguous": noncontiguous,
        "input_dtypes": [str(v.dtype) for v in leaves_actual],
        "input_strides": [list(v.stride()) for v in leaves_actual],
        "mix_weights": mix.cpu().tolist() if multi_tau else None,
        "metrics": metrics, "passed": all(v["passed"] for v in metrics.values()),
    }


def run_model_case(dtype_name, multi_tau, seq_len, hidden_size):
    torch.manual_seed(180)
    reference = LiquidLanguageModel(
        vocab_size=97, embedding_dim=hidden_size, hidden_size=hidden_size,
        num_layers=2, ode_unfolds=1, dropout=0., zoneout=0., tie_weights=True,
        multi_tau=multi_tau, local_conv=True, gate_mode="ltc", ode_solver="euler",
    ).to("cuda").train()
    if multi_tau:
        with torch.no_grad():
            for cell in reference.cells:
                cell.tau_mix.copy_(torch.tensor([-3., 0.2, 2.], device="cuda"))
    actual = copy.deepcopy(reference)
    assert reference.set_kernel("torch") == "torch"
    assert actual.set_kernel("triton") == "triton", "Triton silently fell back"
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in reference.state_dict().items())
    tokens = torch.randint(0, 97, (2, seq_len), device="cuda")
    targets = torch.randint(0, 97, (2, seq_len), device="cuda")
    tail_input_dtypes = set()
    triton_calls = 0
    original_tail = layers._ltc_tail_triton_autograd

    def tracked_tail(h, candidate, raw, mix, *args):
        nonlocal triton_calls
        triton_calls += 1
        tail_input_dtypes.add((str(h.dtype), str(candidate.dtype), str(raw.dtype),
                               str(mix.dtype) if mix is not None else None))
        return original_tail(h, candidate, raw, mix, *args)

    def evaluate(model, record):
        context = (torch.autocast("cuda", dtype=DTYPES[dtype_name])
                   if dtype_name != "fp32" else nullcontext())
        tracking = patch.object(layers, "_ltc_tail_triton_autograd", tracked_tail) if record else nullcontext()
        with tracking, context:
            logits, hidden = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, 97), targets.reshape(-1))
        loss.backward()
        return logits, hidden, loss

    expected_logits, expected_hidden, expected_loss = evaluate(reference, False)
    actual_logits, actual_hidden, actual_loss = evaluate(actual, True)
    tolerance = TOLERANCES["model"][dtype_name]
    metrics = {"logits": compare(actual_logits, expected_logits, tolerance),
               "loss": compare(actual_loss, expected_loss, tolerance)}
    for index, (actual_state, expected_state) in enumerate(zip(actual_hidden, expected_hidden)):
        metrics[f"hidden_{index}"] = compare(actual_state, expected_state, tolerance)
    reference_parameters = dict(reference.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    assert reference_parameters.keys() == actual_parameters.keys()
    for name in reference_parameters:
        metrics["grad_" + name] = compare(
            actual_parameters[name].grad, reference_parameters[name].grad, tolerance
        )
    return {
        "case": f"model_{dtype_name}_multi{int(multi_tau)}", "dtype": dtype_name,
        "multi_tau": multi_tau, "batch_size": 2, "seq_len": seq_len,
        "hidden_size": hidden_size, "num_layers": 2, "ode_unfolds": 1,
        "parameter_dtype": "torch.float32", "triton_tail_calls": triton_calls,
        "tail_input_dtype_combinations": sorted(tail_input_dtypes, key=str),
        "reference_loss": expected_loss.item(), "actual_loss": actual_loss.item(),
        "metrics": metrics,
        "passed": triton_calls > 0 and all(v["passed"] for v in metrics.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs" / "triton_correctness.json")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()
    if args.seq_len < 1 or args.hidden_size < 1:
        parser.error("sequence length and hidden size must be positive")
    if not torch.cuda.is_available() or not layers._TRITON_AVAILABLE:
        raise SystemExit("CUDA and Triton are required; no successful skip is permitted")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 is required for this audit; no successful skip is permitted")
    result = {"torch": str(torch.__version__), "device": torch.cuda.get_device_name(),
              "scope": "correctness only; no full-model speed claim", "tolerances": TOLERANCES,
              "criterion": "max_abs <= atol + rtol*reference_max AND (relative_l2 <= threshold OR max_abs <= atol); all gradients required",
              "cases": []}
    jobs = []
    for dtype_name in DTYPES:
        for multi_tau in (False, True):
            # B*H=1539 checks cross-CTA reduction and final partial CTA; H=513
            # also makes CTA boundaries cross row and tau-bank boundaries.
            for extreme in (False, True):
                jobs.append((f"tail_{dtype_name}_{multi_tau}_{extreme}", run_tail_case,
                             (dtype_name, multi_tau, extreme, (3, 513), extreme)))
            jobs.append((f"model_{dtype_name}_{multi_tau}", run_model_case,
                         (dtype_name, multi_tau, args.seq_len, args.hidden_size)))
    for name, fn, arguments in jobs:
        try:
            case = fn(*arguments)
        except Exception:
            case = {"case": name, "passed": False, "error": traceback.format_exc()}
        result["cases"].append(case)
        failures = [key for key, metric in case.get("metrics", {}).items() if not metric["passed"]]
        print(json.dumps({"case": case["case"], "passed": case["passed"],
                          "failed_metrics": failures, "error": case.get("error")}), flush=True)
        # Save after every case so a later device/compiler failure keeps results.
        result["passed"] = all(item["passed"] for item in result["cases"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"RESULT: {'PASS' if result['passed'] else 'FAIL'}; output={args.output}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
