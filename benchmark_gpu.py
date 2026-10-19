"""Compare CPU/CUDA Liquid LM workloads without changing the model architecture.

Examples (use the same Python/PyTorch environment for both devices):
    python benchmark_gpu.py --device both --output runs/gpu_benchmark.json
    python benchmark_gpu.py --device cuda --kernel triton --quick --output runs/triton.json
    python benchmark_gpu.py --device cpu --cases train_small --precisions fp32 --quick

Timings exclude imports, model construction and input transfers. Training includes
forward, cross entropy, backward and AdamW. Generation includes prompt processing,
sampling and decoding. Every CUDA timing is synchronized at both boundaries.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from liquid_lm.model import LiquidLanguageModel
from liquid_lm.tokenizer import CharTokenizer


WORKLOADS = {
    "train_small": {"kind": "training", "batch_size": 8, "seq_len": 32,
                    "hidden_size": 128, "vocab_size": 512},
    "train_realistic8": {"kind": "training", "batch_size": 8, "seq_len": 64,
                         "hidden_size": 384, "vocab_size": 8000},
    "train_realistic16": {"kind": "training", "batch_size": 16, "seq_len": 64,
                          "hidden_size": 384, "vocab_size": 8000},
    "train_regularized16": {"kind": "training", "batch_size": 16, "seq_len": 64,
                           "hidden_size": 384, "vocab_size": 8000,
                           "dropout": 0.05, "zoneout": 0.1},
    "generate64": {"kind": "generation", "batch_size": 1, "prompt_tokens": 64,
                   "new_tokens": 32, "hidden_size": 384, "vocab_size": 8000},
    "generate128": {"kind": "generation", "batch_size": 1, "prompt_tokens": 128,
                    "new_tokens": 32, "hidden_size": 384, "vocab_size": 8000},
}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--output", type=Path, default=Path("runs/gpu_benchmark.json"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Two measurements, one warmup; small/train B8/generate64 cases")
    parser.add_argument("--cases", nargs="+", choices=tuple(WORKLOADS), default=None)
    parser.add_argument("--precisions", nargs="+", choices=tuple(DTYPES),
                        default=["fp32", "bf16", "fp16"])
    parser.add_argument("--cpu-precisions", nargs="+", choices=tuple(DTYPES), default=None,
                        help="CPU precisions; with --device both defaults to fp32 to avoid CPU AMP emulation")
    parser.add_argument("--kernel", choices=("torch", "triton"), default="torch",
                        help="CUDA backend; CPU always uses the explicit torch reference")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the CUDA recurrent step; CPU remains eager; incompatible with Triton")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.repeats = args.repeats if args.repeats is not None else (2 if args.quick else 5)
    args.warmup = args.warmup if args.warmup is not None else (1 if args.quick else 2)
    if args.threads < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("threads/repeats must be positive and warmup nonnegative")
    if args.compile and args.kernel == "triton":
        parser.error("--compile is incompatible with --kernel triton")
    if args.compile and args.device == "cpu":
        parser.error("--compile requires --device cuda or --device both")
    if args.cases is None:
        args.cases = (["train_small", "train_realistic8", "generate64"]
                      if args.quick else list(WORKLOADS))
    args.cases = list(dict.fromkeys(args.cases))
    args.precisions = list(dict.fromkeys(args.precisions))
    if args.cpu_precisions is not None:
        args.cpu_precisions = list(dict.fromkeys(args.cpu_precisions))
    return args


def model_config(workload: dict) -> dict:
    return {
        "vocab_size": workload["vocab_size"],
        "embedding_dim": workload["hidden_size"],
        "hidden_size": workload["hidden_size"], "num_layers": 2,
        "ode_unfolds": 1, "dropout": workload.get("dropout", 0.0),
        "zoneout": workload.get("zoneout", 0.0),
        "tie_weights": True, "mlp_head": False, "residual": True,
        "multi_tau": True, "local_conv": True, "use_checkpoint": False,
        "gate_mode": "ltc", "ode_solver": "euler", "wiring": "dense",
        "ncp_command_frac": 0.4, "ncp_motor_frac": 0.2,
        "memory_slots": 0, "ffn_mult": 0.0, "norm_type": "layer",
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=DTYPES[precision])


def measure(fn, device: torch.device, repeats: int, warmup: int) -> tuple[dict, object]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    memory = None
    if device.type == "cuda":
        memory = {"baseline_allocated_bytes": torch.cuda.memory_allocated(device),
                  "baseline_reserved_bytes": torch.cuda.memory_reserved(device)}
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    last = None
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        last = fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    if memory is not None:
        memory.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                       "final_allocated_bytes": torch.cuda.memory_allocated(device),
                       "final_reserved_bytes": torch.cuda.memory_reserved(device)})
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples), "samples_ms": samples, "memory": memory}, last


def finite_tensors(tensors) -> bool:
    flags = [torch.isfinite(t.detach()).all() for t in tensors]
    return bool(torch.stack(flags).all().item()) if flags else False


def optimizer_steps(optimizer) -> tuple[int, int]:
    values = [int(state["step"].item()) for state in optimizer.state.values()
              if "step" in state]
    return (min(values), max(values)) if values else (0, 0)


def benchmark_training(model, workload, device, precision, args) -> dict:
    # Generate on CPU before transfer so CPU/CUDA consume exactly the same inputs.
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    shape = (workload["batch_size"], workload["seq_len"])
    tokens = torch.randint(workload["vocab_size"], shape, generator=generator).to(device)
    targets = torch.randint(workload["vocab_size"], shape, generator=generator).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01,
                                 betas=(0.9, 0.999), eps=1e-8, foreach=False)
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16",
                                 init_scale=1024.0, growth_interval=2000)
    model.train()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits, _ = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, workload["vocab_size"]),
                                   targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    before = next(model.parameters()).detach().clone()
    first_loss = step()
    first_loss_value = float(first_loss.item())
    initial_steps = optimizer_steps(optimizer)
    initial_grads_finite = finite_tensors(p.grad for p in model.parameters() if p.grad is not None)
    parameter_update = float((next(model.parameters()).detach() - before).abs().max().item())
    del before, first_loss
    if not initial_grads_finite or not parameter_update or initial_steps != (1, 1):
        raise RuntimeError("Initial optimizer step failed numerical validation: "
                           f"finite_grads={initial_grads_finite}, max_update={parameter_update}, "
                           f"optimizer_steps={initial_steps}")
    start_steps = optimizer_steps(optimizer)
    result, last_loss = measure(step, device, args.repeats, args.warmup)
    end_steps = optimizer_steps(optimizer)
    expected = args.warmup + args.repeats
    completed = tuple(end - start for start, end in zip(start_steps, end_steps))
    loss_value = float(last_loss.item())
    grads_finite = finite_tensors(p.grad for p in model.parameters() if p.grad is not None)
    params_finite = finite_tensors(model.parameters())
    if not grads_finite or not params_finite or not torch.isfinite(last_loss).item():
        raise RuntimeError("Nonfinite loss, gradients or parameters after measured training")
    if completed != (expected, expected):
        raise RuntimeError(f"GradScaler skipped optimizer steps: {completed}; expected {expected}")
    result.update({
        "scope": "preloaded synthetic tokens: forward + CE + backward + AdamW; no data loading",
        "tokens_per_second": shape[0] * shape[1] * 1000.0 / result["median_ms"],
        "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.01,
                      "betas": [0.9, 0.999], "eps": 1e-8, "foreach": False},
        "grad_scaler": {"enabled": scaler.is_enabled(), "initial_scale": 1024.0,
                        "final_scale": scaler.get_scale(), "growth_interval": 2000},
        "validation": {"initial_loss": first_loss_value, "final_loss": loss_value,
                       "finite_gradients": grads_finite, "finite_parameters": params_finite,
                       "first_parameter_max_update": parameter_update,
                       "optimizer_steps_min_max": list(end_steps),
                       "expected_optimizer_steps": 1 + expected},
    })
    return result


def benchmark_generation(model, workload, device, precision, args) -> dict:
    chars = [chr(0x100 + i) for i in range(workload["vocab_size"])]
    tokenizer = CharTokenizer({ch: i for i, ch in enumerate(chars)}, chars)
    prompt = "".join(chars[i % 20] for i in range(workload["prompt_tokens"]))
    generation = {"max_new_tokens": workload["new_tokens"], "temperature": 1.0,
                  "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0,
                  "repetition_window": 0, "only_new": False, "stop_token_ids": None,
                  "memory_chunk_size": 128}
    model.eval()

    def generate():
        with torch.inference_mode(), autocast(device, precision):
            return model.generate(prompt, tokenizer, device=device, **generation)

    generated = generate()
    expected_length = workload["prompt_tokens"] + workload["new_tokens"]
    if len(generated) != expected_length or not generated.startswith(prompt):
        raise RuntimeError("Generation changed the prompt or returned the wrong token count")
    result, last = measure(generate, device, args.repeats, args.warmup)
    if last != generated:
        raise RuntimeError("Greedy generation changed between identical benchmark calls")
    result.update({
        "scope": "full generate: prompt prefill + 32 generated tokens + sampling + decoding",
        "new_tokens_per_second": workload["new_tokens"] * 1000.0 / result["median_ms"],
        "generation": generation,
        "validation": {"output_tokens": len(last), "prompt_preserved": True,
                       "repeatable_output": True,
                       "output_sha256": hashlib.sha256(last.encode("utf-8")).hexdigest()},
    })
    return result


def run_case(case: dict, args) -> None:
    device = torch.device(case["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python/PyTorch environment")
        if case["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support BF16 autocast")
    torch.manual_seed(args.seed)
    # Initialization on CPU makes starting weights identical across device cases.
    model = LiquidLanguageModel(**case["model_config"]).to(device)
    requested = args.kernel if device.type == "cuda" else "torch"
    selected = model.set_kernel(requested)
    actual = model._kernel
    cell_kernels = [cell.kernel for cell in model.cells]
    case["kernel_requested_for_device"] = requested
    case["kernel_actual"] = actual
    case["cell_kernels_actual"] = cell_kernels
    if selected != requested or actual != requested or any(k != requested for k in cell_kernels):
        raise RuntimeError(f"Requested {requested} backend fell back to model={actual}, "
                           f"cells={cell_kernels}; timing rejected")
    compile_requested = args.compile and device.type == "cuda"
    case["compile"] = {"requested": compile_requested, "enabled": False, "actual_mode": None}
    if compile_requested:
        # Cases are independent models/precisions. Do not consume one shared
        # Dynamo specialization budget or silently time eager after its limit.
        import torch._dynamo as dynamo
        dynamo.reset()
        dynamo.utils.counters.clear()
        dynamo.config.fail_on_recompile_limit_hit = True
        case["compile"]["runtime_recompile_limit_policy"] = "error"
        synchronize(device)
        compile_start = time.perf_counter()
        compiled, mode = model.enable_compiled_step(
            batch_size=case["workload"]["batch_size"],
            inference=case["workload"]["kind"] == "generation",
        )
        synchronize(device)
        case["compile"].update({"enabled": compiled, "actual_mode": mode,
                                "setup_ms": (time.perf_counter() - compile_start) * 1000.0})
        if not compiled or model._step_fn == model._step_impl:
            raise RuntimeError("Compiled recurrent step is unavailable; eager fallback timing rejected")
    case["parameters"] = sum(p.numel() for p in model.parameters())
    fn = benchmark_training if case["workload"]["kind"] == "training" else benchmark_generation
    # Untimed initial validation in fn compiles any AMP-specific runtime graphs.
    case.update(fn(model, case["workload"], device, case["precision"], args))
    if compile_requested:
        case["compile"]["graph_stats"] = dict(dynamo.utils.counters["stats"])
        if not case["compile"]["graph_stats"].get("unique_graphs", 0):
            raise RuntimeError("No compiled runtime graph observed; timing rejected")


def environment(include_cuda: bool) -> dict:
    result = {
        "python_executable": sys.executable, "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "torch": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_fp32_precision": getattr(torch.backends.cuda.matmul, "fp32_precision", None),
        "cudnn_fp32_precision": getattr(torch.backends.cudnn, "fp32_precision", None),
        "cudnn_conv_fp32_precision": getattr(torch.backends.cudnn.conv, "fp32_precision", None),
        "cudnn_rnn_fp32_precision": getattr(torch.backends.cudnn.rnn, "fp32_precision", None),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "torch_build_config": torch.__config__.show(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_devices": [],
    }
    if include_cuda and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["cuda_devices"].append({
                "index": index, "name": props.name,
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory,
                "multiprocessors": props.multi_processor_count,
            })
    return result


def comparisons(cases: list[dict]) -> list[dict]:
    output = []
    for gpu in cases:
        if gpu["device"] != "cuda" or gpu["status"] != "ok":
            continue
        cpu = next((case for case in cases if case["device"] == "cpu"
                    and case["name"] == gpu["name"] and case["precision"] == gpu["precision"]
                    and case["status"] == "ok"), None)
        if cpu is not None:
            output.append({"name": gpu["name"], "precision": gpu["precision"],
                           "cpu_median_ms": cpu["median_ms"], "cuda_median_ms": gpu["median_ms"],
                           "cuda_speedup_over_cpu": cpu["median_ms"] / gpu["median_ms"],
                           "cpu_kernel": cpu["kernel_actual"], "cuda_kernel": gpu["kernel_actual"]})
    return output


def write_report(path: Path, result: dict) -> None:
    result["comparisons"] = comparisons(result["cases"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    result = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment("cuda" in devices),
              "settings": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
              "notes": ["Synthetic inputs; excludes model setup, transfers and data loading.",
                        "FP32 master parameters; bf16/fp16 use autocast; fp16 uses GradScaler.",
                        "Dropout and zoneout disabled for comparable CPU/CUDA workloads.",
                        "Models start from identical CPU-initialized weights and CPU-generated inputs.",
                        "FP32 baseline disables TF32 (IEEE); the project CLI can enable TF32 separately.",
                        "CUDA cases run sequentially; wall timings synchronize both boundaries.",
                        "Generation throughput includes prompt processing; it is not decode-only.",
                        "CPU reference always uses torch, including when CUDA requests Triton."],
              "cases": []}
    write_report(args.output, result)
    failures = 0
    for device_name in devices:
        precisions = args.precisions
        if device_name == "cpu":
            precisions = args.cpu_precisions or (["fp32"] if args.device == "both" else args.precisions)
        for name in args.cases:
            for precision in precisions:
                case = {"name": name, "device": device_name, "precision": precision,
                        "workload": dict(WORKLOADS[name]), "model_config": model_config(WORKLOADS[name]),
                        "status": "running"}
                result["cases"].append(case)
                print(f"[START] {name} {device_name} {precision} kernel="
                      f"{args.kernel if device_name == 'cuda' else 'torch'}", flush=True)
                write_report(args.output, result)
                try:
                    run_case(case, args)
                    case["status"] = "ok"
                    print(f"[OK] {name} {device_name} {precision}: "
                          f"{case['median_ms']:.3f} ms; actual_kernel={case['kernel_actual']}", flush=True)
                except Exception as exc:
                    failures += 1
                    case["status"] = "error"
                    case["error"] = {"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc()}
                    print(f"[ERROR] {name} {device_name} {precision}: {type(exc).__name__}: {exc}",
                          flush=True)
                finally:
                    # run_case's local model/optimizer have left scope before cache cleanup.
                    gc.collect()
                    if device_name == "cuda" and torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        except Exception as exc:
                            case["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                            if case["status"] != "error":
                                case["status"] = "error"
                                failures += 1
                    write_report(args.output, result)
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["failures"] = failures
    result["status"] = "error" if failures else "ok"
    write_report(args.output, result)
    print(f"Report: {args.output.resolve()}; failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
