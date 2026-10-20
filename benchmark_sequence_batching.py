"""Paired full-step benchmark for opt-in sequence batching (no external data)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch

from benchmark_compute import autocast, check_equivalence, sync
from liquid_lm.model import LiquidLanguageModel
from liquid_lm.tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--kernel", choices=("torch", "triton"), default="triton")
    parser.add_argument("--cases", nargs="+", choices=("train", "ffn", "regularized", "generate"), default=["train", "ffn", "generate"])
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "bf16", "fp16"), default=["fp32"])
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--identical-control", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/sequence_batching/paired.json"))
    args = parser.parse_args()
    if min(args.pairs, args.hidden_size, args.vocab_size) < 1 or args.warmup < 0:
        parser.error("pairs/dimensions must be positive; warmup must be nonnegative")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"torch": torch.__version__, "device": str(device), "kernel": args.kernel,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "tf32": False, "threads": 1, "warmups": args.warmup,
              "identical_control": args.identical_control, "cases": []}
    for case in args.cases:
        for precision in args.precisions:
            torch.manual_seed(2026)
            base = LiquidLanguageModel(args.vocab_size, args.hidden_size, args.hidden_size, 2,
                                        dropout=0.05 if case == "regularized" else 0,
                                        zoneout=0.1 if case == "regularized" else 0,
                                        ffn_mult=1 if case == "ffn" else 0).to(device)
            models = [base, copy.deepcopy(base)]
            models[1].set_sequence_batching(not args.identical_control)
            for model in models:
                if args.kernel == "triton" and model.set_kernel("triton") != "triton":
                    raise RuntimeError("Requested Triton backend is unavailable")
            tokens = torch.randint(args.vocab_size, (16, 64), device=device)
            targets = torch.randint(args.vocab_size, tokens.shape, device=device)
            parity = check_equivalence(models, tokens, targets, device, precision, 420)
            optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False) for model in models]
            scalers = [torch.amp.GradScaler(device.type, enabled=precision == "fp16", init_scale=1024) for _ in models]
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            if case == "generate":
                chars = [chr(0x100 + i) for i in range(args.vocab_size)]
                tokenizer = CharTokenizer({c: i for i, c in enumerate(chars)}, chars)
                prompt = "".join(chars[i % min(20, args.vocab_size)] for i in range(128))
                for model in models:
                    model.eval()

            def step(index):
                if case == "generate":
                    with autocast(device, precision):
                        return models[index].generate(prompt, tokenizer, max_new_tokens=32,
                                                      top_k=1, top_p=1, repetition_penalty=1, device=device)
                optimizers[index].zero_grad(set_to_none=True)
                with autocast(device, precision):
                    logits, _ = models[index](tokens)
                    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
                scalers[index].scale(loss).backward()
                scalers[index].step(optimizers[index])
                scalers[index].update()
                return loss.detach()

            if case == "generate":
                assert step(0) == step(1), "Greedy generation changed"
            for iteration in range(args.warmup):
                for index in (0, 1):
                    torch.manual_seed(1000 + iteration)
                    step(index)
            pairs = []
            for iteration in range(args.pairs):
                pair = {}
                for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    torch.manual_seed(2000 + iteration)
                    sync(device)
                    start = time.perf_counter()
                    value = step(index)
                    sync(device)
                    pair["before_ms" if index == 0 else "after_ms"] = (time.perf_counter() - start) * 1000
                    if case != "generate" and not torch.isfinite(value):
                        raise RuntimeError("Nonfinite training loss")
                pair["speedup"] = pair["before_ms"] / pair["after_ms"]
                pairs.append(pair)
            row = {"case": case, "precision": precision, "parameters": sum(p.numel() for p in base.parameters()),
                   "sequence_path_selected": [m._can_batch_sequence(64) for m in models],
                   "parity": parity, "pairs": pairs,
                   "before_median_ms": statistics.median(p["before_ms"] for p in pairs),
                   "after_median_ms": statistics.median(p["after_ms"] for p in pairs),
                   "median_paired_speedup": statistics.median(p["speedup"] for p in pairs)}
            if case != "generate":
                expected = args.warmup + args.pairs
                for optimizer in optimizers:
                    steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
                    assert steps and min(steps) == max(steps) == expected, "Optimizer update was skipped"
                row["optimizer_updates"] = expected
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                peaks = []
                for index in (0, 1):
                    baseline = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    step(index)
                    sync(device)
                    peaks.append(torch.cuda.max_memory_allocated(device) - baseline)
                    optimizers[index].zero_grad(set_to_none=True)
                row["peak_additional_live_cuda_bytes"] = peaks
                row["peak_measurement_optimizer_updates"] = 0 if case == "generate" else 1
            result["cases"].append(row)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            print(case, precision, "before", round(row["before_median_ms"], 2), "after", round(row["after_median_ms"], 2),
                  "paired speedup", round(row["median_paired_speedup"], 3), flush=True)


if __name__ == "__main__":
    main()
