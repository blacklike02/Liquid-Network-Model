"""Reproducible CPU audit benchmark; timings exclude imports and model setup.

Example: python benchmark_model.py --threads 1 --output runs/benchmark.json
Use --source-root to measure a saved copy of the package with the same workload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--generation-hidden-size", type=int, default=384)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or args.warmup < 0 or args.generation_hidden_size < 1:
        parser.error("threads/repeats must be positive and warmup nonnegative")
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    import torch.nn.functional as F
    from liquid_lm.model import LiquidLanguageModel
    from liquid_lm.tokenizer import CharTokenizer
    from liquid_lm.training import kd_loss_topk

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        return {"median_ms": statistics.median(samples), "samples_ms": samples}

    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "torch": str(torch.__version__),
        "device": "cpu", "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "seed": 1234,
        "warmup": args.warmup, "repeats": args.repeats,
        "source_root": str(args.source_root.resolve()), "cases": {},
    }
    for gate, wiring in (("ltc", "dense"), ("cfc", "dense")):
        torch.manual_seed(1234)
        model = LiquidLanguageModel(
            512, embedding_dim=64, hidden_size=64, num_layers=2,
            dropout=0., zoneout=0., gate_mode=gate, wiring=wiring,
        ).train()
        tokens = torch.randint(0, 512, (4, 32))
        targets = torch.randint(0, 512, (4, 32))

        def train_forward_backward():
            model.zero_grad(set_to_none=True)
            logits, _ = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, 512), targets.reshape(-1))
            loss.backward()
            return loss

        loss = train_forward_backward()
        case = measure(train_forward_backward)
        case.update({
            "batch_size": 4, "seq_len": 32, "hidden_size": 64, "num_layers": 2,
            "vocab_size": 512, "loss": loss.item(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "gradient_checksum": sum(p.grad.double().sum().item() for p in model.parameters() if p.grad is not None),
            "tokens_per_second": 128000 / case["median_ms"],
            "scope": "model forward + CE + backward; no optimizer or data loading",
        })
        result["cases"][f"train_{gate}_{wiring}"] = case

    torch.manual_seed(1234)
    model = LiquidLanguageModel(
        8000, embedding_dim=args.generation_hidden_size,
        hidden_size=args.generation_hidden_size, num_layers=2, dropout=0., zoneout=0.,
    ).eval()
    chars = [chr(0x100 + i) for i in range(8000)]
    tokenizer = CharTokenizer({ch: i for i, ch in enumerate(chars)}, chars)
    prompt = "".join(chars[i % 20] for i in range(256))
    for new_tokens in (1, 32):
        def generate():
            return model.generate(
                prompt, tokenizer, max_new_tokens=new_tokens, top_k=1,
                top_p=1., repetition_penalty=1., repetition_window=0,
            )

        max_head_elements = [0]

        def record_head(module, inputs, output):
            max_head_elements[0] = max(max_head_elements[0], output.numel())

        with model.readout.register_forward_hook(record_head):
            generated = generate()
        case = measure(generate)
        case.update({
            "prompt_tokens": 256, "new_tokens": new_tokens, "vocab_size": 8000,
            "hidden_size": args.generation_hidden_size, "num_layers": 2,
            "output_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
            "max_readout_elements": max_head_elements[0],
            "scope": "full generate including prompt, sampling, decoding",
        })
        result["cases"][f"generate_{new_tokens}"] = case

    torch.manual_seed(1234)
    student = torch.randn(128, 8000, requires_grad=True)
    teacher = torch.randn(128, 8000)

    def distillation():
        student.grad = None
        loss = kd_loss_topk(student, teacher, 2., 32).mean()
        loss.backward()
        return loss

    kd = measure(distillation)
    kd.update({"positions": 128, "vocab_size": 8000, "k": 32, "temperature": 2.,
               "scope": "KD loss + backward; excludes model"})
    result["cases"]["kd_topk"] = kd
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: round(case["median_ms"], 3) for name, case in result["cases"].items()}))


if __name__ == "__main__":
    main()
