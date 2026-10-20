"""Benchmark retained normalization/memory and a bounded delayed-recall task.

Results are workload-specific. Synthetic recall is not a conversational quality
benchmark. No downloaded data or existing checkpoint is modified.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from liquid_lm.layers import (ChunkMemory, CompactChunkMemory, RMSNorm,
                              chunk_memory_read, chunk_memory_reset, chunk_memory_write)
from liquid_lm.model import LiquidLanguageModel, detach_hidden


def timed(fn, device, repeats=7, inner=5):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(3):
        fn()
    samples = []
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        for _ in range(inner):
            fn()
        sync()
        samples.append((time.perf_counter() - started) / inner)
    return dict(median_ms=statistics.median(samples) * 1000,
                samples_ms=[value * 1000 for value in samples])


def performance(device):
    result = {"normalization": [], "memory": [], "full_model": []}
    for rows in (1, 256, 8192):
        x = torch.randn(rows, 384, device=device)
        native = RMSNorm(384).to(device)
        def old():
            xf = x.float()
            return xf * (xf.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * native.weight
        with torch.inference_mode():
            result["normalization"].append(dict(rows=rows,
                max_error=float((native(x) - old()).abs().max()),
                explicit=timed(old, device, inner=30),
                native=timed(lambda: native(x), device, inner=30)))
    for name, module in (("legacy", ChunkMemory(384, 4)),
                         ("compact", CompactChunkMemory(384, 4, 64))):
        module = module.to(device)
        state = torch.randn(16, 384, device=device)
        memory = module.init_memory(16, device)
        for _ in range(4):
            memory = module.write(torch.randn_like(state), memory)
        def train_read():
            module.zero_grad(set_to_none=True)
            module.read(state, memory).square().mean().backward()
        result["memory"].append(dict(name=name,
            parameters=sum(p.numel() for p in module.parameters()),
            buffer_bytes=memory.numel() * memory.element_size(),
            read_backward=timed(train_read, device)))
    workloads = []
    for norm, slots, width in (("rms", 0, 0), ("rms", 3, 64)):
        torch.manual_seed(14)
        model = LiquidLanguageModel(vocab_size=4096, embedding_dim=384, hidden_size=384,
            num_layers=2, dropout=0, zoneout=0, ffn_mult=1,
            norm_type=norm, memory_slots=slots, memory_dim=width).to(device)
        if device.type == "cuda":
            model.set_kernel("triton")
        model.set_sequence_batching(True)
        tokens = torch.randint(0, 4096, (8, 128), device=device)
        def train(model=model):
            model.zero_grad(set_to_none=True)
            hidden, memory = None, chunk_memory_reset(model, 8, device)
            # Four chunks exercise exactly the history used by the new defaults.
            for index in range(4):
                hidden = chunk_memory_read(model, hidden, memory)
                logits, hidden = model(tokens, hidden)
                F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
                if index < 3:
                    memory = chunk_memory_write(model, hidden, memory)
                hidden = detach_hidden(hidden)
        workloads.append((dict(norm=norm, slots=slots, memory_dim=width,
            parameters=sum(p.numel() for p in model.parameters())), train))
    for _, fn in workloads:
        fn()
    samples = [[] for _ in workloads]
    # Alternate direction to reduce clock/order bias. Loading and compilation
    # are excluded, but Windows scheduling and unlocked power remain variable.
    order = list(range(len(workloads)))
    for index in (order + list(reversed(order))) * 3:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        workloads[index][1]()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples[index].append((time.perf_counter() - started) * 1000)
    for (row, _), values in zip(workloads, samples):
        row["four_chunks_forward_backward"] = dict(median_ms=statistics.median(values),
            samples_ms=values, order="forward/reverse, three rounds")
        result["full_model"].append(row)
    return result


def recall_data(seed, batch, device):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 8, (batch,), generator=generator)
    tokens = torch.randint(10, 24, (batch, 32), generator=generator)
    tokens[:, 7] = labels  # Fact at the end of chunk 1.
    tokens[:, -1] = 8     # Query at the end of chunk 4.
    return tokens.to(device), labels.to(device)


def run_span(model, tokens, labels, train=False, corrupt=False):
    hidden = None
    memory = chunk_memory_reset(model, len(tokens), tokens.device)
    losses = []
    for index, chunk in enumerate(tokens.split(8, 1)):
        if corrupt and memory is not None and index:
            memory = memory.roll(1, 0)
        hidden = chunk_memory_read(model, hidden, memory)
        logits, hidden = model(chunk, hidden)
        # A small local copy objective trains the fact representation. Both
        # arms otherwise receive the same final delayed-answer objective.
        if index in (0, 3):
            loss = F.cross_entropy(logits[:, -1, :8], labels)
            loss = loss * (0.2 if index == 0 else 1.0)
            losses.append(float(loss.detach()))
            if train:
                loss.backward()
        if index < 3:
            memory = chunk_memory_write(model, hidden, memory)
        hidden = detach_hidden(hidden)
    return logits[:, -1, :8], sum(losses)


def learning(device, steps, seeds):
    results = []
    for seed in seeds:
        # Initialize the common core identically despite the added module.
        torch.manual_seed(seed)
        reference = LiquidLanguageModel(vocab_size=24, embedding_dim=32, hidden_size=32,
            num_layers=1, dropout=0, zoneout=0, local_conv=False,
            multi_tau=False, norm_type="rms").to(device)
        base_state = reference.state_dict()
        for enabled in (False, True):
            torch.manual_seed(seed)
            model = LiquidLanguageModel(vocab_size=24, embedding_dim=32, hidden_size=32,
                num_layers=1, dropout=0, zoneout=0, local_conv=False,
                multi_tau=False, norm_type="rms", memory_slots=3 if enabled else 0,
                memory_dim=8 if enabled else 0).to(device)
            model.load_state_dict(base_state, strict=False)
            if device.type == "cuda":
                model.set_kernel("triton")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
            started = time.perf_counter()
            for step in range(steps):
                tokens, labels = recall_data(seed * 10000 + step, 32, device)
                optimizer.zero_grad(set_to_none=True)
                _, loss = run_span(model, tokens, labels, train=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            tokens, labels = recall_data(9000000 + seed, 512, device)
            model.eval()
            with torch.inference_mode():
                logits, _ = run_span(model, tokens, labels)
                accuracy = float(logits.argmax(-1).eq(labels).float().mean())
                shuffled, _ = run_span(model, tokens, labels, corrupt=True)
                corrupted = float(shuffled.argmax(-1).eq(labels).float().mean())
            row = dict(seed=seed, compact_memory=enabled, steps=steps,
                       parameters=sum(p.numel() for p in model.parameters()),
                       training_seconds=seconds, final_training_loss=loss,
                       heldout_accuracy=accuracy, shuffled_memory_accuracy=corrupted)
            results.append(row)
            print(json.dumps(row), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[707, 808, 909])
    parser.add_argument("--skip-learning", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/rmsnorm_only/retained.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    result = dict(torch=torch.__version__, device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  cpu_threads=1, tf32=torch.backends.cuda.matmul.allow_tf32,
                  timing="warm wall time, synchronized; excludes loading/compilation",
                  learning_protocol="RMSNorm in both arms; four 8-token chunks, detached boundaries; 512 fresh examples per seed; "
                      "equal updates and common core initialization; unequal parameter/time budgets; "
                      "12.5% chance; synthetic recall is not chat quality")
    if not args.skip_performance:
        result["performance"] = performance(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Metadata/results saved", flush=True)
    if not args.skip_learning:
        result["learning"] = learning(device, args.steps, args.seeds)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
