"""Paired eager/CUDA Graph timings of the ordinary generation API.

Warm timings include prompt processing, memory, sampling, CPU token reads and
decoding. Weight loading and first-use capture are reported separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from liquid_lm.model import LiquidLanguageModel, create_model_from_config, load_inference_checkpoint
from liquid_lm.tokenizer import CharTokenizer, tokenizer_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--recurrent-passes", type=int, default=1)
    parser.add_argument("--memory-slots", type=int, default=3)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--kernel", choices=["torch", "triton"], default="triton")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--output", type=Path, default=Path("runs/cuda_graphs/generation.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.new_tokens < 2 or args.chunk_size < 1:
        parser.error("pairs/chunk-size must be positive and new-tokens at least two")
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch and a CUDA GPU are required")
    if args.hidden < 1 or not 128 <= args.vocab <= 8192:
        parser.error("hidden must be positive; synthetic character vocabulary must be 128..8192")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(271)
    device = torch.device("cuda")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    started = time.perf_counter()
    if args.checkpoint:
        payload = load_inference_checkpoint(args.checkpoint)
        tokenizer = tokenizer_from_checkpoint(payload)
        eager = create_model_from_config(tokenizer.vocab_size, payload["config"], device).eval()
        eager.load_state_dict(payload.pop("model_state"), strict=True)
        del payload
    else:
        tokenizer = CharTokenizer.train("".join(chr(i+32) for i in range(args.vocab)))
        eager = LiquidLanguageModel(tokenizer.vocab_size, args.hidden, args.hidden, 2,
                       recurrent_passes=args.recurrent_passes, dropout=0, zoneout=0,
                       ffn_mult=1, memory_slots=args.memory_slots, memory_dim=args.memory_dim).to(device).eval()
    eager.set_sequence_batching(True)
    if eager.set_kernel(args.kernel) != args.kernel:
        raise RuntimeError("Requested kernel is unavailable; refusing fallback timing")
    graphed = copy.deepcopy(eager)
    graphed.set_cuda_graphs(True)
    torch.cuda.synchronize()
    loading_seconds = time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats()
    prompt = "Write a short story about a dog."
    options = dict(prompt=prompt, tokenizer=tokenizer, max_new_tokens=args.new_tokens,
                   memory_chunk_size=args.chunk_size, temperature=.8, top_k=40, top_p=.95,
                   repetition_penalty=1.05, repetition_window=128, device=device,
                   stop_token_ids=[], only_new=True)

    def measure(model, seed):
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype!=torch.float32):
            text = model.generate(**options)
        torch.cuda.synchronize()
        return time.perf_counter()-start, text, torch.cuda.get_rng_state()

    cold_seconds, cold_text, cold_rng = measure(graphed, 981)
    _, reference_text, reference_rng = measure(eager, 981)
    if cold_text != reference_text or not torch.equal(cold_rng, reference_rng):
        raise AssertionError("First generation/RNG differs")
    samples = {"eager": [], "cuda_graphs": []}
    for pair in range(args.pairs):
        order = (("eager", eager), ("cuda_graphs", graphed))
        if pair%2: order = order[::-1]
        outputs = []
        for name, model in order:
            seconds, text, rng = measure(model, 1017+pair)
            samples[name].append(seconds)
            outputs.append((text, rng))
        if outputs[0][0] != outputs[1][0] or not torch.equal(outputs[0][1], outputs[1][1]):
            raise AssertionError(f"Paired generation/RNG differs in round {pair}")
    status = graphed.cuda_graph_stats()
    if status["fallbacks"] or not status["captures"] or not status["replays"]:
        raise RuntimeError(f"No valid graph result: {status}")
    eager_seconds = statistics.median(samples["eager"])
    graph_seconds = statistics.median(samples["cuda_graphs"])
    result = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  precision=args.precision, tf32=False, cpu_threads=torch.get_num_threads(),
                  parameters=sum(p.numel() for p in eager.parameters()), kernel=args.kernel,
                  recurrent_passes=eager.recurrent_passes, memory_slots=eager.memory_slots,
                  memory_dim=eager.memory_dim, chunk_size=args.chunk_size, new_tokens=args.new_tokens,
                  prompt_tokens=len(tokenizer.encode(prompt)), pairs=args.pairs,
                  loaded_checkpoint=args.checkpoint is not None, loading_and_copy_seconds=loading_seconds,
                  first_graph_generation_seconds=cold_seconds,
                  warm_eager_seconds=eager_seconds, warm_graph_seconds=graph_seconds,
                  warm_eager_output_tokens_per_second=args.new_tokens/eager_seconds,
                  warm_graph_output_tokens_per_second=args.new_tokens/graph_seconds,
                  paired_speedup=statistics.median(a/b for a,b in zip(samples["eager"],samples["cuda_graphs"])),
                  samples_seconds=samples, identical_outputs_and_rng=True, graph_stats=status,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  generated_sample_sha256=hashlib.sha256(cold_text.encode()).hexdigest(),
                  timing_scope="Generation API including prompt, boundary memory work, sampling, token CPU reads and decode; warm timings exclude loading/capture",
                  limitations="Sequential GPU workloads; unlocked clocks and Windows scheduling; same-weight model copies resident for both modes; no network serving benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
