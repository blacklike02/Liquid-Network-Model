"""Measure context use and compare training horizons without changing the model.

The two arms consume exactly the same token/target windows at every update.
Short-horizon batches reset state at each short chunk; long-horizon batches
backpropagate through the entire window. Checkpoints are inference-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from liquid_lm.data import BOT_TAG, SYSTEM_TAG, USER_TAG, load_texts
from liquid_lm.model import create_model_from_config, load_checkpoint_raw
from liquid_lm.tokenizer import encode_corpus, tokenizer_from_checkpoint
from liquid_lm.training import EMA, evaluate, get_batch
from liquid_lm.utils import configure_tensor_cores, set_seed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def context_curve(model, data, split, seed=7341, batch_size=64, batches=8):
    """Score identical final targets with different amounts of preceding input."""
    model.eval()
    gen = torch.Generator(device=data.device).manual_seed(seed)
    lengths = (8, 16, 32, 64, 128, 256)
    per_window = {n: [] for n in lengths}
    predictions = {n: [] for n in lengths}
    for _ in range(batches):
        x, y = get_batch(data, batch_size, max(lengths), split, len(data), gen=gen)
        for n in lengths:
            logits, _ = model(x[:, -n:], last_token_only=True)
            losses = F.cross_entropy(logits[:, -1], y[:, -1], reduction="none")
            per_window[n].append(losses.cpu())
            predictions[n].append(logits[:, -1].cpu())
    reference = torch.cat(per_window[max(lengths)]).double()
    ref_logp = torch.cat(predictions[max(lengths)]).double().log_softmax(-1)
    result = {}
    for n in lengths:
        losses = torch.cat(per_window[n]).double()
        delta = losses - reference
        logp = torch.cat(predictions[n]).double().log_softmax(-1)
        kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).clamp_min(0.)
        result[str(n)] = {
            "nll": losses.mean().item(),
            "delta_vs_256": delta.mean().item(),
            "paired_standard_error": (delta.std() / math.sqrt(len(delta))).item(),
            "mean_kl_full_vs_short": kl.mean().item(),
        }
    return {"seed": seed, "targets": len(reference), "input_lengths": result}


@torch.inference_mode()
def tau_diagnostics(model, data, split):
    """Observe effective tau, not a claim about the full recurrent Jacobian."""
    if model.gate_mode != "ltc":
        return None
    model.eval()
    raw = [[] for _ in model.cells]
    handles = []
    for i, cell in enumerate(model.cells):
        handles.append(cell.tau_net[2].register_forward_hook(
            lambda _module, _inputs, output, i=i: raw[i].append(output.detach())
        ))
    try:
        gen = torch.Generator(device=data.device).manual_seed(7342)
        x, _ = get_batch(data, 32, 128, split, len(data), gen=gen)
        model(x, last_token_only=True)
    finally:
        for handle in handles:
            handle.remove()
    result = []
    for cell, outputs in zip(model.cells, raw):
        values = torch.cat(outputs).float()
        mix = None
        if cell.multi_tau:
            values = values.reshape(-1, 3, cell.hidden_size)
            offsets = values.new_tensor([-2., 0., 2.])[None, :, None]
            floors = values.new_tensor([.05, .01, .001])[None, :, None]
            mix = cell.tau_mix.softmax(0)
            tau = ((F.softplus(values + offsets) + floors) * mix[None, :, None]).sum(1)
        else:
            tau = F.softplus(values) + cell.min_tau
        quantiles = torch.quantile(tau.flatten(), tau.new_tensor([.1, .5, .9])).tolist()
        result.append({
            "mixture_weights": mix.tolist() if mix is not None else None,
            "tau_p10_p50_p90": quantiles,
            "mean_frozen_decay_dt1": torch.exp(-1. / tau).mean().item(),
        })
    return result


def validate(model, data, split):
    return {
        "native_64": evaluate(model, data, 128, 64, 1, data.device,
                              split, len(data), 8, None),
        "sequence_128": evaluate(model, data, 64, 128, 1, data.device,
                                 split, len(data), 8, None),
    }


def average_checkpoints(paths, source, source_hash):
    """Uniform mean of compatible descendants of one source checkpoint."""
    if len(paths) < 2 or len(set(p.resolve() for p in paths)) != len(paths):
        raise ValueError("averaging requires at least two distinct checkpoints")
    payloads = [load_checkpoint_raw(path, "cpu") for path in paths]
    reference = source["model_state"]
    for payload in payloads:
        if (payload["config"] != source["config"]
                or payload["tokenizer"] != source["tokenizer"]
                or payload.get("corpus_sha256") != source.get("corpus_sha256")
                or payload.get("experiment", {}).get("source_sha256") != source_hash
                or payload["model_state"].keys() != reference.keys()):
            raise ValueError("averaging requires matching config/tokenizer/corpus/source/keys")
    averaged = {}
    for key, tensor in reference.items():
        values = [p["model_state"][key] for p in payloads]
        if any(v.shape != tensor.shape or v.dtype != tensor.dtype for v in values):
            raise ValueError(f"incompatible tensor: {key}")
        if tensor.is_floating_point():
            averaged[key] = torch.stack([v.double() for v in values]).mean(0).to(tensor.dtype)
            if not torch.isfinite(averaged[key]).all():
                raise ValueError(f"nonfinite averaged tensor: {key}")
        else:
            if not all(torch.equal(values[0], v) for v in values[1:]):
                raise ValueError(f"incompatible nonfloating buffer: {key}")
            averaged[key] = values[0].clone()
    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/shakespeare_2m5/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data_shakespeare"))
    parser.add_argument("--output", type=Path, default=Path("runs/liquid_research"))
    parser.add_argument("--steps", type=int, default=0, help="0: diagnostics only")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    parser.add_argument("--short-seq", type=int, default=64)
    parser.add_argument("--long-seq", type=int, default=128)
    parser.add_argument("--tokens-per-step", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--kernel", choices=["torch", "triton"], default="triton")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--average-checkpoints", type=Path, nargs="+",
                        help="average compatible experiment checkpoints; no training")
    args = parser.parse_args()
    if not (0 < args.short_seq < args.long_seq and args.long_seq % args.short_seq == 0):
        parser.error("long-seq must be a multiple of short-seq and larger")
    if args.steps < 0 or args.tokens_per_step % args.long_seq or args.lr <= 0:
        parser.error("invalid steps, tokens-per-step or lr")
    if args.tokens_per_step <= 0 or not math.isfinite(args.lr):
        parser.error("tokens-per-step and finite lr must be positive")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be distinct and nonnegative")
    if args.average_checkpoints and args.steps:
        parser.error("averaging and training are separate experiments")
    planned_names = ["results.json", "diagnostics.json", "averaged_results.json", "averaged.pt"]
    planned_names.extend(f"seed{seed}_seq{seq}.pt" for seed in args.seeds
                         for seq in (args.short_seq, args.long_seq))
    protected_inputs = {p.resolve() for p in [args.checkpoint, *(args.average_checkpoints or [])]}
    if any((args.output / name).resolve() in protected_inputs for name in planned_names):
        parser.error("output would overwrite an input checkpoint")
    if not torch.cuda.is_available():
        parser.error("this GPU experiment requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    configure_tensor_cores(device)
    source_hash = sha256(args.checkpoint)
    source = load_checkpoint_raw(args.checkpoint, "cpu")
    tokenizer = tokenizer_from_checkpoint(source)
    cfg = source["config"]
    # These conditions keep this harness's loss/state handling unambiguous.
    if cfg.get("memory_slots", 0):
        parser.error("this experiment does not implement ChunkMemory")
    corpus = load_texts(args.data, min_chars=1000,
                        shuffle_files=cfg.get("shuffle_files", False),
                        file_headers=cfg.get("file_headers", True))
    corpus_hash = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    if corpus_hash != source.get("corpus_sha256"):
        parser.error("corpus fingerprint differs from the source checkpoint")
    encoded = encode_corpus(tokenizer, corpus)
    if any((encoded == tokenizer.token_to_id(tag)).any()
           for tag in (USER_TAG, SYSTEM_TAG, BOT_TAG)
           if tokenizer.token_to_id(tag) is not None):
        parser.error("use unmasked plain text for this experiment")
    data = torch.as_tensor(encoded, device=device, dtype=torch.long)
    split = int(len(data) * (1. - cfg["val_split"]))

    def fresh_model():
        model = create_model_from_config(tokenizer.vocab_size, cfg, device)
        model.load_state_dict(source["model_state"], strict=True)
        actual = model.set_kernel(args.kernel)
        if actual != args.kernel:
            raise RuntimeError(f"requested {args.kernel}, got {actual}")
        return model

    model = fresh_model()
    result = {
        "source_checkpoint": str(args.checkpoint), "source_sha256": source_hash,
        "corpus_sha256": corpus_hash, "parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "config": vars(args).copy(),
        "baseline": validate(model, data, split), "arms": [],
        "evaluation": {"seed": 1234, "batches": 8, "tokens_per_protocol": 65536},
        "method": "same long windows; short chunks reset state; new AdamW in both arms; final EMA only",
    }
    for key, value in result["config"].items():
        if isinstance(value, Path):
            result["config"][key] = str(value)
        elif key == "average_checkpoints" and value:
            result["config"][key] = [str(p) for p in value]
    if not args.skip_diagnostics:
        result["context_curve"] = context_curve(model, data, split)
        result["tau_diagnostics"] = tau_diagnostics(model, data, split)
    if args.average_checkpoints:
        path = args.output / "averaged.pt"
        protected = [args.checkpoint, *args.average_checkpoints]
        if path.resolve() in {p.resolve() for p in protected}:
            raise ValueError("output would overwrite an input checkpoint")
        input_hashes = [sha256(p) for p in args.average_checkpoints]
        averaged = average_checkpoints(args.average_checkpoints, source, source_hash)
        model.load_state_dict(averaged, strict=True)
        metrics = validate(model, data, split)
        if not args.skip_diagnostics:
            result["averaged_context_curve"] = context_curve(model, data, split)
        torch.save({
            "model_state": averaged, "config": cfg, "tokenizer": tokenizer.to_config(),
            "corpus_sha256": corpus_hash, "val_loss": metrics["native_64"],
            "experiment": {"inference_only": True, "method": "uniform weight mean",
                           "source_sha256": source_hash, "input_sha256": input_hashes},
        }, path)
        loaded = load_checkpoint_raw(path, device)
        model.load_state_dict(loaded["model_state"], strict=True)
        verified = validate(model, data, split)
        if max(abs(metrics[k] - verified[k]) for k in metrics) > 1e-5:
            raise RuntimeError("averaged checkpoint validation failed")
        result["averaged_validation"] = metrics
        result["averaged_checkpoint"] = str(path)
        result["reload_verified"] = True
        result["weights_finite"] = True
        result["inputs_unchanged"] = input_hashes == [sha256(p) for p in args.average_checkpoints]
        result["source_unchanged"] = sha256(args.checkpoint) == source_hash
        if not result["inputs_unchanged"] or not result["source_unchanged"]:
            raise RuntimeError("an input checkpoint changed")
        (args.output / "averaged_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return
    print(json.dumps({k: v for k, v in result.items() if k not in ("config", "arms")}), flush=True)
    del model
    output_path = args.output / ("results.json" if args.steps else "diagnostics.json")

    def persist():
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    persist()
    for seed in args.seeds if args.steps else []:
        for seq in (args.short_seq, args.long_seq):
            model = fresh_model()
            set_seed(seed)
            gen = torch.Generator(device=device).manual_seed(seed + 10000)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                         betas=(.9, .95), weight_decay=.01, fused=True)
            ema = EMA(model, .98)
            model.train()
            losses = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for step in range(args.steps):
                x, y = get_batch(data, args.tokens_per_step // args.long_seq,
                                 args.long_seq, 0, split, gen=gen)
                if seq != args.long_seq:
                    x = x.reshape(-1, seq).contiguous()
                    y = y.reshape(-1, seq).contiguous()
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                ema.update(model)
                losses.append(loss.detach())
                if (step + 1) % 25 == 0:
                    print(f"seed={seed} seq={seq} step={step+1}/{args.steps} loss={loss.item():.5f}", flush=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            raw_metrics = validate(model, data, split)
            ema.copy_to(model)
            metrics = validate(model, data, split)
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise RuntimeError("nonfinite final weights")
            steps = [int(state["step"].item()) for state in optimizer.state.values()]
            if not steps or min(steps) != args.steps or max(steps) != args.steps:
                raise RuntimeError("optimizer step counts disagree")
            path = args.output / f"seed{seed}_seq{seq}.pt"
            torch.save({
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": cfg, "tokenizer": tokenizer.to_config(),
                "corpus_sha256": corpus_hash,
                "val_loss": metrics["native_64"],
                "experiment": {"inference_only": True, "seed": seed, "seq_len": seq,
                               "updates": args.steps, "source_sha256": source_hash,
                               "optimizer": "new AdamW", "lr": args.lr, "ema_decay": .98},
            }, path)
            # Recreate through the public checkpoint/model path and reproduce loss.
            reloaded = load_checkpoint_raw(path, device)
            model.load_state_dict(reloaded["model_state"], strict=True)
            reload_metrics = validate(model, data, split)
            if max(abs(metrics[k] - reload_metrics[k]) for k in metrics) > 1e-5:
                raise RuntimeError("checkpoint validation failed")
            arm = {
                "seed": seed, "seq_len": seq, "updates": args.steps,
                "tokens_seen": args.steps * args.tokens_per_step, "seconds": seconds,
                "tokens_per_second": args.steps * args.tokens_per_step / seconds,
                "peak_allocated_mb": peak_mb, "train_nll": torch.stack(losses).mean().item(),
                "raw_validation": raw_metrics, "ema_validation": metrics,
                "checkpoint": str(path), "reload_verified": True,
                "weights_finite": True, "optimizer_steps_min_max": [min(steps), max(steps)],
            }
            result["arms"].append(arm)
            print(json.dumps(arm), flush=True)
            persist()
            del reloaded, optimizer, ema, model, logits, loss, losses
            torch.cuda.empty_cache()
    result["source_unchanged"] = sha256(args.checkpoint) == source_hash
    if not result["source_unchanged"]:
        raise RuntimeError("source checkpoint changed")
    persist()


if __name__ == "__main__":
    main()
