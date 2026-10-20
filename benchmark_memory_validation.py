"""Ablations, stress tasks and equal-time controls for associative side memory.

Generates all examples in memory and keeps compact JSON only. Reference ablations
live here, not in checkpoint architecture options. No language-quality claims.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from benchmark_innovation import recall_batch, synchronize
from liquid_lm.associative import InnovationMemory
from liquid_lm.model import LiquidLanguageModel


class DeltaControl(InnovationMemory):
    """Exact ordinary delta writes: keep all other gates, scales and weights."""

    def _write_novelty(self, error):
        return torch.ones_like(error[:, :1])


class ScalarDecayControl(InnovationMemory):
    """Same row rates/projections, pooled to one shared decay per example."""

    def _decay_rate(self, controls):
        return super()._decay_rate(controls).mean(dim=-1, keepdim=True)


def overwrite_batch(generator, batch_size, delay):
    tokens, old = recall_batch(generator, batch_size, 0)
    key = tokens[:, -1:]
    new = 11 + ((old - 11 + torch.randint(1, 8, (batch_size,), generator=generator)) % 8)
    write = torch.cat((torch.ones_like(key), key, new[:, None]), dim=-1)
    noise = torch.randint(19, 27, (batch_size, delay), generator=generator)
    return torch.cat((tokens[:, :-2], write, noise, tokens[:, -2:]), dim=1), new, old


def multiquery_batch(generator, batch_size, delay):
    tokens, _ = recall_batch(generator, batch_size, delay)
    keys = tokens[:, :12].reshape(batch_size, 4, 3)[:, :, 1]
    values = tokens[:, :12].reshape(batch_size, 4, 3)[:, :, 2]
    order = torch.rand(batch_size, 4, generator=generator).argsort(-1)
    queries = torch.stack((torch.full_like(keys, 2), keys.gather(1, order)), dim=-1).flatten(1)
    context = tokens[:, :-2]
    positions = torch.arange(context.size(1) + 1, context.size(1) + 8, 2)
    return torch.cat((context, queries), dim=1), values.gather(1, order), positions


def make_models(seed, hidden_size=64, memory_dim=16):
    common = dict(vocab_size=27, embedding_dim=hidden_size, hidden_size=hidden_size,
                  num_layers=1, dropout=0., zoneout=0., local_conv=False,
                  tie_weights=False, residual=True)
    torch.manual_seed(seed)
    baseline = LiquidLanguageModel(**common)
    shared = {k: v.clone() for k, v in baseline.state_dict().items()}
    torch.manual_seed(seed)
    memory = LiquidLanguageModel(**common, innovation_memory_dim=memory_dim)
    added = sum(p.numel() for p in memory.parameters()) - sum(p.numel() for p in baseline.parameters())
    inner = max(8, round((added - 2 * hidden_size) / (3 * hidden_size) / 8) * 8)
    torch.manual_seed(seed)
    ffn = LiquidLanguageModel(**common, ffn_mult=inner / hidden_size)
    state = ffn.state_dict()
    state.update(shared)
    ffn.load_state_dict(state)
    # Construct the ablations from the identical complete memory initialization.
    models = {'baseline': baseline, 'parameter_control': ffn, 'innovation': memory}
    for name, cls in [('delta_control', DeltaControl), ('scalar_decay', ScalarDecayControl)]:
        model = copy.deepcopy(memory)
        replacement = cls(hidden_size, memory_dim)
        replacement.load_state_dict(memory.innovation_memory.state_dict())
        model.innovation_memory = replacement
        models[name] = model
    return models


def make_evaluation(device, count, delays):
    batches = {}
    for delay in delays:
        generator = torch.Generator().manual_seed(90000 + delay)
        # Keep the first 512 examples identical to the original experiment.
        batches[f'recall_{delay}'] = [tuple(t.to(device) for t in recall_batch(generator, 128, delay))
                                      for _ in range(count // 128)]
    generator = torch.Generator().manual_seed(81001)
    batches['overwrite_8'] = [tuple(t.to(device) for t in overwrite_batch(generator, 128, 8))
                              for _ in range(count // 128)]
    generator = torch.Generator().manual_seed(81002)
    batches['multiquery_8'] = [tuple(t.to(device) for t in multiquery_batch(generator, 128, 8))
                               for _ in range(count // 128)]
    return batches


@torch.inference_mode()
def evaluate(model, tasks):
    model.eval()
    scores = {}
    for name, batches in tasks.items():
        loss, correct, total, old_hits = 0., 0, 0, 0
        by_query = torch.zeros(4, dtype=torch.long)
        for batch in batches:
            tokens, targets = batch[:2]
            multi = name.startswith('multiquery')
            logits, hidden = model(tokens, last_token_only=not multi)
            assert all(torch.isfinite(h).all() for h in hidden)
            if multi:
                last = logits[:, batch[2]].float()
            else:
                last = logits[:, -1].float()
            loss += F.cross_entropy(last.reshape(-1, 27), targets.flatten(), reduction='sum').item()
            hit = last.argmax(-1) == targets
            correct += hit.sum().item()
            total += targets.numel()
            if multi:
                by_query += hit.sum(0).cpu()
            if name.startswith('overwrite'):
                old_hits += (last.argmax(-1) == batch[2]).sum().item()
        scores[name] = {'accuracy': correct / total, 'nll': loss / total, 'targets': total}
        if name.startswith('overwrite'):
            scores[name]['obsolete_value_accuracy'] = old_hits / total
        if name.startswith('multiquery'):
            scores[name]['accuracy_by_query'] = (by_query.double() / (total / 4)).tolist()
    model.train()
    return scores


def train(model, seed, device, args):
    model.to(device).train()
    if args.kernel == 'triton' and model.set_kernel('triton') != 'triton':
        raise RuntimeError('Requested Triton backend unavailable')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    generator = torch.Generator().manual_seed(10000 + seed)
    # Generation/transfers are outside the measured budget, identical fixed data
    # prefixes across arms. Time-limited arms may consume different prefix lengths.
    count = args.steps if args.seconds <= 0 else args.max_updates
    batches = [tuple(t.to(device) for t in recall_batch(generator, args.batch_size, 8)) for _ in range(count)]
    for _ in range(3):
        logits, _ = model(batches[0][0], last_token_only=True)
        F.cross_entropy(logits[:, -1], batches[0][1]).backward()
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    step_ms = []
    for step, (tokens, targets) in enumerate(batches, 1):
        synchronize(device)
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(tokens, last_token_only=True)
        loss = F.cross_entropy(logits[:, -1], targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        step_ms.append(1000 * (time.perf_counter() - step_start))
        elapsed = time.perf_counter() - start
        if args.seconds > 0 and elapsed >= args.seconds:
            break
        if step % 200 == 0:
            print(f'  update={step} loss={loss.item():.4f}', flush=True)
    if args.seconds > 0 and elapsed < args.seconds:
        raise RuntimeError('max-updates reached before the requested time budget')
    result = {'updates': step, 'seconds': elapsed, 'median_step_ms': statistics.median(step_ms),
              'training_examples': step * args.batch_size, 'last_train_nll': loss.item(),
              'last_grad_norm': norm.item()}
    if device.type == 'cuda':
        result['training_peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--kernel', choices=['torch', 'triton'], default='torch')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--seconds', type=float, default=0., help='>0: equal measured training seconds per arm')
    parser.add_argument('--max-updates', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-examples', type=int, default=512)
    parser.add_argument('--delays', type=int, nargs='+', default=[8, 32, 64, 128, 256])
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--arms', nargs='+', choices=['baseline','parameter_control','innovation','delta_control','scalar_decay'],
                        default=['baseline','parameter_control','innovation','delta_control','scalar_decay'])
    parser.add_argument('--output', type=Path, default=Path('runs/innovation_memory/validation.json'))
    args = parser.parse_args()
    if min(args.steps, args.max_updates, args.batch_size, args.eval_examples) < 1 or args.eval_examples % 128:
        parser.error('positive sizes required; eval-examples must be divisible by 128')
    if args.seconds < 0 or not all(d >= 0 for d in args.delays):
        parser.error('seconds and delays must be nonnegative')
    if args.kernel == 'triton' and args.device != 'cuda':
        parser.error('Triton requires CUDA')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tasks = make_evaluation(device, args.eval_examples, args.delays)
    results = {'config': {**vars(args), 'output': str(args.output)}, 'torch': str(torch.__version__),
               'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
               'arms': [], 'protocol': 'fixed final weights; identical initial shared weights/data prefixes; no checkpoint selection'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(args.seeds):
        models = make_models(seed)
        # Rotate arm order across seeds; equal-time results remain sensitive to
        # background load. No other benchmark should run simultaneously.
        names = args.arms[index % len(args.arms):] + args.arms[:index % len(args.arms)]
        for name in names:
            print(f'{name} seed={seed} budget={args.seconds or args.steps}', flush=True)
            model = models[name]
            row = {'arm': name, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters())}
            row.update(train(model, seed, device, args))
            row['evaluation'] = evaluate(model, tasks)
            results['arms'].append(row)
            args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
            model.to('cpu')


if __name__ == '__main__':
    main()
