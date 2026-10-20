"""Controlled synthetic associative-recall experiment; no downloaded datasets.

Example: python benchmark_innovation.py --device cuda --steps 400
All arms see identical batches and start with identical shared weights. This is
a diagnostic of associative recall, not evidence of language-model quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from liquid_lm.model import LiquidLanguageModel


def recall_batch(generator, batch_size, delay, pairs=4):
    """Four unique key/value pairs, distractors, QUERY key -> associated value.

    Keys: 3..10, values: 11..18, distractors: 19..26. No answer appears after
    QUERY. Each example's mapping is independent of all other examples.
    """
    keys = torch.rand(batch_size, 8, generator=generator).argsort(dim=-1)[:, :pairs] + 3
    values = torch.randint(11, 19, (batch_size, pairs), generator=generator)
    triplets = torch.stack((torch.ones_like(keys), keys, values), dim=-1).flatten(1)
    noise = torch.randint(19, 27, (batch_size, delay), generator=generator)
    selected = torch.randint(pairs, (batch_size, 1), generator=generator)
    query = torch.cat((torch.full((batch_size, 1), 2), keys.gather(1, selected)), dim=1)
    tokens = torch.cat((triplets, noise, query), dim=1)
    return tokens, values.gather(1, selected).squeeze(1)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def evaluate(model, batches):
    model.eval()
    losses, hits, count = 0., 0, 0
    with torch.inference_mode():
        for tokens, targets in batches:
            logits, _ = model(tokens, last_token_only=True)
            last = logits[:, -1].float()
            losses += F.cross_entropy(last, targets, reduction='sum').item()
            hits += (last.argmax(-1) == targets).sum().item()
            count += targets.numel()
    model.train()
    return {'nll': losses / count, 'accuracy': hits / count, 'examples': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--kernel', choices=['torch', 'triton'], default='torch')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--memory-dim', type=int, default=16)
    parser.add_argument('--delay', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--output', type=Path, default=Path('runs/innovation_memory/results.json'))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.hidden_size, args.memory_dim) < 1 or args.delay < 0:
        parser.error('steps, batch-size, hidden-size and memory-dim must be positive; delay >= 0')
    if args.kernel == 'triton' and args.device != 'cuda':
        parser.error('Triton requires CUDA')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {'config': {**vars(args), 'output': str(args.output)}, 'torch': str(torch.__version__),
               'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
               'task': '4 key/value pairs; random values; held-out random examples; final-query CE only',
               'value_chance_accuracy': 0.125, 'arms': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    for seed in args.seeds:
        common = dict(vocab_size=27, embedding_dim=args.hidden_size, hidden_size=args.hidden_size,
                      num_layers=1, dropout=0., zoneout=0., local_conv=False,
                      tie_weights=False, residual=True)
        torch.manual_seed(seed)
        baseline = LiquidLanguageModel(**common)
        shared = {key: value.clone() for key, value in baseline.state_dict().items()}
        base_count = sum(p.numel() for p in baseline.parameters())
        torch.manual_seed(seed)
        experimental = LiquidLanguageModel(**common, innovation_memory_dim=args.memory_dim)
        added = sum(p.numel() for p in experimental.parameters()) - base_count
        # SwiGLU adds 3*H*inner + norm parameters. Match as closely as its
        # existing multiple-of-eight width constraint allows.
        inner = max(8, round((added - 2 * args.hidden_size) / (3 * args.hidden_size) / 8) * 8)
        torch.manual_seed(seed)
        control = LiquidLanguageModel(**common, ffn_mult=inner / args.hidden_size)
        arms = [('baseline', baseline), ('parameter_control', control), ('innovation', experimental)]
        validation = {}
        for delay in sorted({args.delay, 32, 64}):
            generator = torch.Generator().manual_seed(90000 + delay)
            validation[delay] = [tuple(t.to(device) for t in recall_batch(generator, 128, delay))
                                 for _ in range(4)]
        for name, model in arms:
            state = model.state_dict()
            for key, value in shared.items():
                state[key] = value
            model.load_state_dict(state)
            model.to(device)
            if args.kernel == 'triton':
                if model.set_kernel('triton') != 'triton':
                    raise RuntimeError('Requested Triton backend unavailable')
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            generator = torch.Generator().manual_seed(10000 + seed)
            train = [tuple(t.to(device) for t in recall_batch(generator, args.batch_size, args.delay))
                     for _ in range(args.steps)]
            # Warm up with backward but no optimizer update and preserve weights.
            for _ in range(3):
                logits, _ = model(train[0][0], last_token_only=True)
                F.cross_entropy(logits[:, -1], train[0][1]).backward()
                optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            start = time.perf_counter()
            step_ms = []
            for step, (tokens, targets) in enumerate(train, 1):
                synchronize(device)
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(tokens, last_token_only=True)
                loss = F.cross_entropy(logits[:, -1], targets)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.step()
                synchronize(device)
                step_ms.append((time.perf_counter() - step_start) * 1000)
                if step % 100 == 0:
                    print(f'{name} seed={seed} step={step}/{args.steps} loss={loss.item():.4f}', flush=True)
            synchronize(device)
            seconds = time.perf_counter() - start
            row = {'arm': name, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
                   'updates': args.steps, 'training_examples': args.steps * args.batch_size,
                   'seconds': seconds, 'median_step_ms': statistics.median(step_ms),
                   'last_train_nll': loss.item(), 'last_grad_norm': norm.item(),
                   'validation': {str(delay): evaluate(model, batches) for delay, batches in validation.items()}}
            if device.type == 'cuda':
                row['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
            results['arms'].append(row)
            print(json.dumps(row), flush=True)
            save()
            # No experimental checkpoint or dataset is retained.
            model.to('cpu')
            del optimizer, train
    save()


if __name__ == '__main__':
    main()
