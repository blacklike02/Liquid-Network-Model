"""Small fixed-budget language A/B test on a supplied temporary text corpus.

Train/validation/test are contiguous 80/10/10 splits. Vocabulary is built only
from training. No checkpoint is selected or retained. No network access here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from torch.nn import functional as F

from benchmark_innovation import synchronize
from benchmark_memory_validation import DeltaControl
from liquid_lm.model import LiquidLanguageModel


def prepare_text(raw):
    text = raw.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    start = re.search(r'^\*\*\* START OF .*?\*\*\*\s*$', text, re.M)
    end = re.search(r'^\*\*\* END OF .*?\*\*\*\s*$', text, re.M)
    if start and end:
        text = text[start.end():end.start()].strip()
    return text


def make_splits(text):
    a, b = int(len(text) * 0.8), int(len(text) * 0.9)
    vocabulary = {char: i + 1 for i, char in enumerate(sorted(set(text[:a])))}
    splits = {}
    for name, part in [('train', text[:a]), ('validation', text[a:b]), ('test', text[b:])]:
        splits[name] = torch.tensor([vocabulary.get(char, 0) for char in part], dtype=torch.long)
    return vocabulary, splits


def train_batch(data, generator, batch_size, seq_len):
    starts = torch.randint(len(data) - seq_len, (batch_size, 1), generator=generator)
    offsets = torch.arange(seq_len)[None]
    return data[starts + offsets], data[starts + offsets + 1]


@torch.inference_mode()
def evaluate(model, data, seq_len, device):
    model.eval()
    starts = torch.arange(0, len(data) - seq_len, seq_len)
    loss, count = 0., 0
    for group in starts.split(16):
        offsets = torch.arange(seq_len)[None]
        x = data[group[:, None] + offsets].to(device)
        y = data[group[:, None] + offsets + 1].to(device)
        logits, _ = model(x)
        loss += F.cross_entropy(logits.float().reshape(-1, model.vocab_size), y.flatten(), reduction='sum').item()
        count += y.numel()
    model.train()
    nll = loss / count
    return {'nll': nll, 'perplexity': math.exp(nll), 'target_presentations': count,
            'unknown_characters_in_split': int((data == 0).sum()), 'split_characters': len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--arms', nargs='+', choices=['baseline','parameter_control','innovation','delta_control'],
                        default=['baseline','parameter_control','innovation','delta_control'])
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('runs/innovation_memory/language.json'))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.seq_len) < 1:
        parser.error('positive training sizes required')
    raw = args.data.read_bytes()
    text = prepare_text(raw)
    vocabulary, splits = make_splits(text)
    if min(len(data) for data in splits.values()) <= args.seq_len:
        parser.error('corpus splits are too short')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {'config': {**vars(args), 'data': str(args.data), 'output': str(args.output)},
               'torch': str(torch.__version__), 'source_sha256': hashlib.sha256(raw).hexdigest(),
               'corpus_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
               'characters': len(text), 'vocabulary_size': len(vocabulary) + 1,
               'split_sizes': {name: len(data) for name, data in splits.items()},
               'protocol': 'new random models; train-only char vocabulary; fixed final weights; no val/test selection',
               'arms': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        common = dict(vocab_size=len(vocabulary) + 1, embedding_dim=64, hidden_size=64,
                      num_layers=1, dropout=0., zoneout=0., local_conv=True, tie_weights=False)
        torch.manual_seed(seed)
        baseline = LiquidLanguageModel(**common)
        shared = {k: v.clone() for k, v in baseline.state_dict().items()}
        torch.manual_seed(seed)
        memory = LiquidLanguageModel(**common, innovation_memory_dim=16)
        torch.manual_seed(seed)
        control = LiquidLanguageModel(**common, ffn_mult=0.375)
        control_state = control.state_dict()
        control_state.update(shared)
        control.load_state_dict(control_state)
        delta = LiquidLanguageModel(**common, innovation_memory_dim=16)
        delta.innovation_memory = DeltaControl(64, 16)
        delta.load_state_dict(memory.state_dict())
        models = {'baseline': baseline, 'parameter_control': control, 'innovation': memory, 'delta_control': delta}
        generator = torch.Generator().manual_seed(20000 + seed)
        batches = [tuple(t.to(device) for t in train_batch(splits['train'], generator, args.batch_size, args.seq_len))
                   for _ in range(args.steps)]
        for name in args.arms:
            print(f'{name} seed={seed}', flush=True)
            model = models[name].to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            synchronize(device)
            start = time.perf_counter()
            for step, (x, y) in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite language loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if step % 100 == 0:
                    print(f'  update={step} train_nll={loss.item():.4f}', flush=True)
            synchronize(device)
            seconds = time.perf_counter() - start
            row = {'arm': name, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
                   'updates': args.steps, 'training_target_presentations': args.steps * args.batch_size * args.seq_len,
                   'seconds': seconds, 'last_train_nll': loss.item(),
                   'validation': evaluate(model, splits['validation'], args.seq_len, device),
                   'test': evaluate(model, splits['test'], args.seq_len, device)}
            results['arms'].append(row)
            args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
            model.to('cpu')


if __name__ == '__main__':
    main()
