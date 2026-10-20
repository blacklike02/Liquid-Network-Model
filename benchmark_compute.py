"""Paired benchmark of output batching and shared multi-tau weights.

The reference reproduces the previous per-token normalization/softmax schedule.
It shares the current cell equations, weights, inputs and stochastic masks.
Reports synchronized full-step wall time, not an isolated kernel speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from liquid_lm.model import LiquidLanguageModel
from liquid_lm.tokenizer import CharTokenizer


class RepeatedOutputReference(LiquidLanguageModel):
    """Pre-optimization output schedule; kept for reproducible A/B checks."""

    def _step_impl(self, x_t, xp_t, tx_t, hidden, dt, fused, mix_weights=None, input_projections=None):
        out, hidden = super()._step_impl(x_t, xp_t, tx_t, hidden, dt, fused)
        return self.final_norm(out), hidden

    def forward(self, tokens, hidden=None, dt=1., fused=None, last_token_only=False):
        batch_size, seq_len = tokens.shape
        conv_buf = None
        if hidden is None:
            hidden = self.initial_hidden(batch_size, tokens.device)
        else:
            hidden = list(hidden)
            state_count = self.num_layers + int(self.innovation_memory is not None)
            if len(hidden) > state_count:
                conv_buf = hidden[state_count]
                hidden = hidden[:state_count]
        new_conv_buf = None
        x = self.emb_dropout(self.embedding(tokens))
        if self.local_conv_layer is not None:
            if conv_buf is None:
                conv_buf = x.new_zeros(batch_size, self._conv_left_pad, x.size(-1))
            x_ext = torch.cat([conv_buf.to(x.dtype), x], dim=1)
            new_conv_buf = x_ext[:, -self._conv_left_pad:, :]
            x = x + self.local_conv_layer(x_ext.transpose(1, 2)).transpose(1, 2)
        if fused is None:
            fused = [cell.fused_h_weights() for cell in self.cells]
        xp_seq, tx_seq = self.cells[0].precompute_x(x)
        outputs = []
        step_fn = self._step_impl if self._kernel == 'triton' else (self._step_fn or self._step_impl)
        for t in range(seq_len):
            out, hidden = step_fn(x[:, t], xp_seq[:, t], tx_seq[:, t], hidden, dt, fused)
            outputs.append(out)
        features = outputs[-1].unsqueeze(1) if last_token_only else torch.stack(outputs, dim=1)
        logits = self.readout(features)
        if new_conv_buf is not None:
            hidden = list(hidden) + [new_conv_buf]
        return logits, hidden


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def autocast(device, precision):
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[precision]
    return torch.autocast(device.type, dtype=dtype, enabled=precision != 'fp32')


def check_equivalence(models, tokens, targets, device, precision, seed):
    outputs, states, gradients = [], [], []
    for model in models:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        with autocast(device, precision):
            logits, hidden = model(tokens)
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        loss.backward()
        outputs.append(logits.detach().float())
        states.append([h.detach().float() for h in hidden])
        gradients.append({n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None})
    assert gradients[0].keys() == gradients[1].keys()
    atol, rtol = {'fp32': (8e-5, 8e-4), 'bf16': (0.02, 0.06), 'fp16': (0.002, 0.015)}[precision]
    torch.testing.assert_close(outputs[0], outputs[1], atol=atol, rtol=rtol)
    for a, b in zip(states[0], states[1]):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    max_grad = 0.
    for name in gradients[0]:
        a, b = gradients[0][name], gradients[1][name]
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)
        max_grad = max(max_grad, (a-b).abs().max().item())
    return {'max_logits_abs_diff': (outputs[0]-outputs[1]).abs().max().item(),
            'max_grad_abs_diff': max_grad, 'checked_parameter_tensors': len(gradients[0]),
            'atol': atol, 'rtol': rtol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--kernel', choices=['torch','triton'], default='triton')
    parser.add_argument('--precisions', nargs='+', choices=['fp32','bf16','fp16'], default=['fp32'])
    parser.add_argument('--cases', nargs='+', choices=['train','regularized','generate'], default=['train','regularized','generate'])
    parser.add_argument('--innovation-memory-dim', type=int, default=0)
    parser.add_argument('--identical-control', action='store_true',
                        help='Compare two copies of the old schedule to estimate timing bias')
    parser.add_argument('--pairs', type=int, default=11)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=384)
    parser.add_argument('--vocab-size', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=Path('runs/compute_acceleration/paired.json'))
    args = parser.parse_args()
    if min(args.pairs,args.hidden_size,args.vocab_size) < 1 or args.warmup < 0 or args.innovation_memory_dim < 0:
        parser.error('invalid sizes or negative warmup/memory width')
    if args.device == 'cpu' and args.kernel == 'triton':
        parser.error('CPU requires --kernel torch')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    result = {'config':{**vars(args),'output':str(args.output)}, 'torch':str(torch.__version__),
              'device':torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
              'scope':'adjacent alternating pairs; same initialization/inputs/RNG; full forward/CE/backward/AdamW',
              'cases':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for case in args.cases:
        for precision in args.precisions:
            config=dict(vocab_size=args.vocab_size,embedding_dim=args.hidden_size,hidden_size=args.hidden_size,
                        num_layers=2,dropout=0.05 if case=='regularized' else 0.,
                        zoneout=0.1 if case=='regularized' else 0.,norm_type="rms",
                        innovation_memory_dim=args.innovation_memory_dim)
            torch.manual_seed(2026)
            before=RepeatedOutputReference(**config).to(device)
            after_class=RepeatedOutputReference if args.identical_control else LiquidLanguageModel
            after=after_class(**config).to(device)
            after.load_state_dict(before.state_dict())
            models=[before,after]
            for model in models:
                if args.kernel=='triton' and model.set_kernel('triton')!='triton':
                    raise RuntimeError('Triton unavailable')
            row={'case':case,'precision':precision,'parameters':sum(p.numel() for p in before.parameters()),'pairs':[]}
            if case!='generate':
                generator=torch.Generator().manual_seed(789)
                tokens=torch.randint(args.vocab_size,(16,64),generator=generator).to(device)
                targets=torch.randint(args.vocab_size,(16,64),generator=generator).to(device)
                row['equivalence']=check_equivalence(models,tokens,targets,device,precision,820)
                optimizers=[torch.optim.AdamW(m.parameters(),lr=0.001,foreach=False) for m in models]
                scalers=[torch.amp.GradScaler(device.type,enabled=precision=='fp16',init_scale=1024.) for _ in models]

                def step(index):
                    model,optimizer,scaler=models[index],optimizers[index],scalers[index]
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device,precision):
                        logits,_=model(tokens)
                        loss=F.cross_entropy(logits.float().flatten(0,1),targets.flatten())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    return loss.detach()
            else:
                chars=[chr(0x100+i) for i in range(args.vocab_size)]
                tokenizer=CharTokenizer({c:i for i,c in enumerate(chars)},chars)
                prompt=''.join(chars[i%min(args.vocab_size,20)] for i in range(128))
                for model in models:
                    model.eval()

                def step(index):
                    with torch.inference_mode(),autocast(device,precision):
                        return models[index].generate(prompt,tokenizer,max_new_tokens=32,top_k=1,top_p=1.,
                                                      repetition_penalty=1.,repetition_window=0,device=device)
                a,b=step(0),step(1)
                assert a==b and len(a)==160
                row['equivalence']={'greedy_output_matches':True,'output_tokens':len(a),
                                    'sha256':hashlib.sha256(a.encode('utf-8')).hexdigest()}
            for i in range(args.warmup):
                for index in (0,1):
                    torch.manual_seed(1000+i)
                    step(index)
            sync(device)
            for pair in range(args.pairs):
                measured={}
                for index in ((0,1) if pair%2==0 else (1,0)):
                    torch.manual_seed(2000+pair)
                    sync(device)
                    start=time.perf_counter()
                    value=step(index)
                    sync(device)
                    measured['before_ms' if index==0 else 'after_ms']=(time.perf_counter()-start)*1000
                    if case!='generate' and not torch.isfinite(value):
                        raise RuntimeError('Nonfinite measured loss')
                measured['speedup']=measured['before_ms']/measured['after_ms']
                row['pairs'].append(measured)
            if case!='generate':
                expected=args.warmup+args.pairs
                for model,optimizer in zip(models,optimizers):
                    steps=[int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
                    assert steps and min(steps)==max(steps)==expected,steps
                    assert all(torch.isfinite(p).all() and p.grad is not None and torch.isfinite(p.grad).all()
                               for p in model.parameters())
                row['optimizer_updates']=expected
            row.update(before_median_ms=statistics.median(p['before_ms'] for p in row['pairs']),
                       after_median_ms=statistics.median(p['after_ms'] for p in row['pairs']),
                       median_paired_speedup=statistics.median(p['speedup'] for p in row['pairs']))
            result['cases'].append(row)
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
            print(case,precision,'before',round(row['before_median_ms'],3),'after',round(row['after_median_ms'],3),
                  'paired speedup',round(row['median_paired_speedup'],3),flush=True)


if __name__=='__main__':
    main()
