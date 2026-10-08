"""Entry point: argument parsing and dispatch to the selected mode."""

from __future__ import annotations

from pathlib import Path

from .args import build_parser
from .checkpoint import load_checkpoint_raw, print_checkpoint_info, set_trust_checkpoints
from .data.tokenization import _HAS_TOKENIZERS
from .device import choose_device, print_device_info, set_seed
from .inference import run_generate_only
from .selftest import selftest_triton_tail
from .training.trainer import run_training


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_trust_checkpoints(args.trust_checkpoint)

    set_seed(args.seed)
    device = choose_device(args.device)
    print_device_info(device)

    if args.selftest_triton:
        selftest_triton_tail(device)
        return

    if args.info:
        checkpoint = load_checkpoint_raw(Path(args.checkpoint), device)
        print_checkpoint_info(checkpoint)
        return

    if args.generate_only or args.chat or args.chat_repl:
        run_generate_only(args, device)
        return

    if args.grad_accumulation < 1:
        raise ValueError("--grad-accumulation must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.seq_len < 8:
        raise ValueError("--seq-len must be >= 8")
    if args.state_carry < 1:
        raise ValueError("--state-carry must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.steps_per_epoch < 1:
        raise ValueError("--steps-per-epoch must be >= 1")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 <= args.zoneout < 1.0:
        raise ValueError("--zoneout must be in [0, 1)")
    if not args.resume and args.tie_weights and args.embedding_dim != args.hidden_size:
        raise ValueError("--tie-weights requires --embedding-dim == --hidden-size")
    if args.ema_decay != 0 and not 0.8 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be 0 or in [0.8, 1)")
    if args.tokenizer == "bpe" and _HAS_TOKENIZERS and args.vocab_size < 300:
        raise ValueError("--vocab-size is too small for byte-level BPE (minimum 300)")

    run_training(args, device)
