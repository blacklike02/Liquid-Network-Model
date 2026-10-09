"""Command line: argument parser, Triton self-test and entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .inference import run_generate_only
from .layers import LiquidCell, _TRITON_AVAILABLE
from .model import load_checkpoint_raw, print_checkpoint_info, set_trust_checkpoints
from .tokenizer import _HAS_TOKENIZERS
from .training import run_training
from .utils import choose_device, print_device_info, set_seed


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Liquid RNN language model (v3.7: gate_mode ltc/cfc, ode_solver euler/heun, "
            "SwiGLU FFN + RMSNorm, top-k KD, long-context stage, FIM)"
        )
    )

    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chat", action="store_true",
                        help="A single response in dialogue format (implies --generate-only).")
    parser.add_argument("--chat-repl", action="store_true",
                        help="Interactive chat (implies --generate-only).")
    parser.add_argument("--chat-history-chars", type=int, default=4000)

    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--min-chars", type=int, default=1000)
    parser.add_argument("--max-file-mb", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "char"], default="bpe")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--loss-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dialogue-anchor-ratio", type=float, default=0.7)
    parser.add_argument("--shuffle-files", action=argparse.BooleanOptionalAction, default=True,
                        help="Deterministically shuffle corpus file order (by path hash), "
                             "so the validation tail is not the \"last files alphabetically.\" "
                             "With --resume, use the checkpoint value (legacy: off).")
    parser.add_argument("--file-headers", action=argparse.BooleanOptionalAction, default=False,
                        help="Insert '===== filename =====' lines between files "
                             "(legacy behavior). Off by default. With --resume, use the "
                             "checkpoint value (legacy: on).")
    parser.add_argument("--fim-rate", type=float, default=0.0,
                        help="Fill-in-the-middle for CODE files (.py .js .ts .java .cpp ...). "
                             "0 (default) = off. With p > 0, code files are also read from "
                             "--data, split into ~3000-char chunks on line boundaries, and each "
                             "chunk is rewritten with probability p as "
                             "<|fim_prefix|>P<|fim_suffix|>S<|fim_middle|>M<|end|>. "
                             "Typical value: 0.5. Requires --tokenizer bpe (new tokenizers get "
                             "the <|fim_*|> special tokens; old checkpoints do not have them, "
                             "FIM is then disabled). The rearrangement is deterministic. "
                             "With --resume the value is taken from the checkpoint.")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow unsafe checkpoint loading (weights_only=False), "
                             "if safe loading fails. Only for your own files.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--state-carry", type=int, default=2)
    parser.add_argument("--long-epochs", type=int, default=0,
                        help="Extra epochs of a final long-context stage after --epochs "
                             "(0 = off). The same weights are trained with --long-state-carry "
                             "subchunks per span (longer carried context) and an accelerated LR "
                             "decay (see --stage-lr-ratio). Validation inside this stage uses the "
                             "longer carry, so best_val is reset when the stage starts; early "
                             "stopping in the main stage jumps to this stage instead of ending.")
    parser.add_argument("--long-state-carry", type=int, default=8,
                        help="--state-carry used during the long-context stage.")
    parser.add_argument("--stage-lr-ratio", type=float, default=0.3,
                        help="With --long-epochs: the LR (as a fraction of the peak) at which the "
                             "long stage starts; the main stage cosine-decays to it, then the LR "
                             "falls linearly to --min-lr-ratio over the long stage.")
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ode-unfolds", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--zoneout", type=float, default=0.1)
    parser.add_argument("--tie-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlp-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-tau", action=argparse.BooleanOptionalAction, default=True,
                        help="Multi-timescale tau (fast/medium/slow), mixing — "
                             "three SHARED (not per-channel) softmax weights; an enhancement, "
                             "not a change to the CfLTC concept.")
    parser.add_argument("--local-conv", action=argparse.BooleanOptionalAction, default=True,
                        help="Causal depthwise 1D conv (k=5) before the liquid core "
                             "(left-only padding — does not peek into the future).")
    parser.add_argument("--gate-mode", type=str, choices=["ltc", "cfc"], default="ltc",
                        help="ltc (by default) — exponential decay exp(-dt/tau) to "
                             "one candidate, optionally with multi-tau mixing. cfc — "
                             "closed-form formula from Hasani et al. 2022 (Nature MI): two "
                             "candidates ff1/ff2 and trainable sigmoid interpolation over "
                             "time instead of explicit ODE integration. With cfc, the "
                             "--multi-tau is ignored (this gate has no tau).")
    parser.add_argument("--ode-solver", type=str, choices=["euler", "heun"], default="euler",
                        help="Substep integration method (--ode-unfolds) in the ltc gate "
                             "(does not affect --gate-mode cfc). euler — candidate/tau "
                             "are evaluated once at the start of the substep, then the ODE is solved "
                             "exactly with 'frozen' candidate/tau. heun — "
                             "predictor-corrector (RK2): the substep is computed twice, "
                             "candidate/tau are averaged — second-order accuracy at the cost of "
                             "2x computations per substep.")
    parser.add_argument("--wiring", type=str, choices=["dense", "ncp"], default="dense",
                        help="dense (by default) — standard fully connected recurrent "
                             "matrix. ncp — the recurrent matrix is replaced by a SET of "
                             "physically SMALLER matrices using the Neural Circuit "
                             "Policies (Lechner et al. 2020): neurons are divided into "
                             "inter/command/motor, allowed paths are inter->inter, "
                             "inter->command, command->command, command->motor, and "
                             "feedback motor->command — a real reduction in both "
                             "parameters and recurrent-part FLOPs. Does NOT include "
                             "the biophysical synapse polarity from the original paper "
                             "(see the NCPRecurrent docstring).")
    parser.add_argument("--ncp-command-frac", type=float, default=0.4,
                        help="Fraction of hidden_size allocated to command neurons in --wiring ncp.")
    parser.add_argument("--ncp-motor-frac", type=float, default=0.2,
                        help="Fraction of hidden_size allocated to motor neurons in --wiring ncp "
                             "(the remainder goes to inter neurons).")
    parser.add_argument("--memory-slots", type=int, default=0,
                        help="0 (by default) — disabled. >0 — enables ChunkMemory: "
                             "a sliding FIFO buffer of summaries from previous --state-carry subchunks "
                             "(in the style of Transformer-XL memory / the M-CfC idea), read/written "
                             "around each subchunk in run_training/evaluate. Requires "
                             "--state-carry >= 2. NOT used in generate()/chat.")
    parser.add_argument("--ffn-mult", type=float, default=0.0,
                        help="0 (default) = off. >0 adds a pre-norm SwiGLU channel-mixing block "
                             "(with its own residual) after every liquid layer, with inner width "
                             "~ffn_mult*hidden_size (typical: 2.0). The block runs inside the "
                             "per-token recurrent step, so it adds parameters and per-step cost "
                             "(measure the speed on your GPU before long runs). "
                             "Stored in the checkpoint; old checkpoints are loaded with it off.")
    parser.add_argument("--norm-type", type=str, choices=["layer", "rms"], default="layer",
                        help="Normalization in the cells, FFN blocks and before the readout. "
                             "layer (default) = LayerNorm, rms = RMSNorm (no mean/bias). "
                             "Stored in the checkpoint.")
    parser.add_argument("--kernel", type=str, choices=["torch", "triton"], default="torch",
                        help="Backend for the ltc substep tail. torch (by default) — standard "
                             "PyTorch operations. triton — a SINGLE fused kernel instead of ~8 "
                             "elementwise operations. INFERENCE ONLY (generate/chat/evaluate; "
                             "the kernel has no backward), CUDA + triton package only, only "
                             "--gate-mode ltc --ode-solver euler; in other cases "
                             "falls back to torch with a message. Before use, "
                             "run --selftest-triton on your GPU.")
    parser.add_argument("--selftest-triton", action="store_true",
                        help="Compare the triton kernel with the torch path, benchmark it, then exit.")
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=False,
                        help="Gradient checkpointing on every LiquidCell call. "
                             "WARNING: granularity 'one cell call per "
                             "token' is very fine-grained — checkpoint overhead "
                             "may outweigh the memory savings. Actual TBPTT memory "
                             "uses O(T) activations across the whole window. Measure on "
                             "your --batch-size/--seq-len, before relying on this "
                             "as an OOM solution.")

    parser.add_argument("--teacher-checkpoint", type=str, default=None,
                        help="Path to the teacher checkpoint for distillation (KD). "
                             "WARNING: teacher() is called without carrying hidden "
                             "between state-carry chunks (always from zero) — its "
                             "soft-labels for k>0 chunks are less informative.")
    parser.add_argument("--kd-alpha", type=float, default=0.3,
                        help="KD-loss weight: loss = (1-α)*CE + α*KL.")
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-topk", type=int, default=32,
                        help="Distillation on the teacher's top-k tokens with the KL split by the "
                             "chain rule into a binary (mass on the top-k set) term and a "
                             "conditional KL inside the set. 0 = the old full-vocabulary KL. "
                             "The teacher now also gets the carried hidden state/memory across "
                             "--state-carry subchunks (same context as the student).")

    parser.add_argument("--optimizer", type=str, choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", type=str, choices=["auto", "fp16", "bf16", "off"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode", type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="reduce-overhead = CUDA Graphs (maximum speed); "
             "max-autotune is NOT recommended for this model — tries "
             "dozens of gemm-kernel variants for a gain that these "
             "small matrices do not provide, and may take minutes on "
             "the first call; on failure, automatically falls back to default.",
    )

    parser.add_argument("--checkpoint", type=str, default="liquid_text_best.pt")
    parser.add_argument("--last-checkpoint", type=str, default="liquid_text_last.pt")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--log-backend", type=str, choices=["none", "tensorboard", "wandb"], default="none",
                        help="Log training metrics in addition to stdout. none (by default) — "
                             "writes nothing. tensorboard — via torch.utils.tensorboard "
                             "(pip install tensorboard, then `tensorboard --logdir <--log-dir>`). "
                             "wandb — via Weights & Biases (pip install wandb, requires "
                             "`wandb login` in advance). If the package is not installed, training does not "
                             "fail — a warning is printed and logging is disabled.")
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="Directory for TensorBoard logs (--log-backend tensorboard).")
    parser.add_argument("--wandb-project", type=str, default="liquid-lm",
                        help="W&B project name (--log-backend wandb).")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for logging; by default, taken from the "
                             "--checkpoint.")

    parser.add_argument("--prompt", type=str, default="Neural network")
    parser.add_argument("--generate", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--repetition-window", type=int, default=128)

    parser.add_argument("--seed", type=int, default=1)
    return parser


# ---------------------------------------------------------------------------
# Triton self-test
# ---------------------------------------------------------------------------


def selftest_triton_tail(device) -> None:
    """Compares the triton tail of the ltc substep with the regular torch path on
    random data (single-tau and multi-tau, including uneven bank weights and
    "hard" inputs for softplus) and measures timing. Run manually:
        python liquid_text_model.py --selftest-triton
    Requires CUDA and the triton package; otherwise honestly reports that
    there is nothing to check."""
    if not _TRITON_AVAILABLE:
        print("[SELFTEST] the triton package is not installed — nothing to check.")
        return
    if device.type != "cuda":
        print("[SELFTEST] A CUDA GPU is required — nothing to check.")
        return
    torch.manual_seed(0)
    B, H, IN = 32, 384, 384
    all_ok = True
    for multi_tau in (False, True):
        cell = LiquidCell(IN, H, ode_unfolds=1, multi_tau=multi_tau).to(device).eval()
        if multi_tau:
            cell.tau_mix.data = torch.randn(3, device=device)
        x = torch.randn(B, IN, device=device) * 3.0
        h = torch.randn(B, H, device=device)
        with torch.no_grad():
            xp, tx = cell.precompute_x(x)
            fw, fb = cell.fused_h_weights()
            cell.kernel = "torch"
            ref, _, _ = cell._ltc_step(h, xp, tx, 1.0, fw, fb)
            cell.kernel = "triton"
            if not cell._use_triton_tail(h):
                print("[SELFTEST] The triton path was not activated — check the conditions.")
                return
            got, _, _ = cell._ltc_step(h, xp, tx, 1.0, fw, fb)
            diff = (ref - got).abs().max().item()
            ok = diff < 1e-4
            all_ok &= ok

            def bench(kernel_name: str, iters: int = 300) -> float:
                cell.kernel = kernel_name
                for _ in range(20):
                    cell._ltc_step(h, xp, tx, 1.0, fw, fb)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    cell._ltc_step(h, xp, tx, 1.0, fw, fb)
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / iters * 1e3

            t_torch = bench("torch")
            t_triton = bench("triton")
        print(
            f"[SELFTEST] multi_tau={multi_tau}: max|diff|={diff:.2e} "
            f"{'OK' if ok else 'FAIL'} | torch={t_torch:.3f} ms/step, "
            f"triton={t_triton:.3f} ms/step"
        )
    print("[SELFTEST] RESULT:", "all checks passed" if all_ok else "DIFFERENCES FOUND — do not use --kernel triton")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    if args.long_epochs < 0:
        raise ValueError("--long-epochs must be >= 0")
    if args.long_epochs > 0 and args.long_state_carry < 1:
        raise ValueError("--long-state-carry must be >= 1")
    if not 0.0 < args.stage_lr_ratio <= 1.0:
        raise ValueError("--stage-lr-ratio must be in (0, 1]")
    if args.ffn_mult < 0:
        raise ValueError("--ffn-mult must be >= 0")
    if args.kd_topk < 0:
        raise ValueError("--kd-topk must be >= 0")
    if not 0.0 <= args.fim_rate <= 1.0:
        raise ValueError("--fim-rate must be between 0 and 1")
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
