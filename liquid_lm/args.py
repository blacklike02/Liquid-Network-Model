"""Command-line arguments."""

from __future__ import annotations

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Liquid RNN language model (v3.6: gate_mode ltc/cfc, ode_solver euler/heun)"
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
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow unsafe checkpoint loading (weights_only=False), "
                             "if safe loading fails. Only for your own files.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--state-carry", type=int, default=2)
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
