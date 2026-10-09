"""Text generation and interactive chat."""

from __future__ import annotations

import time
from pathlib import Path

from .data import BOT_TAG, EXAMPLE_END_TAG, USER_TAG
from .model import (
    create_model_from_config,
    load_checkpoint_raw,
    load_model_state_compat,
    print_checkpoint_info,
)
from .tokenizer import default_stop_ids, tokenizer_from_checkpoint
from .utils import cuda_memory_report


def run_chat_repl(args, device):
    checkpoint = load_checkpoint_raw(Path(args.checkpoint), device)
    print_checkpoint_info(checkpoint)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    model = create_model_from_config(
        vocab_size=tokenizer.vocab_size, config=checkpoint["config"], device=device,
    )
    load_model_state_compat(model, checkpoint["model_state"])
    model.eval()
    if args.kernel != "torch":
        model.set_kernel(args.kernel)
    if args.compile:
        ok, mode_used = model.enable_compiled_step(args.compile_mode, batch_size=1, inference=True)
        print(
            f"[COMPILE] Шаг скомпилирован (mode={mode_used})."
            if ok else "[COMPILE] Failed — using eager mode."
        )
    user_tid = tokenizer.token_to_id(USER_TAG)
    stop_ids = default_stop_ids(tokenizer)
    if user_tid is not None:
        stop_ids.append(user_tid)
    print(f"[SYSTEM] {cuda_memory_report()}")
    print("\n[CHAT] Interactive mode. Empty line/'exit' — quit.\n")
    history = ""
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_msg or user_msg.lower() in ("exit", "quit", "quit", "q"):
            break
        history += f"{USER_TAG}\n{user_msg}\n{BOT_TAG}\n"
        if len(history) > args.chat_history_chars:
            cut = history.find(USER_TAG, len(history) - args.chat_history_chars)
            history = history[cut:] if cut != -1 else history[-args.chat_history_chars:]
        text = model.generate(
            prompt=history, tokenizer=tokenizer, max_new_tokens=args.generate,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            device=device, stop_token_ids=stop_ids, only_new=True,
        )
        answer = text.strip()
        print(f"Бот: {answer}\n")
        history += f"{answer}\n{EXAMPLE_END_TAG}\n"


def run_generate_only(args, device):
    if args.chat_repl:
        run_chat_repl(args, device)
        return
    checkpoint = load_checkpoint_raw(Path(args.checkpoint), device)
    print_checkpoint_info(checkpoint)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    model = create_model_from_config(
        vocab_size=tokenizer.vocab_size, config=checkpoint["config"], device=device,
    )
    load_model_state_compat(model, checkpoint["model_state"])
    model.eval()
    if args.kernel != "torch":
        model.set_kernel(args.kernel)
    if args.compile:
        ok, mode_used = model.enable_compiled_step(args.compile_mode, batch_size=1, inference=True)
        if ok:
            print(f"[COMPILE] Шаг скомпилирован (mode={mode_used}).")
        else:
            print("[COMPILE] Failed — using eager mode.")
    stop_ids = default_stop_ids(tokenizer)
    if args.chat:
        user_tid = tokenizer.token_to_id(USER_TAG)
        if user_tid is not None:
            stop_ids.append(user_tid)
        prompt = f"{USER_TAG}\n{args.prompt.strip()}\n{BOT_TAG}\n"
    else:
        prompt = args.prompt
    print(f"[SYSTEM] {cuda_memory_report()}")
    start = time.perf_counter()
    text = model.generate(
        prompt=prompt, tokenizer=tokenizer, max_new_tokens=args.generate,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        device=device, stop_token_ids=stop_ids, only_new=args.chat,
    )
    elapsed = time.perf_counter() - start
    print("\n========== GENERATED ==========\n")
    if args.chat:
        print(text.strip())
    else:
        print(text)
    print("\n================================")
    print(f"[GEN] {args.generate} new tokens (max) in {elapsed:.2f}s")
    print(f"[SYSTEM] {cuda_memory_report()}")
