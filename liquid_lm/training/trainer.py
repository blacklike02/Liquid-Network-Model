"""Main training loop."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..checkpoint import load_checkpoint_raw, print_checkpoint_info, save_checkpoint
from ..data.batching import (
    IGNORE_INDEX,
    build_dialogue_anchors,
    build_train_mask,
    get_batch,
)
from ..data.corpus import load_texts
from ..data.tokenization import (
    _HAS_TOKENIZERS,
    build_tokenizer,
    default_stop_ids,
    encode_corpus,
    tokenizer_from_checkpoint,
)
from ..device import cuda_memory_report
from ..model.chunk_memory import (
    chunk_memory_read,
    chunk_memory_reset,
    chunk_memory_write,
)
from ..model.config import (
    ARCH_KEYS,
    _arch_value,
    create_model_from_config,
    load_model_state_compat,
)
from ..model.ema import EMA
from ..model.liquid_lm import LiquidLanguageModel
from ..model.utils import (
    clone_float_state,
    describe_wiring,
    detach_hidden,
    get_base_model,
    restore_float_state,
)
from ..train_logger import TrainLogger
from .amp import autocast_context, make_grad_scaler, resolve_amp_dtype
from .evaluate import evaluate
from .losses import masked_cross_entropy
from .optim import make_optimizer, make_scheduler


def run_training(args, device):
    data_dir = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    last_checkpoint_path = Path(args.last_checkpoint)

    start_epoch = 0
    global_step = 0
    best_val = math.inf
    bad_epochs = 0
    resume_payload = None

    val_split = float(args.val_split)
    if not 0.01 <= val_split <= 0.5:
        raise ValueError("--val-split must be between 0.01 and 0.5")

    if args.resume:
        resume_payload = load_checkpoint_raw(last_checkpoint_path, device)
        saved_cfg = resume_payload["config"]
        args.shuffle_files = bool(saved_cfg.get("shuffle_files", False))
        args.file_headers = bool(saved_cfg.get("file_headers", True))

    corpus = load_texts(
        data_dir, min_chars=args.min_chars, max_file_mb=args.max_file_mb,
        shuffle_files=args.shuffle_files, file_headers=args.file_headers,
    )
    corpus_sha = hashlib.sha256(corpus.encode("utf-8", errors="ignore")).hexdigest()

    if args.resume:
        tokenizer = tokenizer_from_checkpoint(resume_payload)
        saved_config = resume_payload["config"]
        if args.tokenizer != tokenizer.kind:
            print(f"[RESUME] tokenizer: {args.tokenizer} -> {tokenizer.kind}")
        for name in ARCH_KEYS:
            saved = _arch_value(saved_config, name)
            if saved is None:
                continue
            current = getattr(args, name, None)
            if current != saved:
                print(f"[RESUME] {name}: {current} -> {saved} (from checkpoint)")
                setattr(args, name, saved)
        saved_sha = resume_payload.get("corpus_sha256")
        if saved_sha and saved_sha != corpus_sha:
            print("[RESUME] WARNING: the corpus differs from the one used to train the checkpoint.")
    else:
        kind = args.tokenizer
        if kind == "bpe" and not _HAS_TOKENIZERS:
            print("[DATA] The 'tokenizers' library is not installed — falling back to char.")
            kind = "char"
        tok_corpus = corpus if kind == "char" else corpus[: int(len(corpus) * (1.0 - val_split))]
        tokenizer = build_tokenizer(tok_corpus, kind, args.vocab_size)

    vocab_size = tokenizer.vocab_size
    print(f"[DATA] Tokenizer: {tokenizer.kind} | vocab: {vocab_size}")

    encoded = encode_corpus(tokenizer, corpus)
    print(f"[DATA] Corpus tokens: {len(encoded):,}")

    split = int(len(encoded) * (1.0 - val_split))
    state_carry = max(1, int(args.state_carry))
    if args.memory_slots > 0 and state_carry < 2:
        print("[MODEL] WARNING: --memory-slots has no effect with --state-carry 1 "
              "(memory is read only from the second subchunk).")
    span_len = args.seq_len * state_carry
    if split <= span_len + 2:
        raise ValueError("Training portion is too small.")
    if len(encoded) - split <= span_len + 2:
        raise ValueError("Validation portion is too small.")

    data = torch.from_numpy(encoded.astype(np.int32))
    if device.type == "cuda":
        data = data.to(device)

    train_mask = None
    if args.loss_mask:
        train_mask_np = build_train_mask(encoded, tokenizer)
        if not train_mask_np.all():
            train_mask = torch.from_numpy(train_mask_np)
            if device.type == "cuda":
                train_mask = train_mask.to(device)
            masked_frac = 1.0 - train_mask_np.mean()
            print(f"[DATA] Loss mask: {masked_frac * 100:.1f}% of tokens excluded from loss.")
        else:
            print("[DATA] No dialogue tags found in corpus — loss mask is unnecessary.")

    dialogue_anchors = None
    val_dialogue_anchors = None
    if args.dialogue_anchor_ratio > 0.0:
        anchors_np = build_dialogue_anchors(encoded, tokenizer, span_len, 0, split)
        val_anchors_np = build_dialogue_anchors(encoded, tokenizer, span_len, split, len(encoded))
        if len(anchors_np) > 0:
            dialogue_anchors = torch.from_numpy(anchors_np)
            if device.type == "cuda":
                dialogue_anchors = dialogue_anchors.to(device)
            print(f"[DATA] Найдено {len(anchors_np):,} диалоговых примеров (train).")
        else:
            print("[DATA] No dialogue examples — sampling is random.")
        if len(val_anchors_np) > 0:
            val_dialogue_anchors = torch.from_numpy(val_anchors_np)
            if device.type == "cuda":
                val_dialogue_anchors = val_dialogue_anchors.to(device)

    model = LiquidLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        ode_unfolds=args.ode_unfolds,
        dropout=args.dropout,
        zoneout=args.zoneout,
        tie_weights=args.tie_weights,
        mlp_head=args.mlp_head,
        residual=args.residual,
        multi_tau=args.multi_tau,
        local_conv=args.local_conv,
        use_checkpoint=args.grad_checkpoint,
        gate_mode=args.gate_mode,
        ode_solver=args.ode_solver,
        wiring=args.wiring,
        ncp_command_frac=args.ncp_command_frac,
        ncp_motor_frac=args.ncp_motor_frac,
        memory_slots=args.memory_slots,
    ).to(device)

    if args.gate_mode == "cfc" and args.multi_tau:
        print("[MODEL] --multi-tau is ignored with --gate-mode cfc (there is no tau parameter).")

    if resume_payload is not None:
        load_model_state_compat(model, resume_payload["model_state"])

    base_model = get_base_model(model)
    if args.kernel != "torch":
        base_model.set_kernel(args.kernel)

    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Parameters: {parameter_count:,}")
    print(
        f"[MODEL] emb={args.embedding_dim} | hidden={args.hidden_size} | "
        f"layers={args.num_layers} | unfolds={args.ode_unfolds} | "
        f"carry={state_carry} | gate={args.gate_mode} | "
        f"solver={args.ode_solver if args.gate_mode == 'ltc' else '-'} | "
        f"multi_tau={args.multi_tau if args.gate_mode == 'ltc' else False} | "
        f"wiring={describe_wiring(model)} | "
        f"memory_slots={args.memory_slots} | "
        f"local_conv={args.local_conv} | ckpt={args.grad_checkpoint}"
    )

    optimizer = make_optimizer(model, args, device)
    updates_per_epoch = math.ceil(args.steps_per_epoch / args.grad_accumulation)
    total_updates = args.epochs * updates_per_epoch
    scheduler = make_scheduler(
        optimizer=optimizer, total_steps=total_updates,
        warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
    )

    amp_dtype = resolve_amp_dtype(device, args.amp)
    scaler = make_grad_scaler(amp_dtype == torch.float16, device)
    print(f"[AMP] {'Enabled: ' + str(amp_dtype) if amp_dtype is not None else 'Disabled'}")

    teacher = None
    if args.teacher_checkpoint:
        teacher_payload = load_checkpoint_raw(Path(args.teacher_checkpoint), device)
        teacher_tokenizer = tokenizer_from_checkpoint(teacher_payload)
        if teacher_tokenizer.vocab_size != vocab_size:
            raise ValueError(
                f"Vocab teacher ({teacher_tokenizer.vocab_size}) != student ({vocab_size})."
            )
        teacher = create_model_from_config(
            vocab_size=vocab_size, config=teacher_payload["config"], device=device,
        )
        load_model_state_compat(teacher, teacher_payload["model_state"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[KD] Teacher loaded: {args.teacher_checkpoint}")

    if args.compile:
        ok, mode_used = model.enable_compiled_step(args.compile_mode, batch_size=args.batch_size)
        if ok:
            print(f"[COMPILE] Recurrent step compiled (mode={mode_used}).")
        else:
            print("[COMPILE] torch.compile failed — continuing in eager mode.")

    ema = None
    if args.ema_decay > 0:
        if resume_payload is not None and resume_payload.get("ema_state"):
            ema = EMA.from_state(resume_payload["ema_state"], args.ema_decay)
            print(f"[EMA] Восстановлен (updates={ema.updates})")
        else:
            ema = EMA(model, decay=args.ema_decay)
        print(f"[EMA] Enabled (decay={args.ema_decay})")

    if resume_payload is not None:
        try:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            print(f"[RESUME] Optimizer/scheduler state не подошёл ({exc}); свежие.")
        if scaler is not None and resume_payload.get("scaler_state"):
            scaler.load_state_dict(resume_payload["scaler_state"])
        start_epoch = int(resume_payload["epoch"])
        global_step = int(resume_payload.get("global_step", 0))
        best_val = float(resume_payload.get("best_val", resume_payload["val_loss"]))
        print(f"[RESUME] Продолжаем с эпохи {start_epoch + 1}, best_val={best_val:.6f}")

    print(f"[SYSTEM] Trainable parameters: {trainable_count:,}")
    print(f"[SYSTEM] {cuda_memory_report()}")

    run_name = args.run_name or Path(args.checkpoint).stem
    train_logger = TrainLogger(
        backend=args.log_backend, run_name=run_name, logdir=args.log_dir,
        wandb_project=args.wandb_project, config=vars(args),
    )

    tokens_per_sec = 0.0
    stop_ids = default_stop_ids(tokenizer)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        token_sum = torch.zeros((), device=device)
        finite_ok = torch.ones((), dtype=torch.bool, device=device)
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, args.steps_per_epoch + 1):
            x_span, y_span = get_batch(
                data, args.batch_size, span_len, 0, split,
                anchors=dialogue_anchors, anchor_ratio=args.dialogue_anchor_ratio,
                train_mask=train_mask,
            )
            hidden = None
            memory = chunk_memory_reset(base_model, x_span.shape[0], device)
            full_steps = (args.steps_per_epoch // args.grad_accumulation) * args.grad_accumulation
            group_size = (
                args.grad_accumulation if step <= full_steps
                else args.steps_per_epoch - full_steps
            )

            for k in range(state_carry):
                xk = x_span[:, k * args.seq_len : (k + 1) * args.seq_len]
                yk = y_span[:, k * args.seq_len : (k + 1) * args.seq_len]
                hidden = chunk_memory_read(base_model, hidden, memory)

                with autocast_context(device, amp_dtype):
                    logits, hidden = model(xk, hidden=hidden)
                    raw_loss, sum_loss, valid_count = masked_cross_entropy(
                        logits.reshape(-1, vocab_size), yk.reshape(-1),
                    )

                    if teacher is not None:
                        with torch.no_grad():
                            t_logits, _ = teacher(xk)
                        T = args.kd_temperature
                        valid_mask = (yk.reshape(-1) != IGNORE_INDEX)
                        if valid_mask.any():
                            soft_s = F.log_softmax(logits.reshape(-1, vocab_size) / T, dim=-1)
                            soft_t = F.softmax(t_logits.reshape(-1, vocab_size) / T, dim=-1)
                            kd_per_token = F.kl_div(
                                soft_s, soft_t, reduction="none", log_target=False
                            ).sum(dim=-1)
                            kd_loss = (kd_per_token * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)
                            raw_loss = (1 - args.kd_alpha) * raw_loss + args.kd_alpha * kd_loss * (T * T)

                    loss = raw_loss / (state_carry * group_size)

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_sum += sum_loss.detach().float()
                token_sum += valid_count.float()
                finite_ok &= torch.isfinite(raw_loss.detach())
                if k < state_carry - 1:
                    memory = chunk_memory_write(base_model, hidden, memory)
                    if memory is not None:
                        memory = memory.detach()
                hidden = detach_hidden(hidden)

            should_update = (
                step % args.grad_accumulation == 0
                or step == args.steps_per_epoch
            )
            grad_norm_value = None
            if should_update:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    grad_norm_value = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    ).item()
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if ema is not None:
                    ema.update(model)

            if step % max(1, args.steps_per_epoch // 5) == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                ok_t = finite_ok.item()
                if scaler is None and not ok_t:
                    raise FloatingPointError(f"Loss NaN/Inf около шага {step}.")
                step_loss = (loss_sum / torch.clamp(token_sum, min=1.0)).item()
                print(
                    f"  step {step:4d}/{args.steps_per_epoch} | "
                    f"loss={step_loss:.4f} | "
                    f"lr={current_lr:.6g}"
                )
                step_metrics = {"train/loss_running": step_loss, "train/lr": current_lr}
                if grad_norm_value is not None:
                    step_metrics["train/grad_norm"] = grad_norm_value
                train_logger.log(step_metrics, step=global_step)

        train_loss = (loss_sum / torch.clamp(token_sum, min=1.0)).item()
        if scaler is None and not bool(finite_ok.item()):
            raise FloatingPointError("Loss NaN/Inf during the epoch.")

        elapsed = time.perf_counter() - epoch_start
        tokens_seen = args.batch_size * span_len * args.steps_per_epoch
        tokens_per_sec = tokens_seen / max(elapsed, 1e-9)

        val_loss = evaluate(
            model, data, args.batch_size, args.seq_len, state_carry,
            device, split, len(encoded), args.eval_batches, amp_dtype,
            train_mask=train_mask, anchors=val_dialogue_anchors,
            anchor_ratio=args.dialogue_anchor_ratio,
        )

        val_loss_ema = None
        using_ema_weights = False
        ema_backup = None
        if ema is not None:
            ema_backup = clone_float_state(model)
            ema.copy_to(model)
            val_loss_ema = evaluate(
                model, data, args.batch_size, args.seq_len, state_carry,
                device, split, len(encoded), args.eval_batches, amp_dtype,
                train_mask=train_mask, anchors=val_dialogue_anchors,
                anchor_ratio=args.dialogue_anchor_ratio,
            )
            if val_loss_ema < val_loss:
                using_ema_weights = True
            else:
                restore_float_state(model, ema_backup)

        candidate = val_loss if val_loss_ema is None else min(val_loss, val_loss_ema)
        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(candidate, 20.0))
        current_lr = optimizer.param_groups[0]["lr"]
        ema_note = f" | EMA val={val_loss_ema:.4f}" if val_loss_ema is not None else ""
        print(
            f"[EPOCH {epoch:03d}/{args.epochs}] "
            f"train loss={train_loss:.4f} ppl={train_ppl:.2f} | "
            f"val loss={candidate:.4f} ppl={val_ppl:.2f}{ema_note} | "
            f"lr={current_lr:.6g} | {elapsed:.1f}s | {tokens_per_sec:,.0f} tok/s"
        )
        epoch_metrics = {
            "train/loss": train_loss,
            "train/perplexity": train_ppl,
            "val/loss": candidate,
            "val/perplexity": val_ppl,
            "train/lr": current_lr,
            "perf/tokens_per_sec": tokens_per_sec,
        }
        if val_loss_ema is not None:
            epoch_metrics["val/loss_ema"] = val_loss_ema
        train_logger.log(epoch_metrics, step=global_step)

        improved = candidate < best_val
        if improved:
            best_val = candidate
            bad_epochs = 0
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler,
                epoch, global_step, train_loss, candidate, best_val,
                tokenizer, args,
                ema_state=ema.state_dict() if ema is not None else None,
                corpus_sha256=corpus_sha,
            )
            print(f"[SAVE] New best -> {checkpoint_path}" + (" (EMA)" if using_ema_weights else ""))
        else:
            bad_epochs += 1

        if epoch % 5 == 0 or epoch == args.epochs or improved:
            preview = model.generate(
                prompt=args.prompt, tokenizer=tokenizer,
                max_new_tokens=min(300, args.generate),
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                device=device, stop_token_ids=stop_ids,
            )
            print("\n----- PREVIEW -----")
            print(preview)
            print("-------------------\n")

        if ema is not None and using_ema_weights:
            restore_float_state(model, ema_backup)

        if epoch % max(1, args.save_every) == 0 or epoch == args.epochs:
            save_checkpoint(
                last_checkpoint_path, model, optimizer, scheduler, scaler,
                epoch, global_step, train_loss, val_loss, best_val,
                tokenizer, args,
                ema_state=ema.state_dict() if ema is not None else None,
                corpus_sha256=corpus_sha,
            )
            print(f"[SAVE] Last -> {last_checkpoint_path}")

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[EARLY STOP] {args.patience} эпох без улучшения.")
            break

    train_logger.close()

    final_path = checkpoint_path if checkpoint_path.exists() else last_checkpoint_path
    best_payload = load_checkpoint_raw(final_path, device)
    best_tokenizer = tokenizer_from_checkpoint(best_payload)
    best_model = create_model_from_config(
        vocab_size=best_tokenizer.vocab_size,
        config=best_payload["config"], device=device,
    )
    load_model_state_compat(best_model, best_payload["model_state"])
    if args.kernel != "torch":
        best_model.set_kernel(args.kernel)
    print("\n========== BEST CHECKPOINT ==========")
    print_checkpoint_info(best_payload)
    generated = best_model.generate(
        prompt=args.prompt, tokenizer=best_tokenizer, max_new_tokens=args.generate,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        device=device, stop_token_ids=default_stop_ids(best_tokenizer),
    )
    print("\n========== FINAL GENERATED TEXT ==========\n")
    print(generated)
    print("\n==========================================")
    print(f"[SYSTEM] {cuda_memory_report()}")

    info = {
        "device": str(device),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "parameters": parameter_count,
        "tokenizer": best_tokenizer.kind,
        "vocab_size": best_tokenizer.vocab_size,
        "corpus_chars": len(corpus),
        "corpus_tokens": len(encoded),
        "state_carry": state_carry,
        "best_epoch": best_payload["epoch"],
        "train_loss": best_payload["train_loss"],
        "val_loss": best_payload["val_loss"],
        "config": best_payload["config"],
        "amp": str(amp_dtype),
        "ema_decay": args.ema_decay,
        "multi_tau": args.multi_tau,
        "local_conv": args.local_conv,
        "grad_checkpoint": args.grad_checkpoint,
        "tokens_per_sec_last_epoch": round(tokens_per_sec, 1),
    }
    Path("training_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    Path("generated.txt").write_text(generated, encoding="utf-8")
    print("[SAVE] training_info.json")
    print("[SAVE] generated.txt")
