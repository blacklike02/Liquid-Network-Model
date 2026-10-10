"""Training: batching, loss, AMP, EMA, evaluation and the main training loop."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    BOT_TAG,
    EXAMPLE_END_TAG,
    FIM_MIDDLE_TAG,
    FIM_PREFIX_TAG,
    FIM_SUFFIX_TAG,
    SYSTEM_TAG,
    USER_TAG,
    load_texts,
)
from .layers import chunk_memory_read, chunk_memory_reset, chunk_memory_write
from .model import (
    ARCH_KEYS,
    LiquidLanguageModel,
    _arch_value,
    clone_float_state,
    create_model_from_config,
    describe_wiring,
    detach_hidden,
    get_base_model,
    load_checkpoint_raw,
    load_model_state_compat,
    print_checkpoint_info,
    restore_float_state,
    save_checkpoint,
)
from .optim import make_optimizer, make_scheduler
from .tokenizer import (
    _HAS_TOKENIZERS,
    build_tokenizer,
    default_stop_ids,
    encode_corpus,
    tokenizer_from_checkpoint,
)
from .utils import TrainLogger, cuda_memory_report


# ---------------------------------------------------------------------------
# Batching, dialogue anchors and loss mask
# ---------------------------------------------------------------------------


_ARANGE_CACHE: dict[tuple[str, int], torch.Tensor] = {}


IGNORE_INDEX = -100


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    split_start: int,
    split_end: int,
    gen: torch.Generator | None = None,
    anchors: torch.Tensor | None = None,
    anchor_ratio: float = 0.0,
    train_mask: torch.Tensor | None = None,
):
    max_start = split_end - seq_len - 1
    if max_start < split_start:
        raise ValueError("Not enough text for the selected seq_len and train/val split.")
    key = (str(data.device), seq_len)
    ar = _ARANGE_CACHE.get(key)
    if ar is None or ar.device != data.device:
        ar = torch.arange(seq_len + 1, device=data.device)
        _ARANGE_CACHE[key] = ar
    randint_kwargs = {"generator": gen} if gen is not None else {}
    if anchors is not None and len(anchors) > 0 and anchor_ratio > 0.0:
        n_anchor = int(round(batch_size * min(1.0, anchor_ratio)))
        n_random = batch_size - n_anchor
        anchor_pick = torch.randint(0, len(anchors), (n_anchor,), device=data.device, **randint_kwargs)
        starts_anchor = anchors[anchor_pick]
        if n_random > 0:
            starts_random = torch.randint(
                split_start, max_start + 1, (n_random,),
                device=data.device, **randint_kwargs,
            )
            starts = torch.cat([starts_anchor, starts_random])
        else:
            starts = starts_anchor
    else:
        starts = torch.randint(
            split_start, max_start + 1, (batch_size,),
            device=data.device, **randint_kwargs,
        )
    idx = starts[:, None] + ar
    windows = data[idx]
    x = windows[:, :-1]
    if x.dtype != torch.long:
        x = x.long()
    x = x.contiguous()
    y = windows[:, 1:]
    if y.dtype != torch.long:
        y = y.long()
    if train_mask is not None:
        mask_windows = train_mask[idx][:, 1:]
        y = y.masked_fill(~mask_windows, IGNORE_INDEX)
    return x, y.contiguous()


def build_dialogue_anchors(encoded, tokenizer, seq_len, split_start, split_end) -> np.ndarray:
    user_id = tokenizer.token_to_id(USER_TAG)
    if user_id is None:
        return np.empty(0, dtype=np.int64)
    positions = np.where(encoded == user_id)[0]
    max_start = split_end - seq_len - 1
    valid = positions[(positions >= split_start) & (positions <= max_start)]
    return valid.astype(np.int64)


def build_train_mask(encoded, tokenizer) -> np.ndarray:
    n = len(encoded)
    user_id = tokenizer.token_to_id(USER_TAG)
    system_id = tokenizer.token_to_id(SYSTEM_TAG)
    bot_id = tokenizer.token_to_id(BOT_TAG)
    end_id = tokenizer.token_to_id(EXAMPLE_END_TAG)
    if user_id is None or bot_id is None:
        return np.ones(n, dtype=bool)
    breakpoints: list[tuple[int, bool]] = [(0, True)]
    for tid in (user_id, system_id, bot_id):
        if tid is None:
            continue
        for p in np.where(encoded == tid)[0]:
            breakpoints.append((int(p), False))
    for p in np.where(encoded == bot_id)[0]:
        if p + 1 < n:
            breakpoints.append((int(p) + 1, True))
    if end_id is not None:
        for p in np.where(encoded == end_id)[0]:
            breakpoints.append((int(p), True))
    breakpoints.sort(key=lambda pair: (pair[0], not pair[1]))
    mask = np.empty(n, dtype=bool)
    for i, (start, state) in enumerate(breakpoints):
        end = breakpoints[i + 1][0] if i + 1 < len(breakpoints) else n
        if end > start:
            mask[start:end] = state
    return mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_cross_entropy(logits, targets):
    valid_count = (targets != IGNORE_INDEX).sum()
    sum_loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, reduction="sum")
    loss = sum_loss / valid_count.clamp(min=1)
    return loss, sum_loss, valid_count


# ---------------------------------------------------------------------------
# Knowledge distillation losses
# ---------------------------------------------------------------------------


def kd_loss_full(student_logits, teacher_logits, temperature):
    """Per-token KL(teacher || student) over the full vocabulary, both softened
    by `temperature`. Inputs (N, V) -> output (N,). (Pre-v3.7 behaviour.)"""
    log_ps = F.log_softmax(student_logits.float() / temperature, dim=-1)
    pt = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(log_ps, pt, reduction="none", log_target=False).sum(dim=-1)


def kd_loss_topk(student_logits, teacher_logits, temperature, k):
    """Per-token top-k distillation loss with the KL decomposed by the chain
    rule (the idea used for LFM2 distillation).

    The teacher's top-k tokens I define a support-matched partition of the
    vocabulary into {I, "everything else"}:

        KL = KL_bin(P_t || P_s) + P_t * KL( p_t|I  ||  p_s|I )

    where P_t / P_s are the probability masses that teacher / student put on I
    (binary term: Bernoulli KL), and p|I are the distributions renormalised
    inside I (conditional term). Student probabilities are full-vocabulary
    softmax values, so the gradient still pushes the student to concentrate
    mass on the teacher's top-k. Inputs (N, V) -> output (N,)."""
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    k = max(1, min(int(k), t.size(-1)))
    log_ps = F.log_softmax(s, dim=-1)
    log_pt = F.log_softmax(t, dim=-1)
    idx = torch.topk(t, k, dim=-1).indices  # temperature does not change the order
    lt = log_pt.gather(-1, idx)
    ls = log_ps.gather(-1, idx)
    log_mass_t = torch.logsumexp(lt, dim=-1)
    log_mass_s = torch.logsumexp(ls, dim=-1)
    mass_t = log_mass_t.exp()
    mass_s = log_mass_s.exp()
    tail_t = (1.0 - mass_t).clamp(min=1e-6)
    tail_s = (1.0 - mass_s).clamp(min=1e-6)
    binary = mass_t * (log_mass_t - log_mass_s) + tail_t * (tail_t.log() - tail_s.log())
    cond_t = lt - log_mass_t.unsqueeze(-1)
    cond_s = ls - log_mass_s.unsqueeze(-1)
    cond_kl = (cond_t.exp() * (cond_t - cond_s)).sum(dim=-1)
    return binary + mass_t * cond_kl


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


def resolve_amp_dtype(device, requested):
    if requested == "off" or device.type != "cuda":
        return None
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_grad_scaler(enabled, device):
    if not enabled or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device, amp_dtype):
    if amp_dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)


def grad_scaler_step_succeeded(optimizer, scaler) -> bool:
    """Run one scaled optimizer step and detect GradScaler overflow skips.

    GradScaler lowers its scale when inf/NaN gradients cause it to skip the
    optimizer update. The LR scheduler, global step counter and EMA must only
    advance when the underlying optimizer actually updates the parameters.
    """
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= scale_before


# ---------------------------------------------------------------------------
# Exponential moving average
# ---------------------------------------------------------------------------


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            k: v.detach().clone()
            for k, v in get_base_model(model).state_dict().items()
            if v.dtype.is_floating_point
        }

    @classmethod
    def from_state(cls, state, decay):
        obj = cls.__new__(cls)
        obj.decay = float(decay)
        obj.updates = int(state.get("updates", 0))
        obj.shadow = state["shadow"]
        return obj

    def state_dict(self):
        return {"updates": self.updates, "shadow": self.shadow}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = get_base_model(model).state_dict()
        keys = list(self.shadow.keys())
        shadow = [self.shadow[k] for k in keys]
        params = [msd[k] for k in keys]
        torch._foreach_mul_(shadow, d)
        torch._foreach_add_(shadow, params, alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model):
        msd = get_base_model(model).state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate(
    model, data, batch_size, seq_len, state_carry, device,
    split_start, split_end, batches, amp_dtype,
    train_mask=None, anchors=None, anchor_ratio=0.0,
):
    model.eval()
    base_model = get_base_model(model)
    gen = torch.Generator(device=data.device)
    gen.manual_seed(1234)
    loss_sum = torch.zeros((), device=data.device)
    token_sum = torch.zeros((), device=data.device)
    span = seq_len * state_carry
    for _ in range(batches):
        x_span, y_span = get_batch(
            data, batch_size, span, split_start, split_end, gen=gen,
            anchors=anchors, anchor_ratio=anchor_ratio, train_mask=train_mask,
        )
        hidden = None
        memory = chunk_memory_reset(base_model, x_span.shape[0], device)
        for k in range(state_carry):
            xk = x_span[:, k * seq_len : (k + 1) * seq_len]
            yk = y_span[:, k * seq_len : (k + 1) * seq_len]
            hidden = chunk_memory_read(base_model, hidden, memory)
            with autocast_context(device, amp_dtype):
                logits, hidden = model(xk, hidden=hidden)
                _loss, sum_loss, valid_count = masked_cross_entropy(
                    logits.reshape(-1, model.vocab_size), yk.reshape(-1),
                )
            loss_sum += sum_loss.detach().float()
            token_sum += valid_count.float()
            if k < state_carry - 1:
                memory = chunk_memory_write(base_model, hidden, memory)
                if memory is not None:
                    memory = memory.detach()
            hidden = detach_hidden(hidden)
    return (loss_sum / torch.clamp(token_sum, min=1.0)).item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


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

    resume_tokenizer = None
    if args.resume:
        resume_payload = load_checkpoint_raw(last_checkpoint_path, device)
        saved_cfg = resume_payload["config"]
        args.shuffle_files = bool(saved_cfg.get("shuffle_files", False))
        args.file_headers = bool(saved_cfg.get("file_headers", True))
        args.fim_rate = float(saved_cfg.get("fim_rate", 0.0) or 0.0)
        resume_tokenizer = tokenizer_from_checkpoint(resume_payload)

    fim_rate = float(args.fim_rate)
    if not 0.0 <= fim_rate <= 1.0:
        raise ValueError("--fim-rate must be between 0 and 1")
    if fim_rate > 0.0:
        if resume_tokenizer is not None:
            fim_supported = all(
                resume_tokenizer.token_to_id(tag) is not None
                for tag in (FIM_PREFIX_TAG, FIM_SUFFIX_TAG, FIM_MIDDLE_TAG)
            )
        else:
            fim_supported = args.tokenizer == "bpe" and _HAS_TOKENIZERS
        if not fim_supported:
            print(
                "[DATA] FIM отключён: нужен BPE-токенизатор со специальными токенами "
                "<|fim_*|> (для char-токенизатора и старых чекпоинтов их нет)."
            )
            fim_rate = 0.0
    args.fim_rate = fim_rate

    corpus = load_texts(
        data_dir, min_chars=args.min_chars, max_file_mb=args.max_file_mb,
        shuffle_files=args.shuffle_files, file_headers=args.file_headers,
        fim_rate=fim_rate,
    )
    corpus_sha = hashlib.sha256(corpus.encode("utf-8", errors="ignore")).hexdigest()

    if args.resume:
        tokenizer = resume_tokenizer
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
    long_epochs = max(0, int(args.long_epochs))
    long_state_carry = max(1, int(args.long_state_carry)) if long_epochs > 0 else state_carry
    max_state_carry = max(state_carry, long_state_carry)
    total_epochs = args.epochs + long_epochs
    if long_epochs > 0 and long_state_carry <= state_carry:
        print("[STAGE] WARNING: --long-state-carry is not larger than --state-carry; "
              "the final stage will not use a longer context.")
    if args.memory_slots > 0 and max_state_carry < 2:
        print("[MODEL] WARNING: --memory-slots has no effect with --state-carry 1 "
              "(memory is read only from the second subchunk).")
    span_len = args.seq_len * state_carry
    max_span_len = args.seq_len * max_state_carry
    if split <= max_span_len + 2:
        raise ValueError("Training portion is too small.")
    if len(encoded) - split <= max_span_len + 2:
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
        anchors_np = build_dialogue_anchors(encoded, tokenizer, max_span_len, 0, split)
        val_anchors_np = build_dialogue_anchors(encoded, tokenizer, max_span_len, split, len(encoded))
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
        ffn_mult=args.ffn_mult,
        norm_type=args.norm_type,
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
        f"ffn_mult={args.ffn_mult} | norm={args.norm_type} | "
        f"local_conv={args.local_conv} | ckpt={args.grad_checkpoint}"
    )
    if long_epochs > 0:
        print(
            f"[STAGE] main: {args.epochs} эпох, carry={state_carry} (span {span_len}) | "
            f"long: {long_epochs} эпох, carry={long_state_carry} "
            f"(span {args.seq_len * long_state_carry}) | "
            f"LR на старте long-стадии = {args.stage_lr_ratio:g} от пика, "
            f"затем линейно до {args.min_lr_ratio:g}"
        )

    optimizer = make_optimizer(model, args, device)
    updates_per_epoch = math.ceil(args.steps_per_epoch / args.grad_accumulation)
    total_updates = total_epochs * updates_per_epoch
    scheduler = make_scheduler(
        optimizer=optimizer, total_steps=total_updates,
        warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
        stage_split_step=args.epochs * updates_per_epoch if long_epochs > 0 else None,
        stage_lr_ratio=args.stage_lr_ratio,
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

    epoch = start_epoch
    while epoch < total_epochs:
        epoch += 1
        in_long_stage = epoch > args.epochs
        epoch_state_carry = long_state_carry if in_long_stage else state_carry
        epoch_span_len = args.seq_len * epoch_state_carry
        if in_long_stage and epoch == args.epochs + 1:
            # val loss with a longer carry is not comparable with the main stage,
            # so "best" is tracked from scratch inside the long stage.
            best_val = math.inf
            bad_epochs = 0
            print(
                f"[STAGE] Стадия длинного контекста: carry={epoch_state_carry} "
                f"(span {epoch_span_len}), best_val сброшен."
            )

        model.train()
        loss_sum = torch.zeros((), device=device)
        token_sum = torch.zeros((), device=device)
        finite_ok = torch.ones((), dtype=torch.bool, device=device)
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, args.steps_per_epoch + 1):
            x_span, y_span = get_batch(
                data, args.batch_size, epoch_span_len, 0, split,
                anchors=dialogue_anchors, anchor_ratio=args.dialogue_anchor_ratio,
                train_mask=train_mask,
            )
            hidden = None
            memory = chunk_memory_reset(base_model, x_span.shape[0], device)
            t_hidden = None
            t_memory = (
                chunk_memory_reset(teacher, x_span.shape[0], device)
                if teacher is not None else None
            )
            full_steps = (args.steps_per_epoch // args.grad_accumulation) * args.grad_accumulation
            group_size = (
                args.grad_accumulation if step <= full_steps
                else args.steps_per_epoch - full_steps
            )

            for k in range(epoch_state_carry):
                xk = x_span[:, k * args.seq_len : (k + 1) * args.seq_len]
                yk = y_span[:, k * args.seq_len : (k + 1) * args.seq_len]
                hidden = chunk_memory_read(base_model, hidden, memory)

                with autocast_context(device, amp_dtype):
                    logits, hidden = model(xk, hidden=hidden)
                    raw_loss, sum_loss, valid_count = masked_cross_entropy(
                        logits.reshape(-1, vocab_size), yk.reshape(-1),
                    )

                    if teacher is not None:
                        # The teacher now sees the same carried context as the
                        # student (hidden state and chunk memory are carried
                        # across subchunks and reset at the start of a span).
                        with torch.no_grad():
                            t_hidden = chunk_memory_read(teacher, t_hidden, t_memory)
                            t_logits, t_hidden = teacher(xk, hidden=t_hidden)
                            t_hidden = detach_hidden(t_hidden)
                            if k < epoch_state_carry - 1:
                                t_memory = chunk_memory_write(teacher, t_hidden, t_memory)
                        T = args.kd_temperature
                        kd_weights = (yk.reshape(-1) != IGNORE_INDEX).float()
                        s_flat = logits.reshape(-1, vocab_size)
                        t_flat = t_logits.reshape(-1, vocab_size)
                        if args.kd_topk > 0:
                            kd_per_token = kd_loss_topk(s_flat, t_flat, T, args.kd_topk)
                        else:
                            kd_per_token = kd_loss_full(s_flat, t_flat, T)
                        kd_loss = (kd_per_token * kd_weights).sum() / kd_weights.sum().clamp(min=1.0)
                        raw_loss = (1 - args.kd_alpha) * raw_loss + args.kd_alpha * kd_loss * (T * T)

                    loss = raw_loss / (epoch_state_carry * group_size)

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_sum += sum_loss.detach().float()
                token_sum += valid_count.float()
                finite_ok &= torch.isfinite(raw_loss.detach())
                if k < epoch_state_carry - 1:
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
                    update_succeeded = grad_scaler_step_succeeded(optimizer, scaler)
                else:
                    optimizer.step()
                    update_succeeded = True
                optimizer.zero_grad(set_to_none=True)
                if update_succeeded:
                    scheduler.step()
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
        tokens_seen = args.batch_size * epoch_span_len * args.steps_per_epoch
        tokens_per_sec = tokens_seen / max(elapsed, 1e-9)

        val_loss = evaluate(
            model, data, args.batch_size, args.seq_len, epoch_state_carry,
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
                model, data, args.batch_size, args.seq_len, epoch_state_carry,
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
            f"[EPOCH {epoch:03d}/{total_epochs}]{' [long]' if in_long_stage else ''} "
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

        if epoch % 5 == 0 or epoch == total_epochs or improved:
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

        if epoch % max(1, args.save_every) == 0 or epoch == total_epochs:
            save_checkpoint(
                last_checkpoint_path, model, optimizer, scheduler, scaler,
                epoch, global_step, train_loss, val_loss, best_val,
                tokenizer, args,
                ema_state=ema.state_dict() if ema is not None else None,
                corpus_sha256=corpus_sha,
            )
            print(f"[SAVE] Last -> {last_checkpoint_path}")

        if args.patience > 0 and bad_epochs >= args.patience:
            if not in_long_stage and long_epochs > 0:
                # Do not skip the long-context stage: fast-forward the LR
                # schedule to its start and continue from there.
                for _ in range((args.epochs - epoch) * updates_per_epoch):
                    scheduler.step()
                print(
                    f"[EARLY STOP] {args.patience} эпох без улучшения в основной стадии "
                    f"— переход к стадии длинного контекста."
                )
                epoch = args.epochs
                bad_epochs = 0
                continue
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
        "long_epochs": long_epochs,
        "long_state_carry": long_state_carry,
        "fim_rate": fim_rate,
        "kd_topk": args.kd_topk if teacher is not None else None,
        "ffn_mult": args.ffn_mult,
        "norm_type": args.norm_type,
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
