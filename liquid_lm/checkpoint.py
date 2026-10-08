"""Saving and loading checkpoints."""

from __future__ import annotations

import torch

from .model.utils import get_base_model


def save_checkpoint(
    path, model, optimizer, scheduler, scaler, epoch, global_step,
    train_loss, val_loss, best_val, tokenizer, args,
    ema_state=None, corpus_sha256=None,
):
    base_model = get_base_model(model)
    payload = {
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "ema_state": ema_state,
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "tokenizer": tokenizer.to_config(),
        "corpus_sha256": corpus_sha256,
        "config": vars(args),
    }
    torch.save(payload, path)


_TRUST_CHECKPOINTS = False


def set_trust_checkpoints(value: bool) -> None:
    """Allow unsafe checkpoint loading (weights_only=False) — the --trust-checkpoint flag."""
    global _TRUST_CHECKPOINTS
    _TRUST_CHECKPOINTS = bool(value)


def load_checkpoint_raw(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        if not _TRUST_CHECKPOINTS:
            raise RuntimeError(
                f"Не удалось безопасно загрузить {path} (weights_only=True): {exc}\n"
                "If the checkpoint is yours and trusted, add --trust-checkpoint "
                "(pickle can execute arbitrary code)."
            ) from exc
        print("[CHECKPOINT] WARNING: unsafe loading (--trust-checkpoint).")
        return torch.load(path, map_location=device, weights_only=False)


def print_checkpoint_info(checkpoint):
    tokenizer_config = checkpoint.get("tokenizer", {})
    kind = tokenizer_config.get("kind", "char" if "char_to_idx" in checkpoint else "?")
    train_loss = checkpoint.get("train_loss")
    val_loss = checkpoint.get("val_loss")
    print("[CHECKPOINT]")
    print(f"  epoch       : {checkpoint.get('epoch')}")
    print(f"  global_step : {checkpoint.get('global_step')}")
    print(f"  train_loss  : {train_loss if train_loss is None else round(float(train_loss), 6)}")
    print(f"  val_loss    : {val_loss if val_loss is None else round(float(val_loss), 6)}")
    print(f"  tokenizer   : {kind}")
    print(f"  config      : {checkpoint.get('config', {})}")
