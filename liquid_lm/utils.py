"""Device selection, seeding, Tensor Core setup and metric logging."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Device, seeding and Tensor Cores
# ---------------------------------------------------------------------------


def configure_tensor_cores(device: torch.device) -> None:
    if device.type != "cuda":
        return
    matmul = torch.backends.cuda.matmul
    new_api_ok = False
    if hasattr(matmul, "fp32_precision"):
        try:
            matmul.fp32_precision = "tf32"
            new_api_ok = True
        except Exception:
            new_api_ok = False
    if not new_api_ok:
        if hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if hasattr(matmul, name):
            try:
                setattr(matmul, name, True)
            except Exception:
                pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was selected, but torch.cuda.is_available() == False."
        )
    return device


def print_device_info(device: torch.device) -> None:
    print(f"[SYSTEM] Device: {device}")
    print(f"[SYSTEM] PyTorch: {torch.__version__}")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        print(f"[GPU] {name}")
        print(f"[GPU] VRAM: {total:.2f} GB")
        print(f"[GPU] CUDA: {torch.version.cuda}")
        configure_tensor_cores(device)
        matmul = torch.backends.cuda.matmul
        if hasattr(matmul, "fp32_precision"):
            tf32 = matmul.fp32_precision
        else:
            tf32 = getattr(matmul, "allow_tf32", None)
        red16 = getattr(matmul, "allow_fp16_reduced_precision_reduction", "?")
        redbf = getattr(matmul, "allow_bf16_reduced_precision_reduction", "?")
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        print(f"[TC] TF32={tf32} | fp16-RPR={red16} | bf16-RPR={redbf} | bf16={bf16_ok}")


def cuda_memory_report() -> str:
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"VRAM allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------


class TrainLogger:
    """
    Thin wrapper over TensorBoard/W&B — a single point for logging training
    metrics (train/val loss, perplexity, lr, grad norm, speed).

    backend="none" (default) — a complete no-op: imports nothing, writes
    nothing to disk/network, .log() does nothing. With this flag, training
    behavior does not change AT ALL compared to a version without logging
    — important for those who simply do not use the feature.

    If the requested backend is not installed (no tensorboard/wandb
    package), prints a warning ONCE at startup and silently falls back to
    "none" — training must not crash because of a missing dependency for
    an optional logging feature.
    """

    def __init__(
        self,
        backend: str,
        run_name: str,
        logdir: str = "runs",
        wandb_project: str = "liquid-lm",
        config: dict | None = None,
    ) -> None:
        self.backend = backend
        self._tb = None
        self._wandb_run = None

        if backend == "tensorboard":
            if not _TENSORBOARD_AVAILABLE:
                print(
                    "[LOG] --log-backend tensorboard was requested, but torch.utils.tensorboard "
                    "is unavailable (pip install tensorboard). Logging is disabled."
                )
                self.backend = "none"
                return
            self._tb = _TBWriter(log_dir=str(Path(logdir) / run_name))
        elif backend == "wandb":
            if not _WANDB_AVAILABLE:
                print(
                    "[LOG] --log-backend wandb was requested, but the wandb package is not installed "
                    "(pip install wandb, then run wandb login). Logging is disabled."
                )
                self.backend = "none"
                return
            self._wandb_run = _wandb.init(project=wandb_project, name=run_name, config=config or {})

    def log(self, metrics: dict, step: int) -> None:
        if self.backend == "none":
            return
        if self._tb is not None:
            for key, value in metrics.items():
                self._tb.add_scalar(key, value, global_step=step)
        if self._wandb_run is not None:
            _wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wandb_run is not None:
            _wandb.finish()
