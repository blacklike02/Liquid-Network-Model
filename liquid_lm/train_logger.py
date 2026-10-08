"""Thin wrapper over TensorBoard / Weights & Biases for metric logging."""

from __future__ import annotations

from pathlib import Path

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
