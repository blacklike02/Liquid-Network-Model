from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

if os.name == "posix":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
try:
    from charset_normalizer import from_path as _charset_from_path
    _HAS_CHARSET_NORMALIZER = True
except ImportError:
    _HAS_CHARSET_NORMALIZER = False
try:
    from tokenizers import Tokenizer as _HFTokenizer
    from tokenizers.models import BPE as _BPEModel
    from tokenizers.trainers import BpeTrainer as _BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as _ByteLevelPreTokenizer
    from tokenizers.decoders import ByteLevel as _ByteLevelDecoder
    _HAS_TOKENIZERS = True
except ImportError:
    _HAS_TOKENIZERS = False

DEFAULT_CORPUS = """
This is an example corpus for the Liquid Language Model.

The neural network receives a sequence of tokens and learns to
predict the next token. After training, the hidden state
contains a compressed representation of the text context.

Liquid neural network treats the hidden state as a dynamic
system. The state changes according to a differential equation rather than
a single discrete update.

Quality depends heavily on the amount of data, corpus diversity,
hidden-state size, context length, and the number of training steps.
""".strip()

SUPPORTED_SUFFIXES = {".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".parquet", ".jsonl"}

EXCLUDED_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
}

TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1")


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


def describe_wiring(model) -> str:
    base = get_base_model(model)
    if getattr(base, "wiring", "dense") != "ncp":
        return "dense"
    frac = base.cells[0].recurrent_param_fraction
    if frac is None:
        return "dense"
    return f"ncp (рекуррентная часть ~{frac * 100:.0f}% от dense по параметрам/FLOPs)"


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
    Тонкая обёртка над TensorBoard/W&B — единая точка логирования метрик
    обучения (train/val loss, perplexity, lr, grad norm, скорость).

    backend="none" (по умолчанию) — полный no-op: ничего не импортирует,
    ничего не пишет на диск/в сеть, .log() ничего не делает. Поведение
    обучения при этом флаге не меняется НИКАК по сравнению с версией без
    логирования вообще — важно для тех, кто просто не пользуется фичей.

    Если запрошенный backend не установлен (нет пакета tensorboard/wandb),
    печатает предупреждение ОДИН раз при старте и тихо откатывается на
    "none" — обучение не должно падать из-за отсутствующей зависимости
    для необязательной фичи логирования.
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


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


USER_TAG = "<|user|>"
BOT_TAG = "<|bot|>"
SYSTEM_TAG = "<|system|>"
EXAMPLE_END_TAG = "<|end|>"

DIALOGUE_SPECIAL_TOKENS = [USER_TAG, BOT_TAG, SYSTEM_TAG, EXAMPLE_END_TAG]

_ROLE_TAG_BY_NAME = {
    "user": USER_TAG, "human": USER_TAG,
    "assistant": BOT_TAG, "gpt": BOT_TAG, "bot": BOT_TAG, "model": BOT_TAG,
    "system": SYSTEM_TAG,
}

_QA_KEY_PAIRS = [
    ("instruction", "output"),
    ("instruction", "response"),
    ("question", "answer"),
    ("prompt", "response"),
    ("prompt", "completion"),
    ("input", "output"),
]


def _format_turn(role_tag: str, text: str) -> str:
    return f"{role_tag}\n{text.strip()}"


def dialogue_obj_to_text(obj) -> str | None:
    if not isinstance(obj, dict):
        return None
    for list_key, role_key, text_key in (
        ("messages", "role", "content"),
        ("conversations", "from", "value"),
    ):
        turns_raw = obj.get(list_key)
        if isinstance(turns_raw, list) and turns_raw:
            turns = []
            for msg in turns_raw:
                if not isinstance(msg, dict):
                    continue
                role_tag = _ROLE_TAG_BY_NAME.get(
                    str(msg.get(role_key, "")).lower(), USER_TAG
                )
                content = msg.get(text_key)
                if isinstance(content, str) and content.strip():
                    turns.append(_format_turn(role_tag, content))
            if turns:
                return "\n".join(turns)
    for q_key, a_key in _QA_KEY_PAIRS:
        question = obj.get(q_key)
        answer = obj.get(a_key)
        if (
            isinstance(question, str) and question.strip()
            and isinstance(answer, str) and answer.strip()
        ):
            extra_input = obj.get("input")
            if q_key != "input" and isinstance(extra_input, str) and extra_input.strip():
                question = f"{question.strip()}\n{extra_input.strip()}"
            return _format_turn(USER_TAG, question) + "\n" + _format_turn(BOT_TAG, answer)
    return None


def json_to_text(path: Path) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return ""
    if isinstance(obj, list):
        parts = []
        for item in obj:
            dialogue = dialogue_obj_to_text(item)
            if dialogue:
                parts.append(dialogue + f"\n{EXAMPLE_END_TAG}")
            else:
                extracted = _extract_text_from_obj(item)
                if extracted:
                    parts.append(extracted)
        return "\n\n".join(parts)
    dialogue = dialogue_obj_to_text(obj)
    if dialogue:
        return dialogue + f"\n{EXAMPLE_END_TAG}"
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _extract_text_from_obj(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        parts = [_extract_text_from_obj(item) for item in obj]
        return "\n".join(p for p in parts if p)
    if isinstance(obj, dict):
        for key in ("text", "content", "article", "body", "document"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
        parts = []
        for value in obj.values():
            chunk = _extract_text_from_obj(value)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts)
    return str(obj)


def jsonl_to_text(path: Path) -> str:
    lines_out: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    lines_out.append(raw_line)
                    continue
                dialogue = dialogue_obj_to_text(obj)
                if dialogue:
                    lines_out.append(dialogue + f"\n{EXAMPLE_END_TAG}")
                else:
                    lines_out.append(_extract_text_from_obj(obj))
    except OSError:
        return ""
    return "\n\n".join(lines_out)


def _dataframe_to_text(df: "pd.DataFrame") -> str:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for q_key, a_key in _QA_KEY_PAIRS:
        if q_key in cols_lower and a_key in cols_lower:
            q_col, a_col = cols_lower[q_key], cols_lower[a_key]
            input_col = cols_lower.get("input") if q_key != "input" else None
            questions = df[q_col].tolist()
            answers = df[a_col].tolist()
            extras = df[input_col].tolist() if input_col is not None else None
            parts: list[str] = []
            for i in range(len(questions)):
                q_raw, a_raw = questions[i], answers[i]
                if pd.isna(q_raw) or pd.isna(a_raw):
                    continue
                question, answer = str(q_raw).strip(), str(a_raw).strip()
                if not question or not answer:
                    continue
                if extras is not None:
                    extra_raw = extras[i]
                    extra = "" if pd.isna(extra_raw) else str(extra_raw).strip()
                    if extra:
                        question = f"{question}\n{extra}"
                parts.append(
                    _format_turn(USER_TAG, question)
                    + "\n"
                    + _format_turn(BOT_TAG, answer)
                    + f"\n{EXAMPLE_END_TAG}"
                )
            if parts:
                return "\n\n".join(parts)
    for preferred in ("text", "content", "article", "body", "document"):
        if preferred in df.columns:
            series = df[preferred].dropna().astype(str)
            return "\n".join(series.tolist())
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    parts = []
    for col in text_cols:
        parts.extend(df[col].dropna().astype(str).tolist())
    return "\n".join(parts)


def parquet_to_text(path: Path) -> str:
    if not _HAS_PANDAS:
        print(f"[DATA] Пропуск {path}: pandas не установлен")
        return ""
    try:
        df = pd.read_parquet(path)
        return _dataframe_to_text(df)
    except Exception as exc:
        print(f"[DATA] Ошибка чтения {path}: {exc}")
        return ""


def tabular_to_text(path: Path, sep: str) -> str:
    if not _HAS_PANDAS:
        text, _ = read_text_robust(path)
        return text
    try:
        df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
        return _dataframe_to_text(df)
    except Exception as exc:
        print(f"[DATA] Ошибка чтения {path} как таблицы: {exc}")
        text, _ = read_text_robust(path)
        return text


def read_text_robust(path: Path) -> tuple[str, str | None]:
    if _HAS_CHARSET_NORMALIZER:
        try:
            best = _charset_from_path(path).best()
        except OSError:
            return "", None
        if best is not None:
            return str(best), best.encoding
    for encoding in TEXT_ENCODINGS[:-1]:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
        except OSError:
            return "", None
    try:
        return path.read_text(encoding="latin-1"), f"low-confidence:{TEXT_ENCODINGS[-1]}"
    except OSError:
        return "", None


def load_one_file(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json_to_text(path), None
    if suffix == ".jsonl":
        return jsonl_to_text(path), None
    if suffix == ".parquet":
        return parquet_to_text(path), None
    if suffix == ".csv":
        return tabular_to_text(path, sep=","), None
    if suffix == ".tsv":
        return tabular_to_text(path, sep="\t"), None
    return read_text_robust(path)


def load_texts(
    data_dir: Path, min_chars: int = 200, max_file_mb: float = 0.0,
    shuffle_files: bool = False, file_headers: bool = True,
) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    all_candidates = (
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    files = sorted(
        p for p in all_candidates
        if not any(part in EXCLUDED_DIR_NAMES for part in p.relative_to(data_dir).parts)
    )
    if not files:
        raise ValueError(
            f"В {data_dir} нет файлов корпуса ({', '.join(sorted(SUPPORTED_SUFFIXES))}). "
            "Put text files there and run again."
        )
    if shuffle_files:
        files = sorted(
            files,
            key=lambda p: hashlib.sha256(
                p.relative_to(data_dir).as_posix().encode("utf-8")
            ).hexdigest(),
        )
    max_file_bytes = int(max_file_mb * 1024 * 1024) if max_file_mb > 0 else None
    parts: list[str] = []
    seen_hashes: set[str] = set()
    stats_by_suffix: dict[str, dict[str, int]] = {}
    skipped_too_large: list[str] = []
    skipped_empty: list[str] = []
    skipped_duplicate: list[str] = []
    low_confidence_encodings: list[str] = []
    used_files = 0
    for path in files:
        suffix = path.suffix.lower()
        bucket = stats_by_suffix.setdefault(suffix, {"files": 0, "chars": 0})
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if max_file_bytes is not None and size > max_file_bytes:
            skipped_too_large.append(f"{path} ({size / (1024 * 1024):.1f} MB)")
            continue
        if size == 0:
            skipped_empty.append(str(path))
            continue
        raw, encoding_used = load_one_file(path)
        text = clean_text(raw)
        if not text:
            skipped_empty.append(str(path))
            continue
        if encoding_used and encoding_used.startswith("low-confidence"):
            low_confidence_encodings.append(str(path))
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        if digest in seen_hashes:
            skipped_duplicate.append(str(path))
            continue
        seen_hashes.add(digest)
        if file_headers:
            parts.append(f"\n\n===== {path.name} =====\n\n{text}")
        else:
            parts.append(f"\n\n{text}")
        bucket["files"] += 1
        bucket["chars"] += len(text)
        used_files += 1
    corpus = clean_text("".join(parts))
    if len(corpus) < min_chars:
        raise ValueError(
            f"Корпус слишком маленький: {len(corpus)} символов. Нужно хотя бы {min_chars}."
        )
    print(f"[DATA] Найдено файлов: {len(files)} | использовано: {used_files}")
    for suffix in sorted(stats_by_suffix):
        s = stats_by_suffix[suffix]
        if s["files"]:
            print(f"[DATA]   {suffix}: {s['files']} файлов, {s['chars']:,} символов")
    if skipped_too_large:
        print(f"[DATA] Пропущено (больше {max_file_mb:.0f} MB): {len(skipped_too_large)}")
    if skipped_empty:
        print(f"[DATA] Пропущено (пусто/не прочиталось): {len(skipped_empty)}")
    if skipped_duplicate:
        print(f"[DATA] Пропущено полных дублей: {len(skipped_duplicate)}")
    if low_confidence_encodings:
        print(
            f"[DATA] ВНИМАНИЕ: {len(low_confidence_encodings)} файл(ов) прочитаны "
            f"с неуверенным определением кодировки (latin-1 fallback):"
        )
        for name in low_confidence_encodings[:10]:
            print(f"[DATA]   {name}")
    print(f"[DATA] Итоговый корпус: {len(corpus):,} символов")
    return corpus


class LockedDropout(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)


try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _ltc_tail_kernel(
        h_ptr, cand_ptr, tau_raw_ptr, w_ptr, out_ptr,
        sub_dt, min_tau, n_elements, H,
        MULTI_TAU: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        h = tl.load(h_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        cand = tl.load(cand_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if MULTI_TAU:
            row = offs // H
            col = offs - row * H
            base = row * (3 * H) + col
            r0 = tl.load(tau_raw_ptr + base, mask=mask, other=0.0).to(tl.float32) - 2.0
            r1 = tl.load(tau_raw_ptr + base + H, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(tau_raw_ptr + base + 2 * H, mask=mask, other=0.0).to(tl.float32) + 2.0
            sp0 = tl.maximum(r0, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r0)))
            sp1 = tl.maximum(r1, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r1)))
            sp2 = tl.maximum(r2, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r2)))
            w0 = tl.load(w_ptr).to(tl.float32)
            w1 = tl.load(w_ptr + 1).to(tl.float32)
            w2 = tl.load(w_ptr + 2).to(tl.float32)
            tau = w0 * (sp0 + 0.05) + w1 * (sp1 + 0.01) + w2 * (sp2 + 0.001)
        else:
            r = tl.load(tau_raw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            tau = tl.maximum(r, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(r))) + min_tau
        decay = tl.exp(-sub_dt / tau)
        res = decay * h + (1.0 - decay) * cand
        tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def _ltc_tail_triton(h, candidate, tau_raw, mix_w, sub_dt, min_tau, multi_tau):
    """Обёртка над _ltc_tail_kernel. ТОЛЬКО ИНФЕРЕНС: у ядра нет backward
    (см. LiquidCell._use_triton_tail — при включённом autograd оно не
    вызывается вовсе)."""
    h = h.contiguous()
    candidate = candidate.contiguous()
    tau_raw = tau_raw.contiguous()
    B, H = h.shape
    out = torch.empty_like(h)
    n = B * H
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _ltc_tail_kernel[grid](
        h, candidate, tau_raw, mix_w if mix_w is not None else h, out,
        float(sub_dt), float(min_tau), n, H,
        MULTI_TAU=bool(multi_tau), BLOCK=BLOCK,
    )
    return out


class NCPRecurrent(nn.Module):
    """
    Блочно-структурированная замена ОДНОЙ dense (H,H) рекуррентной матрицы
    (h_proj в ltc-вентиле, ff1_h/ff2_h в cfc) — в духе Neural Circuit
    Policies (Lechner et al., "Neural circuit policies enabling auditable
    autonomy", Nature MI 2020): нейроны разбиты на inter/command/motor,
    связи разрешены только по путям

        inter   -> inter    (рекуррентно)
        inter   -> command
        command -> command  (рекуррентно)
        command -> motor
        motor   -> command  (обратная связь, замыкает контур)

    Для каждого разрешённого пути — своя ОТДЕЛЬНАЯ, физически МЕНЬШАЯ матрица
    (nn.Linear на inter_size/command_size/motor_size, а не на hidden_size).
    Маска поверх dense-матрицы в PyTorch не даёт прироста скорости — обычный
    dense-матмул всё равно считает все H*H произведений. Здесь же
    несуществующих связей в принципе НЕТ ни в весах, ни в вычислениях —
    реальное, а не "бумажное" сокращение и параметров, и FLOPs.

    Внутри каждого разрешённого пути связи по-прежнему ПОЛНОСВЯЗНЫ (не
    прорежены поэлементно).

    ВАЖНО (сознательное упрощение): полярность синапсов (excitatory/
    inhibitory через reversal potential в сигмоидальной проводимости из
    оригинальной LTC-биофизики) не воспроизводится — знак и величину веса на
    разрешённых связях по-прежнему свободно учит градиентный спуск.
    """

    def __init__(
        self,
        hidden_size: int,
        command_frac: float = 0.4,
        motor_frac: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_size < 3:
            raise ValueError("--wiring ncp requires hidden_size >= 3")
        if not (0.0 < command_frac < 1.0) or not (0.0 < motor_frac < 1.0):
            raise ValueError("--ncp-command-frac and --ncp-motor-frac must be in (0, 1)")
        if command_frac + motor_frac >= 1.0:
            raise ValueError("--ncp-command-frac + --ncp-motor-frac must be < 1")

        motor_size = max(1, round(hidden_size * motor_frac))
        command_size = max(1, round(hidden_size * command_frac))
        inter_size = hidden_size - motor_size - command_size
        if inter_size < 1:
            raise ValueError(
                f"hidden_size={hidden_size} слишком мал для "
                f"ncp_motor_frac={motor_frac}, ncp_command_frac={command_frac}"
            )

        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.command_size = command_size
        self.motor_size = motor_size

        self.w_ii = nn.Linear(inter_size, inter_size, bias=False)
        self.w_ic = nn.Linear(inter_size, command_size, bias=False)
        self.w_cc = nn.Linear(command_size, command_size, bias=False)
        self.w_cm = nn.Linear(command_size, motor_size, bias=False)
        self.w_mc = nn.Linear(motor_size, command_size, bias=False)  # обратная связь
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.w_ii, self.w_ic, self.w_cc, self.w_cm, self.w_mc):
            nn.init.orthogonal_(lin.weight, gain=0.99)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_inter, h_command, h_motor = torch.split(
            h, [self.inter_size, self.command_size, self.motor_size], dim=-1
        )
        inter_out = self.w_ii(h_inter)
        command_out = self.w_ic(h_inter) + self.w_cc(h_command) + self.w_mc(h_motor)
        motor_out = self.w_cm(h_command)
        return torch.cat([inter_out, command_out, motor_out], dim=-1)

    @property
    def param_fraction(self) -> float:
        """Доля параметров/FLOPs относительно полной dense H x H матрицы."""
        H, Hi, Hc, Hm = self.hidden_size, self.inter_size, self.command_size, self.motor_size
        used = Hi * Hi + Hi * Hc + Hc * Hc + 2 * Hc * Hm
        return used / (H * H)


class LiquidCell(nn.Module):
    """
    Два режима вентиля (gate_mode):

    - "ltc" (по умолчанию): физически мотивированная линейная ODE
      dh/dt = (candidate - h) / tau, "заморожен" candidate и tau на подшаге и
      решён точно (экспоненциальный интегратор). Внутри подшага candidate/tau
      считаются либо один раз в начале подшага (ode_solver="euler"), либо
      дважды с усреднением (ode_solver="heun" — predictor-corrector, второй
      порядок точности; вдвое дороже за подшаг).

    - "cfc": closed-form формула из Hasani et al., "Closed-form
      Continuous-time Neural Networks" (Nature Machine Intelligence, 2022):
      два кандидата состояния (ff1, ff2) и обучаемая сигмоидная интерполяция
      по времени между ними. При gate_mode="cfc" параметр --multi-tau не
      имеет смысла (нет tau) и игнорируется.

    wiring="dense" (по умолчанию) — обычная полносвязная рекуррентная
    матрица. wiring="ncp" — та же матрица заменяется на NCPRecurrent.
    В cfc-режиме ff1_h и ff2_h — два независимых экземпляра NCPRecurrent.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ode_unfolds: int = 1,
        min_tau: float = 0.05,
        dropout: float = 0.0,
        zoneout: float = 0.0,
        multi_tau: bool = True,
        gate_mode: str = "ltc",
        ode_solver: str = "euler",
        wiring: str = "dense",
        ncp_command_frac: float = 0.4,
        ncp_motor_frac: float = 0.2,
    ) -> None:
        super().__init__()
        if ode_unfolds < 1:
            raise ValueError("ode_unfolds must be >= 1")
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be > 0")
        if gate_mode not in ("ltc", "cfc"):
            raise ValueError(f"Неизвестный gate_mode: {gate_mode!r} (ожидался 'ltc' или 'cfc')")
        if ode_solver not in ("euler", "heun"):
            raise ValueError(f"Неизвестный ode_solver: {ode_solver!r} (ожидался 'euler' или 'heun')")
        if wiring not in ("dense", "ncp"):
            raise ValueError(f"Неизвестный wiring: {wiring!r} (ожидался 'dense' или 'ncp')")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        self.min_tau = min_tau
        self.zoneout = zoneout
        self.gate_mode = gate_mode
        self.ode_solver = ode_solver
        self.wiring = wiring
        self.kernel = "torch"
        self.multi_tau = bool(multi_tau) and gate_mode == "ltc"
        self._inv_unfolds = 1.0 / ode_unfolds

        if gate_mode == "cfc":
            self.ff1_x = nn.Linear(input_size, hidden_size)
            self.ff2_x = nn.Linear(input_size, hidden_size)
            if wiring == "ncp":
                self.ff1_h_ncp = NCPRecurrent(hidden_size, ncp_command_frac, ncp_motor_frac)
                self.ff2_h_ncp = NCPRecurrent(hidden_size, ncp_command_frac, ncp_motor_frac)
            else:
                self.ff1_h = nn.Linear(hidden_size, hidden_size, bias=False)
                self.ff2_h = nn.Linear(hidden_size, hidden_size, bias=False)
            self.time_net = nn.Linear(input_size + hidden_size, hidden_size * 2)
        else:
            self.x_proj = nn.Linear(input_size, hidden_size)
            if wiring == "ncp":
                self.h_proj_ncp = NCPRecurrent(hidden_size, ncp_command_frac, ncp_motor_frac)
            else:
                self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            tau_out = hidden_size * 3 if self.multi_tau else hidden_size
            self.tau_net = nn.Sequential(
                nn.Linear(input_size + hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, tau_out),
            )
            if self.multi_tau:
                self.tau_mix = nn.Parameter(torch.zeros(3))

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    @property
    def recurrent_param_fraction(self) -> float | None:
        """Доля параметров/FLOPs рекуррентной матрицы относительно полной
        dense H x H (только при wiring="ncp", иначе None)."""
        if self.wiring != "ncp":
            return None
        ncp = self.ff1_h_ncp if self.gate_mode == "cfc" else self.h_proj_ncp
        return ncp.param_fraction

    def reset_parameters(self) -> None:
        if self.gate_mode == "cfc":
            for lin in (self.ff1_x, self.ff2_x):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            if self.wiring != "ncp":
                for lin in (self.ff1_h, self.ff2_h):
                    nn.init.orthogonal_(lin.weight, gain=0.99)
            nn.init.xavier_uniform_(self.time_net.weight)
            nn.init.zeros_(self.time_net.bias)
            return
        if self.wiring != "ncp":
            nn.init.orthogonal_(self.h_proj.weight, gain=0.99)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)
        nn.init.xavier_uniform_(self.tau_net[0].weight)
        nn.init.zeros_(self.tau_net[0].bias)
        nn.init.xavier_uniform_(self.tau_net[2].weight)
        nn.init.constant_(self.tau_net[2].bias, 0.1)

    def fused_h_weights(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Готовит один большой (weight, bias) для F.linear ОДНИМ вызовом —
        только при wiring="dense". При wiring="ncp" рекуррентная часть — уже
        НЕСКОЛЬКО физически меньших матмулов, их некуда "склеивать" — вместо
        этого используется _fused_from_h(), а этот метод возвращает (None, None)."""
        if self.wiring == "ncp":
            return None, None
        H = self.hidden_size
        if self.gate_mode == "cfc":
            fused_w = torch.cat(
                [self.ff1_h.weight, self.ff2_h.weight, self.time_net.weight[:, self.input_size:]],
                dim=0,
            )
            fused_b = torch.cat([fused_w.new_zeros(2 * H), self.time_net.bias])
            return fused_w, fused_b
        fused_w = torch.cat(
            [self.h_proj.weight, self.tau_net[0].weight[:, self.input_size:]],
            dim=0,
        )
        fused_b = torch.cat([fused_w.new_zeros(H), self.tau_net[0].bias])
        return fused_w, fused_b

    def _fused_from_h(
        self, h: torch.Tensor,
        fused_w: torch.Tensor | None, fused_b: torch.Tensor | None,
    ) -> torch.Tensor:
        """Считает "fused"-проекцию текущего h: при wiring="dense" — одним
        F.linear; при wiring="ncp" — через блочно-структурированный
        NCPRecurrent + отдельный dense-матмул для tau/time-гейтинга,
        результаты конкатенируются в тот же (B, 2H) / (B, 4H) вид, который
        ожидают _ltc_step/_cfc_step."""
        if self.wiring != "ncp":
            return F.linear(h, fused_w, fused_b)
        if self.gate_mode == "cfc":
            ff1h = self.ff1_h_ncp(h)
            ff2h = self.ff2_h_ncp(h)
            time_h = F.linear(h, self.time_net.weight[:, self.input_size:], self.time_net.bias)
            return torch.cat([ff1h, ff2h, time_h], dim=-1)
        h_proj_out = self.h_proj_ncp(h)
        tau_h = F.linear(h, self.tau_net[0].weight[:, self.input_size:], self.tau_net[0].bias)
        return torch.cat([h_proj_out, tau_h], dim=-1)

    def precompute_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Часть проекций, зависящая только от x (не от h) — считается
        один раз на всю последовательность заранее (см.
        LiquidLanguageModel.forward), а не заново на каждом шаге."""
        if self.gate_mode == "cfc":
            xp = torch.cat([self.ff1_x(x), self.ff2_x(x)], dim=-1)
            tx = F.linear(x, self.time_net.weight[:, :self.input_size])
        else:
            xp = self.x_proj(x)
            tx = F.linear(x, self.tau_net[0].weight[:, :self.input_size])
        return xp, tx

    def _use_triton_tail(self, h: torch.Tensor) -> bool:
        """Triton-хвост включается ТОЛЬКО когда безопасно: явно запрошен
        (--kernel triton), пакет есть, тензор на CUDA, это ltc+euler, ячейка
        в eval-режиме и autograd выключен (у ядра нет backward)."""
        return (
            self.kernel == "triton"
            and _TRITON_AVAILABLE
            and h.is_cuda
            and not self.training
            and not torch.is_grad_enabled()
            and self.gate_mode == "ltc"
            and self.ode_solver == "euler"
        )

    def _ltc_step(
        self, h: torch.Tensor, xp: torch.Tensor, tx: torch.Tensor,
        sub_dt: float, fused_w: torch.Tensor, fused_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Один explicit-подшаг ltc-вентиля: candidate/tau оцениваются в
        точке h (начале подшага), затем ODE dh/dt=(candidate-h)/tau
        решается точно на этом подшаге при "замороженных" candidate/tau.
        Возвращает (h_new, candidate, tau) — candidate и tau нужны наружу
        для heun-корректора (см. forward)."""
        H = self.hidden_size
        fused = self._fused_from_h(h, fused_w, fused_b)
        candidate = torch.tanh(xp + fused[:, :H])
        tau_raw = self.tau_net[2](torch.tanh(fused[:, H:] + tx))
        if self._use_triton_tail(h):
            mix_w = F.softmax(self.tau_mix, dim=0) if self.multi_tau else None
            h_new = _ltc_tail_triton(
                h, candidate, tau_raw, mix_w, sub_dt, self.min_tau, self.multi_tau
            )
            return h_new, candidate, None
        if self.multi_tau:
            tau_raw = tau_raw.view(-1, 3, H)
            tau_fast = F.softplus(tau_raw[:, 0, :] - 2.0) + 0.05
            tau_med = F.softplus(tau_raw[:, 1, :]) + 0.01
            tau_slow = F.softplus(tau_raw[:, 2, :] + 2.0) + 0.001
            w = F.softmax(self.tau_mix, dim=0)
            tau = w[0] * tau_fast + w[1] * tau_med + w[2] * tau_slow
        else:
            tau = F.softplus(tau_raw) + self.min_tau
        decay = torch.exp(-sub_dt / tau)
        h_new = decay * h + (1.0 - decay) * candidate
        return h_new, candidate, tau

    def _cfc_step(
        self, h: torch.Tensor, xp: torch.Tensor, tx: torch.Tensor,
        sub_dt: float, fused_w: torch.Tensor, fused_b: torch.Tensor,
    ) -> torch.Tensor:
        """Замкнутая форма CfC: два кандидата ff1/ff2 плюс обучаемая
        сигмоидная интерполяция t_interp(x,h,dt) между ними — без ODE."""
        H = self.hidden_size
        fused = self._fused_from_h(h, fused_w, fused_b)
        ff1 = torch.tanh(xp[:, :H] + fused[:, :H])
        ff2 = torch.tanh(xp[:, H:] + fused[:, H:2 * H])
        t_a = fused[:, 2 * H:3 * H] + tx[:, :H]
        t_b = fused[:, 3 * H:] + tx[:, H:]
        t_interp = torch.sigmoid(t_a * sub_dt + t_b)
        return ff1 * (1.0 - t_interp) + t_interp * ff2

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        dt: float = 1.0,
        xp: torch.Tensor | None = None,
        tx: torch.Tensor | None = None,
        fused_w: torch.Tensor | None = None,
        fused_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sub_dt = dt * self._inv_unfolds
        h_in = h

        if xp is None or tx is None:
            xp, tx = self.precompute_x(x)
        if self.wiring != "ncp" and (fused_w is None or fused_b is None):
            fused_w, fused_b = self.fused_h_weights()

        for _ in range(self.ode_unfolds):
            if self.gate_mode == "cfc":
                h = self._cfc_step(h, xp, tx, sub_dt, fused_w, fused_b)
            elif self.ode_solver == "heun":
                h_pred, cand0, tau0 = self._ltc_step(h, xp, tx, sub_dt, fused_w, fused_b)
                _, cand1, tau1 = self._ltc_step(h_pred, xp, tx, sub_dt, fused_w, fused_b)
                candidate = 0.5 * (cand0 + cand1)
                tau = 0.5 * (tau0 + tau1)
                decay = torch.exp(-sub_dt / tau)
                h = decay * h + (1.0 - decay) * candidate
            else:
                h, _, _ = self._ltc_step(h, xp, tx, sub_dt, fused_w, fused_b)

        out = self.norm(self.dropout(h))

        if self.training and self.zoneout > 0.0:
            prev = self.norm(h_in)
            keep = torch.rand_like(out) < self.zoneout
            out = torch.where(keep, prev, out)

        return out


class ChunkMemory(nn.Module):
    """
    Скользящая память поверх --state-carry — отдельный, более медленный
    канал долгой памяти, не подверженный exp-decay динамике самой liquid-
    ячейки. Идея — "явная персистентная память поверх ODE-состояния для
    длинных последовательностей" (по мотивам M-CfC; формула из статьи НЕ
    воспроизводится), механически ближе всего к кэш-памяти Transformer-XL.

    В конце каждого --state-carry подчанка финальное скрытое состояние
    последнего liquid-слоя кладётся в FIFO-буфер (--memory-slots последних
    сводок; переполнение вытесняет самую старую). В начале СЛЕДУЮЩЕГО
    подчанка эта память читается через dot-product attention и подмешивается
    (через обучаемый gate) в переносимое между чанками скрытое состояние
    последнего слоя.

    Между подчанками hidden и буфер детачатся (ради памяти GPU) — прямого
    градиента через границу чанка нет. Веса read/write учатся по эффекту
    внутри каждого отдельного подчанка.

    Буфер хранит СЫРЫЕ (детачнутые) финальные состояния, а summarize
    применяется лениво в read() — поэтому summarize реально получает
    градиент. Память работает только при --state-carry >= 2.

    Подключено к обучению и evaluate(); generate()/чат эту память НЕ
    используют (осознанное ограничение).
    """

    def __init__(self, hidden_size: int, memory_slots: int) -> None:
        super().__init__()
        if memory_slots < 1:
            raise ValueError("--memory-slots must be >= 1")
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.summarize = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.init_slot = nn.Parameter(torch.zeros(hidden_size))
        self._scale = hidden_size ** 0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.summarize.weight)
        nn.init.zeros_(self.summarize.bias)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.read_gate.weight)
        nn.init.zeros_(self.read_gate.bias)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Свежий буфер в начале нового span — аналог initial_hidden().
        init_slot НЕ детачится: это обучаемый параметр, и градиент из
        read() в первом подчанке спана должен доходить до него."""
        return (
            self.init_slot
            .to(device=device)
            .view(1, 1, -1)
            .expand(batch_size, self.memory_slots, -1)
            .contiguous()
        )

    def read(self, h0: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mem = torch.tanh(self.summarize(memory))                       # (B,slots,H)
        q = self.query_proj(h0).unsqueeze(1)                           # (B,1,H)
        k = self.key_proj(mem)                                         # (B,slots,H)
        attn = torch.softmax((q * k).sum(dim=-1) / self._scale, dim=-1)  # (B,slots)
        retrieved = (attn.unsqueeze(-1) * mem).sum(dim=1)              # (B,H)
        gate = torch.sigmoid(self.read_gate(torch.cat([h0, retrieved], dim=-1)))
        return h0 + gate * retrieved

    def write(self, h_final: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        raw = h_final.detach().to(memory.dtype).unsqueeze(1)           # (B,1,H)
        return torch.cat([memory[:, 1:, :], raw], dim=1)               # FIFO-сдвиг


class LiquidLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 384,
        hidden_size: int = 384,
        num_layers: int = 2,
        ode_unfolds: int = 1,
        dropout: float = 0.05,
        zoneout: float = 0.1,
        tie_weights: bool = True,
        mlp_head: bool = False,
        residual: bool = True,
        multi_tau: bool = True,
        local_conv: bool = True,
        use_checkpoint: bool = False,
        gate_mode: str = "ltc",
        ode_solver: str = "euler",
        wiring: str = "dense",
        ncp_command_frac: float = 0.4,
        ncp_motor_frac: float = 0.2,
        memory_slots: int = 0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if tie_weights and embedding_dim != hidden_size:
            raise ValueError(
                "tie_weights=True requires embedding_dim == hidden_size "
                f"(сейчас {embedding_dim} != {hidden_size})."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ode_unfolds = ode_unfolds
        self.dropout_rate = dropout
        self.zoneout = zoneout
        self.tie_weights = tie_weights
        self.mlp_head = mlp_head
        self.residual = residual
        self.multi_tau = multi_tau
        self.local_conv = local_conv
        self.use_checkpoint = use_checkpoint
        self.gate_mode = gate_mode
        self.ode_solver = ode_solver
        self.wiring = wiring
        self.ncp_command_frac = ncp_command_frac
        self.ncp_motor_frac = ncp_motor_frac
        self.memory_slots = memory_slots
        self.chunk_memory = ChunkMemory(hidden_size, memory_slots) if memory_slots > 0 else None

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emb_dropout = LockedDropout(dropout)

        if local_conv:
            # ПРИЧИННАЯ (каузальная) свёртка: паддинг только слева
            # (kernel_size-1 нулей/буфер), справа — 0. Симметричный padding
            # у Conv1d видит БУДУЩИЕ токены на позиции t — прямая утечка
            # ответа в train/val loss (см. forward()).
            self._conv_kernel_size = 5
            self._conv_left_pad = self._conv_kernel_size - 1
            self.local_conv_layer = nn.Conv1d(
                embedding_dim, embedding_dim,
                kernel_size=self._conv_kernel_size,
                groups=embedding_dim, padding=0,
            )
        else:
            self.local_conv_layer = None

        cells = []
        for layer in range(num_layers):
            input_size = embedding_dim if layer == 0 else hidden_size
            cells.append(
                LiquidCell(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    ode_unfolds=ode_unfolds,
                    dropout=dropout,
                    zoneout=zoneout,
                    multi_tau=multi_tau,
                    gate_mode=gate_mode,
                    ode_solver=ode_solver,
                    wiring=wiring,
                    ncp_command_frac=ncp_command_frac,
                    ncp_motor_frac=ncp_motor_frac,
                )
            )
        self.cells = nn.ModuleList(cells)

        self.h0 = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(hidden_size)
        if mlp_head:
            self.readout = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, vocab_size),
            )
        else:
            self.readout = nn.Linear(hidden_size, vocab_size)

        if residual:
            self._do_residual = tuple(
                [embedding_dim == hidden_size] + [True] * (num_layers - 1)
            )
        else:
            self._do_residual = tuple([False] * num_layers)

        self._reset_parameters()
        self._step_fn = self._step_impl
        self._kernel = "torch"

    def set_kernel(self, kernel: str) -> str:
        """Выбор бэкенда хвоста ltc-подшага: "torch" (по умолчанию) или
        "triton" (только инференс, см. _ltc_tail_kernel). При невозможности
        включить triton — печатает причину и остаётся на torch, не падая.
        Возвращает фактически выбранный бэкенд."""
        if kernel not in ("torch", "triton"):
            raise ValueError(f"Неизвестный kernel: {kernel!r} (ожидался 'torch' или 'triton')")
        actual = kernel
        if kernel == "triton":
            reason = None
            if not _TRITON_AVAILABLE:
                reason = "the triton package is not installed"
            elif not torch.cuda.is_available():
                reason = "CUDA is unavailable"
            elif not next(self.parameters()).is_cuda:
                reason = "the model is not on CUDA"
            elif self.gate_mode != "ltc" or self.ode_solver != "euler":
                reason = "only --gate-mode ltc with --ode-solver euler is supported"
            if reason is not None:
                print(f"[KERNEL] triton недоступен ({reason}) — используется torch.")
                actual = "torch"
        for cell in self.cells:
            cell.kernel = actual
        self._kernel = actual
        return actual

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        if self.local_conv_layer is not None:
            nn.init.kaiming_normal_(
                self.local_conv_layer.weight, mode="fan_out", nonlinearity="relu"
            )
            if self.local_conv_layer.bias is not None:
                nn.init.zeros_(self.local_conv_layer.bias)

        out_linear = self.readout[-1] if self.mlp_head else self.readout
        if self.tie_weights:
            out_linear.weight = self.embedding.weight
        else:
            nn.init.xavier_uniform_(out_linear.weight)
        if out_linear.bias is not None:
            nn.init.zeros_(out_linear.bias)
        if self.mlp_head:
            nn.init.xavier_uniform_(self.readout[0].weight)
            nn.init.zeros_(self.readout[0].bias)

    def initial_hidden(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        return [
            self.h0[i].unsqueeze(0).expand(batch_size, -1).contiguous()
            for i in range(self.num_layers)
        ]

    def enable_compiled_step(
        self, mode: str = "default", batch_size: int = 2, inference: bool = False,
    ) -> tuple[bool, str | None]:
        """Компилирует рекуррентный шаг. inference=False — прогрев в train-режиме
        с backward (для обучения); inference=True — прогрев в eval-режиме под
        inference_mode (для генерации/чата: иначе первый же generate() вызвал бы
        перекомпиляцию из-за смены training-флага и режима градиентов)."""
        if not hasattr(torch, "compile"):
            return False, None
        device = next(self.parameters()).device
        b = max(1, int(batch_size))
        x_t = torch.zeros(b, self.embedding_dim, device=device)
        xp_dim = self.hidden_size * 2 if self.gate_mode == "cfc" else self.hidden_size
        xp_t = torch.zeros(b, xp_dim, device=device)
        tx_t = torch.zeros(b, xp_dim, device=device)
        was_training = self.training
        last_error: str | None = None
        try:
            if inference:
                self.eval()
            else:
                self.train()
            modes = [mode] if mode == "default" else [mode, "default"]
            for try_mode in modes:
                try:
                    compiled = torch.compile(
                        self._step_impl, dynamic=False, mode=try_mode
                    )
                    for _ in range(3):
                        hidden = self.initial_hidden(b, device)
                        if inference:
                            with torch.inference_mode():
                                fused = [cell.fused_h_weights() for cell in self.cells]
                                compiled(x_t, xp_t, tx_t, hidden, 1.0, fused)
                        else:
                            # fused пересчитывается на каждой итерации: граф
                            # torch.cat не должен переиспользоваться после backward.
                            fused = [cell.fused_h_weights() for cell in self.cells]
                            out, _ = compiled(x_t, xp_t, tx_t, hidden, 1.0, fused)
                            out.sum().backward()
                    if not inference:
                        self.zero_grad(set_to_none=True)
                    self._step_fn = compiled
                    return True, try_mode
                except Exception as exc:
                    self._step_fn = self._step_impl
                    last_error = f"[{try_mode}] {type(exc).__name__}: {exc}"
                    continue
            if last_error:
                print(f"[COMPILE] Причина: {last_error}")
            return False, None
        finally:
            if was_training:
                self.train()
            else:
                self.eval()

    def _step_impl(self, x_t, xp_t, tx_t, hidden, dt, fused):
        hidden = list(hidden)
        do_residual = self._do_residual
        use_ckpt = self.use_checkpoint and self.training

        for i, cell in enumerate(self.cells):
            w, b = fused[i]
            if use_ckpt:
                h_new = cp.checkpoint(
                    cell,
                    hidden[i], x_t,
                    dt=dt,
                    xp=xp_t if i == 0 else None,
                    tx=tx_t if i == 0 else None,
                    fused_w=w,
                    fused_b=b,
                    use_reentrant=False,
                )
            else:
                if i == 0:
                    h_new = cell(hidden[i], x_t, dt=dt, xp=xp_t, tx=tx_t, fused_w=w, fused_b=b)
                else:
                    h_new = cell(hidden[i], x_t, dt=dt, fused_w=w, fused_b=b)

            hidden[i] = h_new
            if do_residual[i]:
                x_t = x_t + h_new
            else:
                x_t = h_new

        return self.final_norm(x_t), hidden

    def forward(
        self,
        tokens: torch.Tensor,
        hidden: list[torch.Tensor] | None = None,
        dt: float = 1.0,
        fused: list | None = None,
    ):
        batch_size, seq_len = tokens.shape
        conv_buf = None
        if hidden is None:
            hidden = self.initial_hidden(batch_size, tokens.device)
        else:
            hidden = list(hidden)
            if len(hidden) > self.num_layers:
                conv_buf = hidden[self.num_layers]
                hidden = hidden[: self.num_layers]
        new_conv_buf = None

        x = self.emb_dropout(self.embedding(tokens))

        if self.local_conv_layer is not None:
            if conv_buf is None:
                conv_buf = x.new_zeros(batch_size, self._conv_left_pad, x.size(-1))
            x_ext = torch.cat([conv_buf.to(x.dtype), x], dim=1)
            new_conv_buf = x_ext[:, -self._conv_left_pad:, :]
            x_conv = self.local_conv_layer(x_ext.transpose(1, 2)).transpose(1, 2)
            x = x + x_conv

        if fused is None:
            fused = [cell.fused_h_weights() for cell in self.cells]

        cell0 = self.cells[0]
        xp_seq, tx_seq = cell0.precompute_x(x)

        outputs = []
        # Компилированный шаг не содержит triton-ветку — поэтому при активном
        # triton-ядре инференс идёт через eager _step_impl.
        if self._kernel == "triton" and not self.training:
            step_fn = self._step_impl
        else:
            step_fn = self._step_fn
        for t in range(seq_len):
            out_t, hidden = step_fn(
                x[:, t, :], xp_seq[:, t, :], tx_seq[:, t, :], hidden, dt, fused
            )
            outputs.append(out_t)

        features = torch.stack(outputs, dim=1)
        logits = self.readout(features)
        if new_conv_buf is not None:
            hidden = list(hidden) + [new_conv_buf]
        return logits, hidden

    @staticmethod
    def _apply_repetition_penalty(logits, recent_tokens, penalty):
        if penalty <= 1.0 or not recent_tokens:
            return logits
        logits = logits.clone()
        uniq = torch.tensor(
            sorted(set(recent_tokens)), dtype=torch.long, device=logits.device
        )
        vals = logits.index_select(1, uniq)
        logits.index_copy_(
            1, uniq, torch.where(vals < 0, vals * penalty, vals / penalty)
        )
        return logits

    @staticmethod
    def _top_k_top_p_filter(logits, top_k=0, top_p=1.0):
        logits = logits.clone()
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, k=k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
            sorted_remove[:, 0] = False
            remove_mask = torch.zeros_like(logits, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove_mask, float("-inf"))
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 500,
        temperature: float = 0.85,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        repetition_window: int = 128,
        device: torch.device | str = "cpu",
        stop_token_ids: list[int] | None = None,
        only_new: bool = False,
    ) -> str:
        self.eval()
        if not prompt:
            prompt = " "
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = tokenizer.encode(" ")
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        stop_ids = set(stop_token_ids or ())

        fused = [cell.fused_h_weights() for cell in self.cells]
        logits_all, hidden = self.forward(tokens, fused=fused)
        next_logits = logits_all[:, -1, :].float()

        generated_ids = list(prompt_ids)
        recent_tokens = generated_ids[-repetition_window:]
        temperature = max(float(temperature), 1e-4)

        for step_i in range(max_new_tokens):
            logits = self._apply_repetition_penalty(
                next_logits, recent_tokens, repetition_penalty
            )
            logits = logits / temperature
            logits = self._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            if token_id in stop_ids:
                break
            generated_ids.append(token_id)
            recent_tokens.append(token_id)
            recent_tokens = recent_tokens[-repetition_window:]
            if step_i + 1 < max_new_tokens:
                logits_all, hidden = self.forward(next_token, hidden=hidden, fused=fused)
                next_logits = logits_all[:, -1, :].float()

        if only_new:
            return tokenizer.decode(generated_ids[len(prompt_ids):])
        return tokenizer.decode(generated_ids)


class CharTokenizer:
    kind = "char"

    def __init__(self, char_to_idx: dict[str, int], idx_to_char: list[str]):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_char)

    @classmethod
    def train(cls, corpus: str) -> "CharTokenizer":
        chars = sorted(set(corpus))
        return cls({ch: i for i, ch in enumerate(chars)}, chars)

    def encode(self, text: str) -> list[int]:
        fallback = self.char_to_idx.get(" ", 0)
        return [self.char_to_idx.get(ch, fallback) for ch in text]

    def encode_ids_np(self, text: str) -> np.ndarray:
        fallback = self.char_to_idx.get(" ", 0)
        return np.fromiter(
            (self.char_to_idx.get(ch, fallback) for ch in text),
            dtype=np.int64, count=len(text),
        )

    def decode(self, ids) -> str:
        return "".join(self.idx_to_char[i] for i in ids)

    def token_to_id(self, token: str):
        return None

    def to_config(self) -> dict:
        return {"kind": self.kind, "char_to_idx": self.char_to_idx, "idx_to_char": self.idx_to_char}

    @classmethod
    def from_config(cls, config: dict) -> "CharTokenizer":
        return cls(config["char_to_idx"], config["idx_to_char"])


_BPE_ENCODE_CHUNK = 2_000_000


class BPETokenizer:
    kind = "bpe"

    def __init__(self, tokenizer: "_HFTokenizer"):
        self._tok = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @classmethod
    def train(cls, corpus: str, vocab_size: int, special_tokens: list[str]) -> "BPETokenizer":
        if not _HAS_TOKENIZERS:
            raise RuntimeError("The 'tokenizers' library is not installed.")
        tok = _HFTokenizer(_BPEModel(unk_token="<unk>"))
        # add_prefix_space=False: иначе к началу каждого текста/чанка добавляется
        # лишний пробел (и decode отдаёт его обратно).
        tok.pre_tokenizer = _ByteLevelPreTokenizer(add_prefix_space=False)
        tok.decoder = _ByteLevelDecoder()
        trainer = _BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<pad>"] + list(special_tokens),
            # Все 256 байт в словаре: любой символ, не встретившийся в train,
            # всё равно кодируется без <unk>.
            initial_alphabet=_ByteLevelPreTokenizer.alphabet(),
            show_progress=False,
        )
        chunk_size = 10_000
        chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)] or [corpus]
        tok.train_from_iterator(chunks, trainer=trainer)
        return cls(tok)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_ids_np(self, text: str) -> np.ndarray:
        if len(text) <= _BPE_ENCODE_CHUNK:
            return np.asarray(self._tok.encode(text).ids, dtype=np.int64)
        parts = []
        i, n = 0, len(text)
        while i < n:
            j = min(i + _BPE_ENCODE_CHUNK, n)
            if j < n:
                cut = text.rfind("\n\n", i, j)
                if cut > i:
                    j = cut + 2
            parts.append(np.asarray(self._tok.encode(text[i:j]).ids, dtype=np.int64))
            i = j
        return np.concatenate(parts)

    def decode(self, ids) -> str:
        return self._tok.decode(list(int(i) for i in ids))

    def token_to_id(self, token: str):
        return self._tok.token_to_id(token)

    def to_config(self) -> dict:
        return {"kind": self.kind, "tokenizer_json": self._tok.to_str()}

    @classmethod
    def from_config(cls, config: dict) -> "BPETokenizer":
        if not _HAS_TOKENIZERS:
            raise RuntimeError("The 'tokenizers' library is not installed.")
        return cls(_HFTokenizer.from_str(config["tokenizer_json"]))


def build_tokenizer(corpus: str, kind: str, vocab_size: int):
    if kind == "char":
        return CharTokenizer.train(corpus)
    if kind == "bpe":
        est_tokens = max(1, len(corpus) // 4)
        cap = max(300, est_tokens // 20)
        if vocab_size > cap:
            print(f"[DATA] vocab-size уменьшен {vocab_size} -> {cap}.")
            vocab_size = cap
        return BPETokenizer.train(corpus, vocab_size=vocab_size, special_tokens=DIALOGUE_SPECIAL_TOKENS)
    raise ValueError(f"Неизвестный тип токенизатора: {kind}")


def tokenizer_from_config(config: dict):
    kind = config.get("kind", "char")
    if kind == "char":
        return CharTokenizer.from_config(config)
    if kind == "bpe":
        return BPETokenizer.from_config(config)
    raise ValueError(f"Неизвестный тип токенизатора в чекпоинте: {kind}")


def tokenizer_from_checkpoint(checkpoint: dict):
    if "tokenizer" in checkpoint:
        return tokenizer_from_config(checkpoint["tokenizer"])
    if "char_to_idx" in checkpoint:
        return CharTokenizer(checkpoint["char_to_idx"], checkpoint["idx_to_char"])
    raise KeyError("The checkpoint contains no tokenizer data.")


def encode_corpus(tokenizer, text: str) -> np.ndarray:
    method = getattr(tokenizer, "encode_ids_np", None)
    if method is not None:
        return method(text)
    return np.asarray(tokenizer.encode(text), dtype=np.int64)


def default_stop_ids(tokenizer) -> list[int]:
    ids = []
    tid = tokenizer.token_to_id(EXAMPLE_END_TAG)
    if tid is not None:
        ids.append(tid)
    return ids


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
    if max_start <= split_start:
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


def masked_cross_entropy(logits, targets):
    valid_count = (targets != IGNORE_INDEX).sum()
    sum_loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, reduction="sum")
    loss = sum_loss / valid_count.clamp(min=1)
    return loss, sum_loss, valid_count


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


def detach_hidden(hidden):
    return [h.detach() for h in hidden]


def chunk_memory_reset(base_model, batch_size: int, device: torch.device):
    """Свежий буфер ChunkMemory в начале нового span — вызывать там же,
    где обнуляется hidden. None, если --memory-slots выключен (0)."""
    if base_model.chunk_memory is None:
        return None
    return base_model.chunk_memory.init_memory(batch_size, device)


def chunk_memory_read(base_model, hidden, memory):
    """Подмешивает память в состояние последнего слоя ПЕРЕД обработкой
    очередного --state-carry подчанка. Если hidden ещё не материализован
    (самый первый подчанк span'а), сначала берёт обучаемое h0 — если
    chunk_memory=None, просто возвращает hidden как есть."""
    if base_model.chunk_memory is None:
        return hidden
    if hidden is None:
        hidden = base_model.initial_hidden(memory.shape[0], memory.device)
    hidden = list(hidden)
    last = base_model.num_layers - 1
    hidden[last] = base_model.chunk_memory.read(hidden[last], memory)
    return hidden


def chunk_memory_write(base_model, hidden, memory):
    """Обновляет буфер ПОСЛЕ обработки подчанка финальным hidden[-1]."""
    if base_model.chunk_memory is None:
        return memory
    return base_model.chunk_memory.write(hidden[base_model.num_layers - 1], memory)


def get_base_model(model):
    return getattr(model, "_orig_mod", model)


def clone_float_state(model):
    return {
        k: v.detach().clone()
        for k, v in get_base_model(model).state_dict().items()
        if v.dtype.is_floating_point
    }


def restore_float_state(model, backup):
    msd = get_base_model(model).state_dict()
    for k, v in backup.items():
        msd[k].copy_(v)


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


ARCH_KEYS = (
    "embedding_dim", "hidden_size", "num_layers", "ode_unfolds",
    "dropout", "zoneout", "tie_weights", "mlp_head", "residual",
    "multi_tau", "local_conv", "gate_mode", "ode_solver",
    "wiring", "ncp_command_frac", "ncp_motor_frac", "memory_slots",
)

LEGACY_ARCH_DEFAULTS = {
    "zoneout": 0.0, "tie_weights": False, "mlp_head": False,
    "residual": False, "multi_tau": False, "local_conv": False,
    "gate_mode": "ltc", "ode_solver": "euler",
    "wiring": "dense",
    "ncp_command_frac": 0.4, "ncp_motor_frac": 0.2,
    "memory_slots": 0,
}


def _arch_value(config, key):
    if key in config:
        return config[key]
    return LEGACY_ARCH_DEFAULTS.get(key)


def create_model_from_config(vocab_size, config, device):
    return LiquidLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=int(config["embedding_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        ode_unfolds=int(config["ode_unfolds"]),
        dropout=float(config.get("dropout", 0.05)),
        zoneout=float(_arch_value(config, "zoneout") or 0.0),
        tie_weights=bool(_arch_value(config, "tie_weights")),
        mlp_head=bool(_arch_value(config, "mlp_head")),
        residual=bool(_arch_value(config, "residual")),
        multi_tau=bool(_arch_value(config, "multi_tau")),
        local_conv=bool(_arch_value(config, "local_conv")),
        use_checkpoint=False,
        gate_mode=str(_arch_value(config, "gate_mode") or "ltc"),
        ode_solver=str(_arch_value(config, "ode_solver") or "euler"),
        wiring=str(_arch_value(config, "wiring") or "dense"),
        ncp_command_frac=float(_arch_value(config, "ncp_command_frac") or 0.4),
        ncp_motor_frac=float(_arch_value(config, "ncp_motor_frac") or 0.2),
        memory_slots=int(_arch_value(config, "memory_slots") or 0),
    ).to(device)


def load_model_state_compat(model, state):
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Неожиданные ключи в checkpoint: {unexpected}")
    allowed_missing = {f"h0.{i}" for i in range(model.num_layers)}
    if model.multi_tau:
        allowed_missing.add("tau_mix")
    if model.local_conv_layer is not None:
        allowed_missing.add("local_conv_layer.weight")
        allowed_missing.add("local_conv_layer.bias")
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"Checkpoint is missing weights: {bad_missing}")
    print("[CHECKPOINT] Compatible weights loaded (new features initialized).")


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


def _zeropower_via_newton_schulz5(G, steps):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            weight_decay, nesterov = group["weight_decay"], group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError("Muon supports only 2D parameters.")
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update_src = g.add(buf, alpha=momentum) if nesterov else buf
                u = _zeropower_via_newton_schulz5(update_src, steps=ns_steps)
                scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
                p.add_(u, alpha=-lr * scale)
        return loss


class MuonAdamWHybrid:
    """
    Ведёт себя как torch.optim.Optimizer (param_groups/step/zero_grad/
    state_dict/load_state_dict) через duck typing, но НЕ наследуется от
    torch.optim.Optimizer напрямую: у него нет собственного списка параметров
    (он собирается из двух РЕАЛЬНЫХ вложенных оптимизаторов), а
    Optimizer.__init__() вызвать нечем — без него LRScheduler упал бы на
    несуществующем _step_count. GradScaler и clip_grad_norm_ работают с этим
    объектом чисто по duck typing.

    LambdaLR в новых версиях PyTorch проверяет isinstance(optimizer, Optimizer),
    поэтому scheduler для этого объекта создаётся отдельно для Muon и AdamW
    (см. MultiScheduler/make_scheduler).
    """

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW):
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.muon.step()
        return self.adamw.step(closure)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


class MultiScheduler:
    """Несколько LambdaLR под одним интерфейсом (step/state_dict/...)."""

    def __init__(self, schedulers):
        self.schedulers = list(schedulers)

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def get_last_lr(self):
        return [lr for sch in self.schedulers for lr in sch.get_last_lr()]

    def state_dict(self):
        return {"schedulers": [sch.state_dict() for sch in self.schedulers]}

    def load_state_dict(self, state):
        states = state.get("schedulers") if isinstance(state, dict) else None
        if not states or len(states) != len(self.schedulers):
            raise ValueError("scheduler state format was not accepted")
        for sch, st in zip(self.schedulers, states):
            sch.load_state_dict(st)


def make_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    total_steps = max(total_steps, 1)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step + 1, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    if isinstance(optimizer, MuonAdamWHybrid):
        return MultiScheduler([
            torch.optim.lr_scheduler.LambdaLR(optimizer.muon, lr_lambda),
            torch.optim.lr_scheduler.LambdaLR(optimizer.adamw, lr_lambda),
        ])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_optimizer(model, args, device):
    if args.optimizer == "muon":
        base = get_base_model(model)
        out_linear = base.readout[-1] if base.mlp_head else base.readout
        excluded_ids = {id(base.embedding.weight), id(out_linear.weight)}
        if out_linear.bias is not None:
            excluded_ids.add(id(out_linear.bias))
        muon_params, adamw_params = [], []
        for p in base.parameters():
            if not p.requires_grad:
                continue
            if p.ndim != 2 or id(p) in excluded_ids:
                adamw_params.append(p)
            else:
                muon_params.append(p)
        print(
            f"[OPT] Muon: {sum(p.numel() for p in muon_params):,} параметров "
            f"(hidden-матрицы) | AdamW: {sum(p.numel() for p in adamw_params):,}"
        )
        muon = Muon(muon_params, lr=args.muon_lr, momentum=args.muon_momentum,
                    weight_decay=args.weight_decay, ns_steps=args.muon_ns_steps)
        adamw = torch.optim.AdamW(adamw_params, lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
        return MuonAdamWHybrid(muon, adamw)
    kwargs = dict(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


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


def selftest_triton_tail(device) -> None:
    """Сверяет triton-хвост ltc-подшага с обычным torch-путём на случайных
    данных (single-tau и multi-tau, включая неравномерные веса банков и
    "жёсткие" входы для softplus) и замеряет время. Запускается вручную:
        python liquid_text_model.py --selftest-triton
    Нужны CUDA и пакет triton; иначе честно сообщает, что проверять нечего."""
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


def main():
    global _TRUST_CHECKPOINTS
    parser = build_parser()
    args = parser.parse_args()
    _TRUST_CHECKPOINTS = bool(args.trust_checkpoint)

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
    # При --resume архитектуру диктует чекпоинт (см. run_training), поэтому
    # проверку по CLI-значениям делаем только для нового обучения.
    if not args.resume and args.tie_weights and args.embedding_dim != args.hidden_size:
        raise ValueError("--tie-weights requires --embedding-dim == --hidden-size")
    if args.ema_decay != 0 and not 0.8 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be 0 or in [0.8, 1)")
    if args.tokenizer == "bpe" and _HAS_TOKENIZERS and args.vocab_size < 300:
        raise ValueError("--vocab-size is too small for byte-level BPE (minimum 300)")

    run_training(args, device)


if __name__ == "__main__":
    main()
