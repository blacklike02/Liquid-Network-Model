"""Corpus: file constants, dialogue tags, file readers and corpus assembly."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dialogue tags
# ---------------------------------------------------------------------------


USER_TAG = "<|user|>"


BOT_TAG = "<|bot|>"


SYSTEM_TAG = "<|system|>"


EXAMPLE_END_TAG = "<|end|>"


DIALOGUE_SPECIAL_TOKENS = [USER_TAG, BOT_TAG, SYSTEM_TAG, EXAMPLE_END_TAG]


# ---------------------------------------------------------------------------
# Fill-in-the-middle (FIM)
# ---------------------------------------------------------------------------


FIM_PREFIX_TAG = "<|fim_prefix|>"


FIM_SUFFIX_TAG = "<|fim_suffix|>"


FIM_MIDDLE_TAG = "<|fim_middle|>"


FIM_SPECIAL_TOKENS = [FIM_PREFIX_TAG, FIM_SUFFIX_TAG, FIM_MIDDLE_TAG]


# Code files are read only when FIM is enabled (--fim-rate > 0); FIM is applied
# only to them. Plain text/dialogue files are never rearranged.
CODE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".sh", ".sql", ".html", ".css",
}


FIM_CHUNK_CHARS = 3000
FIM_MIN_CHUNK_CHARS = 16


def _stable_rng(*parts) -> random.Random:
    """Deterministic RNG derived from the parts (independent of PYTHONHASHSEED
    and of the global seed), so the same corpus always yields the same FIM
    rearrangement and the same corpus_sha256."""
    digest = hashlib.sha256("\x00".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _split_into_line_chunks(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        add = len(line) + 1
        if current and current_len + add > max_chars:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += add
    if current:
        chunks.append("\n".join(current))
    return chunks


def apply_fim(
    text: str, rate: float, key: str, chunk_chars: int = FIM_CHUNK_CHARS,
) -> tuple[str, int, int]:
    """Splits `text` into ~chunk_chars chunks on line boundaries and, with
    probability `rate` per chunk, rewrites a chunk in PSM order:

        <|fim_prefix|>PREFIX<|fim_suffix|>SUFFIX<|fim_middle|>MIDDLE<|end|>

    where PREFIX/MIDDLE/SUFFIX come from two random character cut points of the
    chunk. <|end|> after the middle lets the model learn where an infill stops
    (it is also a default stop token for generation).
    Returns (new_text, number_of_fim_chunks, total_chunks)."""
    if rate <= 0.0 or not text:
        return text, 0, 0
    out: list[str] = []
    n_fim = 0
    chunks = _split_into_line_chunks(text, chunk_chars)
    for i, chunk in enumerate(chunks):
        rng = _stable_rng("fim", key, i)
        if len(chunk) >= FIM_MIN_CHUNK_CHARS and rng.random() < rate:
            a, b = sorted(rng.sample(range(len(chunk) + 1), 2))
            prefix, middle, suffix = chunk[:a], chunk[a:b], chunk[b:]
            out.append(
                f"{FIM_PREFIX_TAG}{prefix}{FIM_SUFFIX_TAG}{suffix}"
                f"{FIM_MIDDLE_TAG}{middle}{EXAMPLE_END_TAG}"
            )
            n_fim += 1
        else:
            out.append(chunk)
    return "\n".join(out), n_fim, len(chunks)


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


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def load_texts(
    data_dir: Path, min_chars: int = 200, max_file_mb: float = 0.0,
    shuffle_files: bool = False, file_headers: bool = True,
    fim_rate: float = 0.0,
) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    scan_suffixes = SUPPORTED_SUFFIXES | CODE_SUFFIXES if fim_rate > 0.0 else SUPPORTED_SUFFIXES
    all_candidates = (
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in scan_suffixes
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
    fim_chunks = 0
    fim_total_chunks = 0
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
        if fim_rate > 0.0 and suffix in CODE_SUFFIXES:
            text, n_fim, n_chunks = apply_fim(
                text, fim_rate, key=path.relative_to(data_dir).as_posix(),
            )
            fim_chunks += n_fim
            fim_total_chunks += n_chunks
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
    if fim_rate > 0.0:
        print(
            f"[DATA] FIM (rate={fim_rate}): {fim_chunks:,} из {fim_total_chunks:,} "
            f"фрагментов кода переписаны в PSM-формат."
        )
        if fim_total_chunks == 0:
            print("[DATA] ВНИМАНИЕ: --fim-rate задан, но файлов с кодом не найдено.")
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
