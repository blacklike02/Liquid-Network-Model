"""Constants shared across the package."""

from __future__ import annotations


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
