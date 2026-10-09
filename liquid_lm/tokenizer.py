"""Tokenizers (char / byte-level BPE) and helper functions."""

from __future__ import annotations

import numpy as np

try:
    from tokenizers import Tokenizer as _HFTokenizer
    from tokenizers.models import BPE as _BPEModel
    from tokenizers.trainers import BpeTrainer as _BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as _ByteLevelPreTokenizer
    from tokenizers.decoders import ByteLevel as _ByteLevelDecoder
    _HAS_TOKENIZERS = True
except ImportError:
    _HAS_TOKENIZERS = False

from .data import DIALOGUE_SPECIAL_TOKENS, EXAMPLE_END_TAG, FIM_SPECIAL_TOKENS


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
        # add_prefix_space=False: otherwise an extra leading space gets added to
        # every text/chunk (and decode returns it back).
        tok.pre_tokenizer = _ByteLevelPreTokenizer(add_prefix_space=False)
        tok.decoder = _ByteLevelDecoder()
        trainer = _BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<pad>"] + list(special_tokens),
            # All 256 bytes are in the vocabulary: any character unseen in train
            # is still encoded without <unk>.
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
        return BPETokenizer.train(
            corpus, vocab_size=vocab_size,
            special_tokens=DIALOGUE_SPECIAL_TOKENS + FIM_SPECIAL_TOKENS,
        )
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
