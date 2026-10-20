"""Dataset regressions; fixtures are temporary and require no downloaded data."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from liquid_lm import data
from liquid_lm.tokenizer import (
    BPETokenizer, _HAS_TOKENIZERS, encode_corpus, encode_corpus_split,
)
from liquid_lm.training import get_batch, restore_resume_data_settings


class DatasetReaderTests(unittest.TestCase):
    def test_single_json_matches_list_and_jsonl_text_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for obj in ({"text": "actual training text", "id": 123}, "plain text"):
                with self.subTest(obj=obj):
                    expected = obj["text"] if isinstance(obj, dict) else obj
                    for filename, value in (("one.json", obj), ("list.json", [obj]), ("one.jsonl", obj)):
                        path = root / filename
                        path.write_text(json.dumps(value), encoding="utf-8-sig")
                        self.assertEqual(data.load_one_file(path)[0], expected)

    def test_bom_dialogue_keeps_roles_and_example_end(self):
        obj = {"instruction": "Question", "input": "Context", "output": "Answer"}
        expected = "<|user|>\nQuestion\nContext\n<|bot|>\nAnswer\n<|end|>"
        with tempfile.TemporaryDirectory() as folder:
            for suffix in (".json", ".jsonl"):
                path = Path(folder) / ("dialogue" + suffix)
                path.write_text(json.dumps(obj), encoding="utf-8-sig")
                self.assertEqual(data.load_one_file(path)[0], expected)

    def test_jsonl_rejects_invalid_utf8_instead_of_deleting_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.jsonl"
            path.write_bytes(b'{"text":"bad\xfftext"}\n')
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                data.jsonl_to_text(path)

    def test_malformed_jsonl_line_retains_documented_raw_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mixed.jsonl"
            path.write_text('not json\n{"text":"valid"}\n', encoding="utf-8")
            self.assertEqual(data.jsonl_to_text(path), "not json\n\nvalid")

    def test_valid_utf8_bypasses_detector_and_removes_bom(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "text.txt"
            expected = "Hello \u043c\u0438\u0440 \U0001f600"
            for encoding in ("utf-8", "utf-8-sig"):
                path.write_text(expected, encoding=encoding)
                with patch.object(data, "_HAS_CHARSET_NORMALIZER", True), \
                     patch.object(data, "_charset_from_path", create=True) as detector:
                    self.assertEqual(data.read_text_robust(path)[0], expected)
                    detector.assert_not_called()

    def test_legacy_encoding_fallback_without_optional_detector(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "text.txt"
            expected = "\u041f\u0440\u0438\u0432\u0435\u0442"
            path.write_bytes(expected.encode("cp1251"))
            with patch.object(data, "_HAS_CHARSET_NORMALIZER", False):
                self.assertEqual(data.read_text_robust(path)[0], expected)

    def test_discovery_prunes_excluded_trees_and_preserves_deduplication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            excluded = root / "node_modules" / "deep"
            excluded.mkdir(parents=True)
            (excluded / "bad.txt").write_text("excluded", encoding="utf-8")
            (root / "a.txt").write_text("first\r\n\r\n\r\ntext  ", encoding="utf-8")
            (root / "b.txt").write_text("first\n\ntext", encoding="utf-8-sig")
            (root / "c.txt").write_text("second", encoding="utf-8")
            (root / "large.txt").write_text("x" * 2000, encoding="utf-8")
            walked = []
            original_walk = data.os.walk

            def tracked_walk(*args, **kwargs):
                for item in original_walk(*args, **kwargs):
                    walked.append(Path(item[0]))
                    yield item

            with patch.object(data.os, "walk", side_effect=tracked_walk), contextlib.redirect_stdout(io.StringIO()):
                result = data.load_texts(root, min_chars=1, max_file_mb=0.001, file_headers=False)
            self.assertEqual(result, "first\n\ntext\n\nsecond")
            self.assertEqual(walked, [root])


class _UnevenTokenizer:
    """Compress A pairs, leaving the B suffix at one token per character."""
    def encode(self, text):
        return [1 if c == "A" else 2 for c in text.replace("AA", "A")]


class DatasetSplitTests(unittest.TestCase):
    def test_character_boundary_stays_aligned_with_fitting_prefix(self):
        tokenizer = _UnevenTokenizer()
        text = "A" * 80 + "B" * 20
        encoded, split = encode_corpus_split(tokenizer, text, 0.2, "characters_v1")
        self.assertEqual(split, 40)
        np.testing.assert_array_equal(encoded[:split], [1] * 40)
        np.testing.assert_array_equal(encoded[split:], [2] * 20)
        # The former token-percent boundary puts validation characters in train.
        legacy, old_split = encode_corpus_split(tokenizer, text, 0.2, "legacy_tokens")
        np.testing.assert_array_equal(legacy, encoded)
        self.assertEqual(old_split, 48)
        self.assertTrue(np.any(legacy[:old_split] == 2))
        tensor = torch.from_numpy(encoded)
        for start, end, expected in ((0, split, 1), (split, len(encoded), 2)):
            x, y = get_batch(tensor, 16, 8, start, end)
            self.assertTrue(torch.all(x == expected))
            self.assertTrue(torch.all(y == expected))

    def test_resume_retains_legacy_policy_and_restores_new_policy(self):
        args = SimpleNamespace(val_split=0.1)
        restore_resume_data_settings(args, {})
        self.assertEqual(args.data_split_policy, "legacy_tokens")
        restore_resume_data_settings(args, {"data_split_policy": "characters_v1"})
        self.assertEqual(args.data_split_policy, "characters_v1")

    def test_legacy_validation_can_overlap_tokenizer_fitting_text(self):
        text = "B" * 80 + "A" * 20
        tokenizer = _UnevenTokenizer()
        _, new_split = encode_corpus_split(tokenizer, text, 0.2, "characters_v1")
        _, legacy_split = encode_corpus_split(tokenizer, text, 0.2, "legacy_tokens")
        self.assertEqual(new_split, 80)
        self.assertEqual(legacy_split, 72)
        # B positions 72..79 were fitted by BPE but evaluated as validation.
        self.assertLess(legacy_split, new_split)

    def test_unknown_policy_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Unknown data split policy"):
            encode_corpus_split(_UnevenTokenizer(), "AAAA", 0.2, "typo")

    @unittest.skipUnless(_HAS_TOKENIZERS, "optional tokenizers not installed")
    def test_real_bpe_validation_only_unicode_roundtrip_and_no_crossing(self):
        train = "ordinary training words\n" * 100
        validation = "\u4e16\u754c \U0001f600\n" * 25
        text = train + validation
        fraction = len(validation) / len(text)
        boundary = int(len(text) * (1 - fraction))
        tokenizer = BPETokenizer.train(text[:boundary], 300, [])
        encoded, split = encode_corpus_split(tokenizer, text, fraction, "characters_v1")
        np.testing.assert_array_equal(encoded[:split], encode_corpus(tokenizer, text[:boundary]))
        self.assertEqual(tokenizer.decode(encoded[:split]), text[:boundary])
        self.assertEqual(tokenizer.decode(encoded[split:]), text[boundary:])
        self.assertNotIn(tokenizer.token_to_id("<unk>"), encoded.tolist())


if __name__ == "__main__":
    unittest.main()
