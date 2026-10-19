"""Guard against invalid or semantically incompatible checkpoint averaging."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from research_liquid import average_checkpoints


class CheckpointAveragingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = [Path(self.directory.name) / f"model{i}.pt" for i in range(2)]
        self.source = {
            "config": {"hidden_size": 2},
            "tokenizer": {"kind": "char", "idx_to_char": ["a", "b"]},
            "corpus_sha256": "corpus",
            "model_state": {"weight": torch.tensor([2e38, 0.]), "count": torch.tensor(3)},
        }
        self.payloads = [copy.deepcopy(self.source) for _ in self.paths]
        for i, payload in enumerate(self.payloads):
            payload["experiment"] = {"source_sha256": "source"}
            payload["model_state"]["weight"][1] = 2. + 2. * i

    def average(self):
        for path, payload in zip(self.paths, self.payloads):
            torch.save(payload, path)
        return average_checkpoints(self.paths, self.source, "source")

    def test_average_preserves_dtype_buffers_and_avoids_intermediate_overflow(self):
        result = self.average()
        torch.testing.assert_close(result["weight"], torch.tensor([2e38, 3.]))
        self.assertEqual(result["weight"].dtype, torch.float32)
        self.assertEqual(result["count"].item(), 3)
        self.assertEqual(self.source["model_state"]["weight"][1].item(), 0.)

    def test_rejects_same_sized_tokenizers_with_different_token_meanings(self):
        self.payloads[1]["tokenizer"]["idx_to_char"].reverse()
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            self.average()

    def test_rejects_unrelated_sources_and_nonfinite_weights(self):
        original = copy.deepcopy(self.payloads[1])
        for mutation in ("source", "corpus", "config", "nan", "inf"):
            with self.subTest(mutation=mutation):
                payload = self.payloads[1] = copy.deepcopy(original)
                if mutation == "source":
                    payload["experiment"]["source_sha256"] = "other"
                elif mutation == "corpus":
                    payload["corpus_sha256"] = "other"
                elif mutation == "config":
                    payload["config"]["hidden_size"] = 3
                else:
                    payload["model_state"]["weight"][0] = float(mutation)
                with self.assertRaises(ValueError):
                    self.average()

    def test_rejects_incompatible_shapes_dtypes_and_discrete_buffers(self):
        original = copy.deepcopy(self.payloads[1])
        for mutation in ("shape", "dtype", "buffer"):
            with self.subTest(mutation=mutation):
                state = copy.deepcopy(original["model_state"])
                self.payloads[1]["model_state"] = state
                if mutation == "shape":
                    state["weight"] = torch.zeros(3)
                elif mutation == "dtype":
                    state["weight"] = state["weight"].double()
                else:
                    state["count"].add_(1)
                with self.assertRaises(ValueError):
                    self.average()

    def test_refuses_duplicate_inputs(self):
        self.paths[1] = self.paths[0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            average_checkpoints(self.paths, self.source, "source")


if __name__ == "__main__":
    unittest.main()
