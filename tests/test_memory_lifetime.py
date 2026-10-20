"""Storage ownership and object lifetime regressions for memory audit fixes."""
import contextlib
import copy
import gc
import io
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

from liquid_lm import training
from liquid_lm.cli import build_parser
from liquid_lm.inference import run_generate_only
from liquid_lm.model import LiquidLanguageModel, load_inference_checkpoint
from liquid_lm.tokenizer import CharTokenizer, encode_corpus_split


def small_model(**kwargs):
    return LiquidLanguageModel(5, 8, 8, 1, dropout=0, zoneout=0, **kwargs)


def training_args(folder):
    return build_parser().parse_args([
        "--data", folder, "--tokenizer", "char", "--embedding-dim", "8",
        "--hidden-size", "8", "--num-layers", "1", "--batch-size", "2",
        "--seq-len", "8", "--epochs", "1", "--steps-per-epoch", "1",
        "--eval-batches", "1", "--amp", "off", "--generate", "0",
        "--checkpoint", str(Path(folder) / "best.pt"),
        "--last-checkpoint", str(Path(folder) / "last.pt"),
    ])


class MemoryLifetimeTests(unittest.TestCase):
    def test_eager_model_does_not_wait_for_cyclic_collection(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            model = small_model()
            reference = weakref.ref(model)
            parameter = weakref.ref(model.embedding.weight)
            del model
            self.assertIsNone(reference())
            self.assertIsNone(parameter())
        finally:
            if enabled:
                gc.enable()

    def test_carried_conv_tail_owns_only_its_logical_storage(self):
        model = small_model().eval()
        with torch.inference_mode():
            _, hidden = model(torch.zeros(2, 64, dtype=torch.long), last_token_only=True)
        tail = hidden[-1]
        self.assertEqual(tail.untyped_storage().nbytes(), tail.numel() * tail.element_size())
        self.assertFalse(tail.requires_grad)

    def test_conv_tail_clone_preserves_cross_chunk_outputs_and_gradients(self):
        model = small_model().double().eval()
        reference = copy.deepcopy(model)
        tokens = torch.randint(0, 5, (2, 15))
        expected = reference(tokens)[0]
        first, state = model(tokens[:, :7])
        second, _ = model(tokens[:, 7:], state)
        actual = torch.cat((first, second), dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                       rtol=1e-9, atol=1e-9)

    def test_batch_offset_cache_is_bounded_and_evicts_old_storage(self):
        training._batch_offsets.cache_clear()
        try:
            tensor = torch.arange(128, dtype=torch.int32)
            training.get_batch(tensor, 2, 2, 0, len(tensor))
            reference = weakref.ref(training._batch_offsets(tensor.device, 2))
            for length in range(3, 24):
                training.get_batch(tensor, 2, length, 0, len(tensor))
            self.assertLessEqual(training._batch_offsets.cache_info().currsize, 8)
            self.assertIsNone(reference())
        finally:
            training._batch_offsets.cache_clear()

    def test_inference_checkpoint_discards_training_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            torch.save({"model_state": small_model().state_dict(), "config": {"seq_len": 8},
                        "optimizer_state": {"moment": torch.ones(1024)},
                        "ema_state": {"shadow": torch.ones(1024)},
                        "scheduler_state": {}, "scaler_state": None, "epoch": 2}, path)
            payload = load_inference_checkpoint(path)
            self.assertEqual(set(payload), {"model_state", "config", "epoch"})
            self.assertTrue(all(t.device.type == "cpu" for t in payload["model_state"].values()))

    def test_inference_releases_checkpoint_weights_before_generation(self):
        references = []

        def checkpoint(*args, **kwargs):
            state = {k: v.clone() for k, v in small_model().state_dict().items()}
            references.extend(weakref.ref(value) for value in state.values())
            return {"model_state": state, "config": dict(vars(training_args("unused")), norm_type="rms",
                                                               memory_slots=0, memory_dim=0),
                    "tokenizer": CharTokenizer.train(" abcd").to_config(),
                    "optimizer_state": {"moment": torch.ones(1024)}}

        def generate(model, **kwargs):
            self.assertTrue(references)
            self.assertTrue(all(reference() is None for reference in references))
            return ""

        args = build_parser().parse_args(["--generate-only", "--generate", "0", "--kernel", "torch"])
        with patch("liquid_lm.model.load_checkpoint_raw", side_effect=checkpoint) as loader, \
             patch.object(LiquidLanguageModel, "generate", generate), contextlib.redirect_stdout(io.StringIO()):
            run_generate_only(args, torch.device("cpu"))
        self.assertEqual(loader.call_args.args[1], torch.device("cpu"))

    def test_dataset_preparation_releases_text_and_int64_ids_before_optimizer(self):
        class TrackedText(str):
            pass

        references = []

        def corpus(*args, **kwargs):
            text = TrackedText(" abcd" * 400)
            references.append(weakref.ref(text))
            return text

        def encode(*args, **kwargs):
            encoded, split = encode_corpus_split(*args, **kwargs)
            references.append(weakref.ref(encoded))
            return encoded, split

        def optimizer(*args, **kwargs):
            self.assertEqual(len(references), 2)
            self.assertTrue(all(reference() is None for reference in references))
            raise RuntimeError("lifetime verified")

        with tempfile.TemporaryDirectory() as folder, \
             patch("liquid_lm.training.load_texts", new=corpus), \
             patch("liquid_lm.training.encode_corpus_split", new=encode), \
             patch("liquid_lm.training.make_optimizer", side_effect=optimizer), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "lifetime verified"):
                training.run_training(training_args(folder), torch.device("cpu"))

    def test_failed_training_closes_logger_once(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch("liquid_lm.training.load_texts", return_value=" abcd" * 400), \
             patch("liquid_lm.training.TrainLogger") as logger, \
             patch.object(LiquidLanguageModel, "forward", side_effect=RuntimeError("forward failed")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                training.run_training(training_args(folder), torch.device("cpu"))
            logger.return_value.close.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MemoryLifetimeGpuTests(unittest.TestCase):
    def test_cpu_checkpoint_ema_moves_to_cuda_and_continues_update(self):
        source = small_model()
        saved = training.EMA(source).state_dict()
        model = copy.deepcopy(source).cuda()
        restored = training.EMA.from_state(saved, decay=0.99, device=torch.device("cuda"))
        restored.update(model)
        restored.copy_to(model)
        self.assertEqual(restored.updates, 1)
        self.assertTrue(all(value.is_cuda for value in restored.shadow.values()))
        for name, value in restored.shadow.items():
            torch.testing.assert_close(value.cpu(), saved["shadow"][name])


if __name__ == "__main__":
    unittest.main()
