"""Numerical, checkpoint and inference regressions found in the October audit."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from liquid_lm.cli import build_parser, main
from liquid_lm.layers import ChunkMemory
from liquid_lm.model import LiquidLanguageModel, load_checkpoint_raw, save_checkpoint
from liquid_lm.tokenizer import CharTokenizer
from liquid_lm.training import kd_loss_full, kd_loss_topk, run_training


class DistillationNumericsTests(unittest.TestCase):
    def test_saturated_student_matches_double_precision_partition_and_gradient(self):
        for student_values, teacher_values in (
            ([30., 0., 0.], [1., 0., 0.]),
            ([0., 30., 0.], [1., 0., 0.]),
            ([0., 0., 0.], [30., 0., 0.]),
        ):
            with self.subTest(student=student_values, teacher=teacher_values):
                student = torch.tensor([student_values], requires_grad=True)
                teacher = torch.tensor([teacher_values])
                reference_student = student.detach().double().requires_grad_()
                ls = reference_student.log_softmax(-1)
                lt = teacher.double().log_softmax(-1)
                # The teacher's top token is index 0 in each case.
                log_tail_t = torch.logsumexp(lt[:, 1:], -1)
                log_tail_s = torch.logsumexp(ls[:, 1:], -1)
                expected = (
                    lt[:, 0].exp() * (lt[:, 0] - ls[:, 0])
                    + log_tail_t.exp() * (log_tail_t - log_tail_s)
                )
                actual = kd_loss_topk(student, teacher, 1., 1)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(
                    student.grad.double(), reference_student.grad, atol=2e-6, rtol=2e-6,
                )

    def test_full_support_matches_full_kl_and_gradients(self):
        torch.manual_seed(12)
        teacher = torch.randn(4, 11) * 8
        student = (torch.randn(4, 11) * 8).requires_grad_()
        for k in (11, 20):
            actual = kd_loss_topk(student, teacher, 2., k)
            expected = kd_loss_full(student, teacher, 2.)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(
                torch.autograd.grad(actual.sum(), student)[0],
                torch.autograd.grad(expected.sum(), student)[0],
            )


class ChunkMemoryDtypeTests(unittest.TestCase):
    def test_carried_amp_state_can_be_read_outside_autocast(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                memory = ChunkMemory(8, 2)
                state = torch.randn(2, 8, dtype=dtype, requires_grad=True)
                slots = memory.init_memory(2, torch.device("cpu"))
                output = memory.read(state, slots)
                reference = memory.read(state.float(), slots)
                torch.testing.assert_close(output, reference)
                self.assertEqual(output.dtype, torch.float32)
                output.square().mean().backward()
                self.assertIsNotNone(state.grad)
                self.assertTrue(torch.isfinite(state.grad).all().item())


class CheckpointWriteTests(unittest.TestCase):
    @staticmethod
    def save(path):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        save_checkpoint(
            path, model, optimizer, scheduler, None, 1, 2, 3., 4., 4.,
            CharTokenizer.train("ab "), SimpleNamespace(embedding_dim=2),
        )

    def test_nested_destination_roundtrips_with_safe_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new" / "nested" / "model.pt"
            self.save(path)
            payload = load_checkpoint_raw(path, torch.device("cpu"))
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["global_step"], 2)
            self.assertEqual(set(payload["model_state"]), {"weight", "bias"})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_write_preserves_previous_checkpoint_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            self.save(path)
            previous = path.read_bytes()

            def interrupted_save(payload, destination):
                if hasattr(destination, "write"):
                    destination.write(b"interrupted")
                else:
                    Path(destination).write_bytes(b"interrupted")
                raise OSError("simulated full disk")

            with patch("liquid_lm.model.torch.save", side_effect=interrupted_save):
                with self.assertRaisesRegex(OSError, "full disk"):
                    self.save(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(path.parent.iterdir()), [path])


class GenerationPrefillTests(unittest.TestCase):
    def test_last_token_projection_matches_full_output_and_hidden(self):
        torch.manual_seed(23)
        for gate, wiring, norm, mlp in (
            ("ltc", "dense", "rms", False),
            ("ltc", "dense", "rms", True),
            ("cfc", "dense", "rms", True),
            ("cfc", "dense", "rms", False),
        ):
            with self.subTest(gate=gate, wiring=wiring, norm=norm, mlp=mlp):
                model = LiquidLanguageModel(
                    17, embedding_dim=8, hidden_size=8, num_layers=2,
                    dropout=0., zoneout=0., gate_mode=gate, wiring=wiring,
                    norm_type=norm, mlp_head=mlp, ffn_mult=1.,
                ).eval()
                tokens = torch.randint(0, 17, (2, 9))
                with torch.inference_mode():
                    full, full_hidden = model(tokens)
                    last, last_hidden = model(tokens, last_token_only=True)
                self.assertEqual(last.shape, (2, 1, 17))
                torch.testing.assert_close(last, full[:, -1:, :])
                for a, b in zip(last_hidden, full_hidden):
                    torch.testing.assert_close(a, b)

    def test_generation_projects_only_last_position_with_and_without_memory(self):
        tokenizer = CharTokenizer.train(" abcd")
        for memory_slots in (0, 2):
            with self.subTest(memory_slots=memory_slots):
                model = LiquidLanguageModel(
                    tokenizer.vocab_size, embedding_dim=8, hidden_size=8,
                    num_layers=1, dropout=0., zoneout=0., memory_slots=memory_slots,
                )
                with patch.object(model.readout, "forward", wraps=model.readout.forward) as head:
                    result = model.generate(
                        "abcdabcd", tokenizer, max_new_tokens=3, top_k=1,
                        memory_chunk_size=3, only_new=True,
                    )
                self.assertEqual(len(result), 3)
                self.assertTrue(head.call_args_list)
                self.assertTrue(all(call.args[0].size(1) == 1 for call in head.call_args_list))


class TrainingFailureTests(unittest.TestCase):
    def test_invalid_preview_settings_fail_before_training(self):
        with patch("sys.argv", ["liquid_text_model.py", "--device", "cpu", "--temperature", "nan"]), \
             patch("liquid_lm.cli.run_training") as train:
            with self.assertRaisesRegex(ValueError, "temperature"):
                main()
            train.assert_not_called()

    def test_nonfinite_gradient_never_reaches_optimizer(self):
        for grad_clip in (0., 1.):
            with self.subTest(grad_clip=grad_clip), tempfile.TemporaryDirectory() as folder:
                args = build_parser().parse_args([
                    "--device", "cpu", "--tokenizer", "char", "--epochs", "1",
                    "--steps-per-epoch", "1", "--eval-batches", "1", "--batch-size", "1",
                    "--seq-len", "8", "--state-carry", "1", "--grad-accumulation", "1",
                    "--embedding-dim", "8", "--hidden-size", "8", "--num-layers", "1",
                    "--dropout", "0", "--zoneout", "0", "--ema-decay", "0",
                    "--grad-clip", str(grad_clip), "--generate", "0",
                    "--data", folder, "--checkpoint", str(Path(folder) / "best.pt"),
                    "--last-checkpoint", str(Path(folder) / "last.pt"),
                ])

                def make_bad_model(**kwargs):
                    model = LiquidLanguageModel(**kwargs)
                    model.embedding.weight.register_hook(lambda grad: grad * float("nan"))
                    return model

                with patch("liquid_lm.training.load_texts", return_value=" abcd" * 400), \
                     patch("liquid_lm.training.LiquidLanguageModel", side_effect=make_bad_model), \
                     patch("torch.optim.AdamW.step", side_effect=AssertionError("optimizer received nonfinite gradients")) as update:
                    with self.assertRaisesRegex(RuntimeError, "non-finite"):
                        run_training(args, torch.device("cpu"))
                    update.assert_not_called()
                self.assertFalse((Path(folder) / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
