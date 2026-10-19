"""Regression tests for Liquid-Network-Model's critical sequence paths."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from liquid_lm.layers import LiquidCell, RMSNorm, chunk_memory_read, chunk_memory_write
from liquid_lm.cli import validate_eval_batches, validate_training_ranges
from liquid_lm.model import LiquidLanguageModel, _trim_recent_tokens, validate_generation_options
from liquid_lm.tokenizer import CharTokenizer
from liquid_lm.training import (
    evaluate,
    get_batch,
    grad_scaler_step_succeeded,
    masked_cross_entropy,
    normalize_gradients_by_valid_tokens,
    restore_resume_data_settings,
    validate_distillation_tokenizers,
    validate_training_targets,
)


class BatchRegressionTests(unittest.TestCase):
    def test_get_batch_accepts_exactly_one_complete_window(self):
        # Four tokens can produce one (x, y) pair of length three.
        data = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        x, y = get_batch(
            data,
            batch_size=1,
            seq_len=3,
            split_start=0,
            split_end=4,
            gen=torch.Generator().manual_seed(0),
        )
        self.assertEqual(x.tolist(), [[10, 11, 12]])
        self.assertEqual(y.tolist(), [[11, 12, 13]])

    def test_get_batch_rejects_a_split_shorter_than_one_window(self):
        data = torch.tensor([10, 11, 12], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "Not enough text"):
            get_batch(data, batch_size=1, seq_len=3, split_start=0, split_end=3)

    def test_masked_cross_entropy_is_finite_when_every_target_is_ignored(self):
        logits = torch.randn(2, 4, 7, requires_grad=True)
        targets = torch.full((2, 4), -100, dtype=torch.long)
        loss, summed_loss, valid_count = masked_cross_entropy(
            logits.reshape(-1, 7), targets.reshape(-1)
        )
        self.assertEqual(valid_count.item(), 0)
        self.assertEqual(summed_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())


class DistillationTokenizerRegressionTests(unittest.TestCase):
    def test_rejects_same_vocab_size_with_different_token_id_mapping(self):
        student = CharTokenizer({"a": 0, "b": 1}, ["a", "b"])
        teacher = CharTokenizer({"a": 1, "b": 0}, ["b", "a"])

        with self.assertRaisesRegex(ValueError, "identical tokenizer configurations"):
            validate_distillation_tokenizers(student, teacher)

    def test_accepts_identical_tokenizer_configurations(self):
        student = CharTokenizer({"a": 0, "b": 1}, ["a", "b"])
        teacher = CharTokenizer({"a": 0, "b": 1}, ["a", "b"])

        validate_distillation_tokenizers(student, teacher)


class _MaskedEvalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 7
        self.chunk_memory = None

    def forward(self, tokens, hidden=None):
        logits = torch.zeros(
            tokens.size(0), tokens.size(1), self.vocab_size,
            dtype=torch.float32, device=tokens.device,
        )
        if hidden is None:
            hidden = [torch.zeros(tokens.size(0), 4, device=tokens.device)]
        return logits, hidden


class EvaluationRegressionTests(unittest.TestCase):
    def test_evaluate_rejects_validation_batches_without_unmasked_targets(self):
        model = _MaskedEvalModel()
        data = torch.arange(64, dtype=torch.long)
        no_targets = torch.zeros(64, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "zero unmasked target tokens"):
            evaluate(
                model, data, batch_size=1, seq_len=8, state_carry=1,
                device=torch.device("cpu"), split_start=0, split_end=64,
                batches=1, amp_dtype=None, train_mask=no_targets,
            )


class ResumeConfigurationRegressionTests(unittest.TestCase):
    def test_resume_restores_validation_split_from_checkpoint(self):
        args = SimpleNamespace(
            shuffle_files=True, file_headers=False, fim_rate=0.0, val_split=0.1,
        )
        restore_resume_data_settings(
            args,
            {"shuffle_files": False, "file_headers": True, "fim_rate": 0.5, "val_split": 0.2},
        )
        self.assertFalse(args.shuffle_files)
        self.assertTrue(args.file_headers)
        self.assertEqual(args.fim_rate, 0.5)
        self.assertEqual(args.val_split, 0.2)


class TrainingMaskRegressionTests(unittest.TestCase):
    def test_rejects_mask_without_training_targets(self):
        with self.assertRaisesRegex(ValueError, "excluded every target token"):
            validate_training_targets([False] * 100, split=80)

    def test_accepts_mask_with_training_targets(self):
        mask = [False] * 100
        mask[10] = True
        validate_training_targets(mask, split=80)


class RMSNormRegressionTests(unittest.TestCase):
    def test_preserves_bfloat16_output_dtype_and_finite_gradients(self):
        norm = RMSNorm(8)
        x = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
        y = norm(x)
        self.assertEqual(y.dtype, x.dtype)
        y.float().square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad.float()).all().item())
        self.assertTrue(torch.isfinite(norm.weight.grad).all().item())


class ZoneoutRegressionTests(unittest.TestCase):
    def test_full_zoneout_preserves_the_previous_hidden_state(self):
        cell = LiquidCell(
            input_size=8, hidden_size=8, dropout=0.0,
            zoneout=1.0, multi_tau=False,
        ).train()
        previous = torch.randn(2, 8)
        current_input = torch.randn(2, 8)

        actual = cell(previous, current_input)

        torch.testing.assert_close(actual, previous)


class GenerationOptionsRegressionTests(unittest.TestCase):
    def test_zero_repetition_window_disables_repetition_tracking(self):
        self.assertEqual(_trim_recent_tokens([1, 2, 3], 0), [])
        self.assertEqual(_trim_recent_tokens([1, 2, 3], 2), [2, 3])

    def test_negative_repetition_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "repetition_window must be >= 0"):
            _trim_recent_tokens([1, 2, 3], -1)

    def test_invalid_generation_options_are_rejected(self):
        cases = [
            (-1, 0.8, 0, 0.95, 1.05, 128),
            (10, 0.0, 0, 0.95, 1.05, 128),
            (10, 0.8, -1, 0.95, 1.05, 128),
            (10, 0.8, 0, 1.5, 1.05, 128),
            (10, 0.8, 0, 0.95, 0.0, 128),
            (10, 0.8, 0, 0.95, 1.05, -1),
        ]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    validate_generation_options(*options)


class GenerationMemoryRegressionTests(unittest.TestCase):
    def test_generation_reads_and_updates_chunk_memory(self):
        chars = [" ", "a", "b", "c", "d", "e"]
        tokenizer = CharTokenizer({ch: i for i, ch in enumerate(chars)}, chars)
        model = LiquidLanguageModel(
            vocab_size=len(chars), embedding_dim=8, hidden_size=8, num_layers=1,
            dropout=0.0, zoneout=0.0, tie_weights=True, local_conv=False,
            memory_slots=2,
        ).eval()

        with patch("liquid_lm.model.chunk_memory_read", wraps=chunk_memory_read) as read_memory, \
             patch("liquid_lm.model.chunk_memory_write", wraps=chunk_memory_write) as write_memory:
            model.generate(
                prompt="abcde", tokenizer=tokenizer, max_new_tokens=3,
                temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                repetition_window=0, device=torch.device("cpu"),
                stop_token_ids=[], memory_chunk_size=2,
            )

        self.assertEqual(read_memory.call_count, 4)
        self.assertEqual(write_memory.call_count, 3)

    def test_generation_rejects_non_positive_memory_chunk_size(self):
        chars = [" ", "a", "b"]
        tokenizer = CharTokenizer({ch: i for i, ch in enumerate(chars)}, chars)
        model = LiquidLanguageModel(
            vocab_size=len(chars), embedding_dim=8, hidden_size=8, num_layers=1,
            dropout=0.0, zoneout=0.0, tie_weights=True, local_conv=False,
            memory_slots=1,
        ).eval()
        with self.assertRaisesRegex(ValueError, "memory_chunk_size must be >= 1"):
            model.generate(
                prompt="ab", tokenizer=tokenizer, max_new_tokens=1,
                device=torch.device("cpu"), memory_chunk_size=0,
            )


class CliValidationRegressionTests(unittest.TestCase):
    def test_eval_batches_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "--eval-batches must be >= 1"):
            validate_eval_batches(0)
        validate_eval_batches(1)

    def test_invalid_training_ranges_are_rejected(self):
        defaults = dict(
            kd_alpha=0.3, kd_temperature=2.0, dialogue_anchor_ratio=0.7,
            min_lr_ratio=0.1, memory_slots=0, grad_clip=1.0,
            optimizer="adamw", muon_ns_steps=5,
        )
        validate_training_ranges(SimpleNamespace(**defaults))
        invalid = [
            {"kd_alpha": 1.1}, {"kd_temperature": 0.0},
            {"dialogue_anchor_ratio": -0.1}, {"min_lr_ratio": 1.1},
            {"memory_slots": -1}, {"grad_clip": -1.0},
            {"optimizer": "muon", "muon_ns_steps": 0},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                values = {**defaults, **changes}
                with self.assertRaises(ValueError):
                    validate_training_ranges(SimpleNamespace(**values))


class _FakeGradScaler:
    """Small deterministic GradScaler double for testing overflow handling."""

    def __init__(self, scale, skip_update):
        self.scale = float(scale)
        self.skip_update = skip_update

    def get_scale(self):
        return self.scale

    def step(self, optimizer):
        if not self.skip_update:
            optimizer.step()

    def update(self):
        self.scale = self.scale / 2 if self.skip_update else self.scale * 2


class GradientScalerRegressionTests(unittest.TestCase):
    def test_overflow_skip_is_not_counted_as_optimizer_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([1.0])
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = _FakeGradScaler(scale=8.0, skip_update=True)

        updated = grad_scaler_step_succeeded(optimizer, scaler)

        self.assertFalse(updated)
        self.assertEqual(parameter.item(), 1.0)

    def test_successful_scaled_update_is_counted(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([1.0])
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = _FakeGradScaler(scale=8.0, skip_update=False)

        updated = grad_scaler_step_succeeded(optimizer, scaler)

        self.assertTrue(updated)
        self.assertAlmostEqual(parameter.item(), 0.9, places=6)


class TokenWeightedGradientNormalizationTests(unittest.TestCase):
    def test_rescales_gradient_to_mean_over_valid_targets(self):
        model = torch.nn.Linear(1, 1, bias=False)
        # A summed-loss gradient of 3 divided by 6 possible positions is 0.5.
        model.weight.grad = torch.tensor([[0.5]])

        had_targets = normalize_gradients_by_valid_tokens(
            model, max_tokens=6, valid_tokens=3,
        )

        self.assertTrue(had_targets)
        self.assertAlmostEqual(model.weight.grad.item(), 1.0, places=6)

    def test_empty_group_is_not_rescaled(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.tensor([[0.5]])

        had_targets = normalize_gradients_by_valid_tokens(
            model, max_tokens=6, valid_tokens=0,
        )

        self.assertFalse(had_targets)
        self.assertAlmostEqual(model.weight.grad.item(), 0.5, places=6)

    def test_max_tokens_must_be_positive(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with self.assertRaisesRegex(ValueError, "max_tokens must be >= 1"):
            normalize_gradients_by_valid_tokens(model, max_tokens=0, valid_tokens=1)


class RecurrentModelRegressionTests(unittest.TestCase):
    @staticmethod
    def make_model(gate_mode, local_conv, wiring="dense"):
        return LiquidLanguageModel(
            vocab_size=23,
            embedding_dim=12,
            hidden_size=12,
            num_layers=2,
            ode_unfolds=2,
            dropout=0.0,
            zoneout=0.0,
            tie_weights=True,
            residual=True,
            multi_tau=True,
            local_conv=local_conv,
            gate_mode=gate_mode,
            wiring=wiring,
            ncp_command_frac=0.4,
            ncp_motor_frac=0.2,
        ).eval()

    def test_tokenwise_forward_matches_full_sequence_forward(self):
        # This checks recurrent state and the left-padded causal-convolution buffer.
        torch.manual_seed(7)
        tokens = torch.randint(0, 23, (2, 9), dtype=torch.long)
        for gate_mode in ("ltc", "cfc"):
            for local_conv in (False, True):
                with self.subTest(gate_mode=gate_mode, local_conv=local_conv):
                    model = self.make_model(gate_mode, local_conv)
                    with torch.inference_mode():
                        full_logits, _ = model(tokens)
                        hidden = None
                        pieces = []
                        for t in range(tokens.size(1)):
                            logits, hidden = model(tokens[:, t:t + 1], hidden=hidden)
                            pieces.append(logits)
                        streamed_logits = torch.cat(pieces, dim=1)
                    torch.testing.assert_close(
                        streamed_logits, full_logits, rtol=1e-4, atol=1e-5
                    )

    def test_dense_and_ncp_cells_backpropagate_finite_gradients(self):
        torch.manual_seed(11)
        tokens = torch.randint(0, 23, (2, 6), dtype=torch.long)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        for gate_mode in ("ltc", "cfc"):
            for wiring in ("dense", "ncp"):
                with self.subTest(gate_mode=gate_mode, wiring=wiring):
                    model = self.make_model(gate_mode, False, wiring=wiring)
                    model.train()
                    logits, _ = model(inputs)
                    loss = F.cross_entropy(logits.reshape(-1, 23), targets.reshape(-1))
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
                    if wiring == "ncp":
                        self.assertLess(model.cells[0].recurrent_param_fraction, 1.0)


class TritonLtcAutogradRegressionTests(unittest.TestCase):
    def test_triton_tail_forward_and_backward_match_torch_reference(self):
        from liquid_lm.layers import _TRITON_AVAILABLE, _ltc_tail_triton_autograd

        if not torch.cuda.is_available() or not _TRITON_AVAILABLE:
            self.skipTest("requires CUDA and the optional Triton package")

        torch.manual_seed(41)
        device = torch.device("cuda")
        batch, hidden = 3, 32
        probe = torch.randn(batch, hidden, device=device)

        for multi_tau in (False, True):
            with self.subTest(multi_tau=multi_tau):
                tau_width = hidden * (3 if multi_tau else 1)
                h0 = torch.randn(batch, hidden, device=device)
                cand0 = torch.tanh(torch.randn(batch, hidden, device=device))
                raw0 = torch.randn(batch, tau_width, device=device) * 0.7
                mix0 = torch.softmax(torch.randn(3, device=device), dim=0) if multi_tau else None

                h = h0.detach().clone().requires_grad_(True)
                cand = cand0.detach().clone().requires_grad_(True)
                raw = raw0.detach().clone().requires_grad_(True)
                mix = mix0.detach().clone().requires_grad_(True) if multi_tau else None
                actual = _ltc_tail_triton_autograd(h, cand, raw, mix, 0.75, 0.05, multi_tau)
                (actual * probe).sum().backward()
                actual_grads = [h.grad, cand.grad, raw.grad]
                if multi_tau:
                    actual_grads.append(mix.grad)

                hr = h0.detach().clone().requires_grad_(True)
                cr = cand0.detach().clone().requires_grad_(True)
                rr = raw0.detach().clone().requires_grad_(True)
                mr = mix0.detach().clone().requires_grad_(True) if multi_tau else None
                if multi_tau:
                    banks = rr.view(batch, 3, hidden)
                    taus = torch.stack((
                        F.softplus(banks[:, 0] - 2.0) + 0.05,
                        F.softplus(banks[:, 1]) + 0.01,
                        F.softplus(banks[:, 2] + 2.0) + 0.001,
                    ), dim=1)
                    tau = (taus * mr.view(1, 3, 1)).sum(dim=1)
                else:
                    tau = F.softplus(rr) + 0.05
                decay = torch.exp(-0.75 / tau)
                expected = decay * hr + (1.0 - decay) * cr
                (expected * probe).sum().backward()
                expected_grads = [hr.grad, cr.grad, rr.grad]
                if multi_tau:
                    expected_grads.append(mr.grad)

                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-4, atol=4e-5)


if __name__ == "__main__":
    unittest.main()
