"""Regression tests for Liquid-Network-Model's critical sequence paths."""
import unittest

import torch
import torch.nn.functional as F

from liquid_lm.model import LiquidLanguageModel
from liquid_lm.training import (
    get_batch,
    grad_scaler_step_succeeded,
    masked_cross_entropy,
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


if __name__ == "__main__":
    unittest.main()
