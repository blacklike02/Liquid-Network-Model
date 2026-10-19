"""Regression checks for fused LTC activations and first-order training."""

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import liquid_lm.layers as layers
from liquid_lm.model import LiquidLanguageModel


@unittest.skipUnless(torch.cuda.is_available() and layers._TRITON_AVAILABLE,
                     "requires CUDA and Triton")
class TritonFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.old_tf32
        torch.set_num_threads(cls.old_threads)

    def test_activations_preserve_strides_dtypes_and_all_input_gradients(self):
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            for mixed in (False, True):
                with self.subTest(dtype=dtype, mixed=mixed):
                    torch.manual_seed(71)
                    # Odd width, several CTAs, noncontiguous sequence slices
                    # and a noncontiguous recurrent projection.
                    batch, hidden = 3, 513
                    fused = torch.randn(2 * hidden, batch, device="cuda", dtype=dtype).T
                    xp = torch.randn(batch, 3, hidden, device="cuda",
                                     dtype=torch.float32 if mixed else dtype)[:, 1]
                    tx = torch.randn(batch, hidden * 2, device="cuda", dtype=dtype)[:, ::2]
                    reference = [x.detach().requires_grad_() for x in (fused, xp, tx)]
                    actual = [x.detach().requires_grad_() for x in (fused, xp, tx)]
                    expected = (torch.tanh(reference[1] + reference[0][:, :hidden]),
                                torch.tanh(reference[0][:, hidden:] + reference[2]))
                    result = layers._ltc_activations_triton_autograd(*actual)
                    # Include a broadcast gradient and a transposed gradient.
                    gradients = (torch.randn(1, hidden, device="cuda").expand(batch, -1),
                                 torch.randn(hidden, batch, device="cuda").T)
                    expected_grads = torch.autograd.grad(expected, reference, gradients)
                    actual_grads = torch.autograd.grad(result, actual, gradients)
                    atol = {torch.float32: 5e-7, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
                    for got, want in zip((*result, *actual_grads), (*expected, *expected_grads)):
                        self.assertEqual(got.dtype, want.dtype)
                        self.assertTrue(torch.isfinite(got).all().item())
                        torch.testing.assert_close(got, want, atol=atol, rtol=atol)
                    with torch.no_grad():
                        inference = layers._ltc_activations_triton_autograd(*actual)
                    for got, want in zip(inference, result):
                        torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_unused_activation_has_zero_input_gradient(self):
        for output_index in (0, 1):
            with self.subTest(output=output_index):
                fused = torch.randn(2, 34, device="cuda", requires_grad=True)
                xp = torch.randn(2, 17, device="cuda", requires_grad=True)
                tx = torch.randn_like(xp, requires_grad=True)
                outputs = layers._ltc_activations_triton_autograd(fused, xp, tx)
                outputs[output_index].sum().backward()
                inactive = tx if output_index == 0 else xp
                self.assertEqual(torch.count_nonzero(inactive.grad).item(), 0)
                inactive_slice = fused.grad[:, 17:] if output_index == 0 else fused.grad[:, :17]
                self.assertEqual(torch.count_nonzero(inactive_slice).item(), 0)
                self.assertTrue(torch.isfinite(fused.grad).all().item())

    def test_tanh_backward_preserves_native_rounding(self):
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                values = torch.tensor([[0.5, 1.5, 2.5, 3.5, -0.75, -1.75]],
                                      device="cuda", dtype=dtype)
                reference = [torch.zeros(1, 12, device="cuda", dtype=dtype, requires_grad=True),
                             values.clone().requires_grad_(), values.clone().requires_grad_()]
                actual = [x.detach().clone().requires_grad_() for x in reference]
                expected = (torch.tanh(reference[1] + reference[0][:, :6]),
                            torch.tanh(reference[0][:, 6:] + reference[2]))
                result = layers._ltc_activations_triton_autograd(*actual)
                gradients = tuple(torch.ones_like(x) for x in expected)
                expected_grads = torch.autograd.grad(expected, reference, gradients)
                actual_grads = torch.autograd.grad(result, actual, gradients)
                for got, want in zip((*result, *actual_grads), (*expected, *expected_grads)):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_regularized_optimizer_steps_match_torch_with_checkpoint_and_ncp(self):
        for wiring, norm, checkpoint in (("dense", "layer", False), ("ncp", "rms", True)):
            with self.subTest(wiring=wiring, norm=norm, checkpoint=checkpoint):
                torch.manual_seed(81)
                reference = LiquidLanguageModel(
                    23, embedding_dim=16, hidden_size=16, num_layers=2,
                    dropout=0.15, zoneout=0.2, ode_unfolds=2,
                    ffn_mult=1.0, wiring=wiring, norm_type=norm,
                    use_checkpoint=checkpoint,
                ).cuda().train()
                actual = copy.deepcopy(reference)
                self.assertEqual(actual.set_kernel("triton"), "triton")
                self.assertEqual(list(actual.state_dict()), list(reference.state_dict()))
                tokens = torch.randint(0, 23, (2, 6), device="cuda")
                targets = torch.randint(0, 23, tokens.shape, device="cuda")
                optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3)
                              for m in (reference, actual)]
                with patch.object(layers, "_ltc_activations_triton_autograd",
                                  wraps=layers._ltc_activations_triton_autograd) as fused:
                    for step in range(3):
                        outputs = []
                        for model, optimizer in zip((reference, actual), optimizers):
                            torch.manual_seed(100 + step)
                            optimizer.zero_grad(set_to_none=True)
                            logits, _ = model(tokens)
                            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                            loss.backward()
                            outputs.append(logits.detach())
                        torch.testing.assert_close(outputs[1], outputs[0], atol=2e-5, rtol=8e-4)
                        for (name, got), (_, want) in zip(actual.named_parameters(), reference.named_parameters()):
                            self.assertIsNotNone(got.grad, name)
                            self.assertIsNotNone(want.grad, name)
                            self.assertTrue(torch.isfinite(got.grad).all().item(), name)
                            torch.testing.assert_close(got.grad, want.grad, atol=2e-5, rtol=1e-3,
                                                       msg=lambda message: f"{name}: {message}")
                        for optimizer in optimizers:
                            optimizer.step()
                    self.assertGreater(fused.call_count, 0)
                for got, want in zip(actual.parameters(), reference.parameters()):
                    torch.testing.assert_close(got, want, atol=3e-5, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
