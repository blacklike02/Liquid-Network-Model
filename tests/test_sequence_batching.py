"""Sequence batching preserves causality, carried state and fallback semantics."""
import copy
import unittest
from unittest.mock import patch

import torch

from benchmark_compute import check_equivalence
from liquid_lm.cli import build_parser
from liquid_lm.layers import _TRITON_AVAILABLE, chunk_memory_read, chunk_memory_reset, chunk_memory_write
from liquid_lm.model import ARCH_KEYS, LiquidLanguageModel


def pair(**overrides):
    config = dict(vocab_size=23, embedding_dim=16, hidden_size=16, num_layers=2, dropout=0, zoneout=0)
    config.update(overrides)
    torch.manual_seed(73)
    before = LiquidLanguageModel(**config)
    after = copy.deepcopy(before)
    after.set_sequence_batching(True)
    return before, after


class SequenceBatchingTests(unittest.TestCase):
    def test_outputs_states_and_all_gradients_across_architectures(self):
        tokens = torch.randint(0, 23, (2, 7))
        targets = torch.randint(0, 23, tokens.shape)
        for config in ({}, {"num_layers": 1, "ffn_mult": 1},
                       {"num_layers": 3, "multi_tau": False, "ode_unfolds": 1},
                       {"gate_mode": "cfc", "wiring": "dense", "ffn_mult": 1},
                       {"norm_type": "rms", "innovation_memory_dim": 4, "ffn_mult": 0.5},
                       {"residual": False, "mlp_head": True, "multi_tau": False,
                        "embedding_dim": 12, "tie_weights": False, "local_conv": False},
                       {"use_checkpoint": True, "wiring": "dense", "ffn_mult": 0.5}):
            with self.subTest(config=config):
                models = pair(**config)
                check_equivalence(models, tokens, targets, torch.device("cpu"), "fp32", 111)

    def test_double_precision_gradient_parity(self):
        before, after = [model.double() for model in pair(ffn_mult=1)]
        tokens = torch.randint(0, 23, (2, 6))
        expected, expected_state = before(tokens)
        actual, state = after(tokens)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for p, q in zip(before.parameters(), after.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-10)
        for p, q in zip(expected_state, state):
            torch.testing.assert_close(p, q, rtol=1e-11, atol=1e-11)

    def test_prefill_last_token_causality_and_chunked_memory(self):
        before, after = pair(ffn_mult=0.5, memory_slots=3, innovation_memory_dim=4)
        before.eval()
        after.eval()
        tokens = torch.randint(0, 23, (2, 12))
        with torch.inference_mode():
            full, state = after(tokens)
            last, last_state = after(tokens, last_token_only=True)
            torch.testing.assert_close(last, full[:, -1:])
            for p, q in zip(state, last_state):
                torch.testing.assert_close(p, q)
            self.assertEqual(last_state[-1].untyped_storage().nbytes(), last_state[-1].numel() * last_state[-1].element_size())
            changed = tokens.clone()
            changed[:, 7:] = (changed[:, 7:] + 1) % 23
            torch.testing.assert_close(after(changed)[0][:, :7], full[:, :7])
            hidden, memories = [None, None], [chunk_memory_reset(m, 2, torch.device("cpu")) for m in (before, after)]
            for start in (0, 4, 8):
                outputs = []
                for i, model in enumerate((before, after)):
                    hidden[i] = chunk_memory_read(model, hidden[i], memories[i])
                    output, hidden[i] = model(tokens[:, start:start + 4], hidden[i])
                    outputs.append(output)
                    memories[i] = chunk_memory_write(model, hidden[i], memories[i])
                torch.testing.assert_close(outputs[0], outputs[1], atol=2e-5, rtol=2e-5)
                for p, q in zip(hidden[0], hidden[1]):
                    torch.testing.assert_close(p, q, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(memories[0], memories[1], atol=2e-5, rtol=2e-5)

    def test_stochastic_training_keeps_outputs_gradients_and_rng_exact(self):
        for config in ({"dropout": 0.15}, {"zoneout": 0.2},
                       {"dropout": 0.1, "zoneout": 0.2, "ffn_mult": 1, "use_checkpoint": True}):
            before, after = pair(**config)
            tokens = torch.randint(0, 23, (2, 6))
            results, rng = [], []
            with patch.object(after, "_forward_sequence_layers", side_effect=AssertionError("stochastic schedule reordered")):
                for model in (before, after):
                    torch.manual_seed(927)
                    output = model(tokens)[0]
                    output.square().sum().backward()
                    results.append(output)
                    rng.append(torch.get_rng_state())
            self.assertTrue(torch.equal(results[0], results[1]))
            self.assertTrue(torch.equal(rng[0], rng[1]))
            for p, q in zip(before.parameters(), after.parameters()):
                self.assertTrue(torch.equal(p.grad, q.grad))

    def test_projection_and_ffn_use_sequence_dimension_but_decode_uses_step(self):
        model = pair(ffn_mult=1)[1].eval()
        shapes = []
        handles = [ffn.register_forward_pre_hook(lambda m, args: shapes.append(tuple(args[0].shape))) for ffn in model.ffns]
        try:
            with patch.object(model.cells[1], "precompute_x", wraps=model.cells[1].precompute_x) as project:
                with torch.inference_mode():
                    model(torch.zeros(2, 8, dtype=torch.long))
                self.assertEqual(len(project.call_args_list), 1)
                self.assertEqual(tuple(project.call_args.args[0].shape), (2, 8, 16))
            self.assertEqual(shapes, [(2, 8, 16)] * 2)
            with patch.object(model, "_forward_sequence_layers", side_effect=AssertionError("decode reordered")):
                with torch.inference_mode():
                    model(torch.zeros(2, 1, dtype=torch.long))
        finally:
            for handle in handles:
                handle.remove()

    def test_compiled_step_keeps_its_existing_execution_path(self):
        before, after = pair()
        after._step_fn = torch.compile(after._step_impl, backend="eager", fullgraph=True)
        tokens = torch.randint(0, 23, (2, 3))
        with patch.object(after, "_forward_sequence_layers", side_effect=AssertionError("compiled step bypassed")):
            torch.testing.assert_close(after(tokens)[0], before(tokens)[0])

    def test_runtime_switch_preserves_state_keys_and_default_is_off(self):
        before, after = pair()
        self.assertFalse(before._sequence_batching)
        self.assertTrue(after._sequence_batching)
        self.assertEqual(list(before.state_dict()), list(after.state_dict()))
        self.assertNotIn("sequence_batching", ARCH_KEYS)
        self.assertFalse(build_parser().parse_args([]).sequence_batching)
        self.assertTrue(build_parser().parse_args(["--sequence-batching"]).sequence_batching)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SequenceBatchingGpuTests(unittest.TestCase):
    def test_cuda_precisions_gradients_checkpointing_and_triton(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            tokens = torch.randint(0, 23, (2, 7), device="cuda")
            targets = torch.randint(0, 23, tokens.shape, device="cuda")
            for precision in ("fp32", "bf16", "fp16"):
                for config in ({"ffn_mult": 1}, {"gate_mode": "cfc", "innovation_memory_dim": 4, "ffn_mult": 0.5},
                               {"wiring": "dense", "use_checkpoint": True}):
                    with self.subTest(precision=precision, config=config):
                        models = [m.cuda() for m in pair(**config)]
                        if _TRITON_AVAILABLE and config.get("gate_mode") != "cfc":
                            for model in models:
                                self.assertEqual(model.set_kernel("triton"), "triton")
                        check_equivalence(models, tokens, targets, torch.device("cuda"), precision, 420)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    unittest.main()
