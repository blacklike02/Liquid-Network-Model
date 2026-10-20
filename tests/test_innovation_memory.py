"""Causality, state transport, gradients and compatibility of side memory."""

import copy
import unittest

import torch
from torch.nn import functional as F

from liquid_lm.associative import InnovationMemory
from liquid_lm.cli import build_parser, validate_training_ranges
from liquid_lm.layers import _TRITON_AVAILABLE, chunk_memory_read, chunk_memory_reset
from liquid_lm.model import (
    ARCH_KEYS, LiquidLanguageModel, create_model_from_config, detach_hidden,
    load_model_state_compat,
)
from liquid_lm.tokenizer import CharTokenizer


def options(**overrides):
    result = dict(vocab_size=24, embedding_dim=16, hidden_size=16, num_layers=2,
                  dropout=0., zoneout=0., local_conv=True, innovation_memory_dim=4)
    result.update(overrides)
    return result


class InnovationMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)

    def test_equations_read_before_write_and_zero_time(self):
        module = InnovationMemory(8, 3).double()
        x = torch.randn(2, 8, dtype=torch.float64)
        state = torch.randn(2, 9, dtype=torch.float64)
        dt = 0.7
        q, k, v = module.qkv(module.norm(x)).chunk(3, -1)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        controls = module.control(module.norm(x))
        rate = F.softplus(module.log_rate) * (2 * controls[:, :3].sigmoid())
        old = state.reshape(2, 3, 3) * (-dt * rate).exp().unsqueeze(-1)
        predicted = torch.einsum('bij,bj->bi', old, k)
        error = v.tanh() - predicted
        energy = error.square().mean(-1, keepdim=True)
        novelty = energy / (energy + F.softplus(module.log_threshold) + 1e-6)
        beta = -torch.expm1(-dt * F.softplus(controls[:, -2:-1]))
        expected = old + torch.einsum('bi,bj->bij', beta * novelty * error, k)
        read = torch.einsum('bij,bj->bi', old, q) * controls[:, -1:].sigmoid()
        out, new = module(x, state, dt)
        torch.testing.assert_close(new, expected.flatten(1))
        torch.testing.assert_close(out, x + module.output(read))
        _, frozen = module(x, state, 0.)
        torch.testing.assert_close(frozen, state)
        for invalid in (-1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'dt'):
                module(x, state, invalid)
        # No current-token write can masquerade as successful retrieval.
        zero = torch.zeros_like(state)
        torch.testing.assert_close(module(x, zero)[0], x)

    def test_persistent_state_gradcheck(self):
        module = InnovationMemory(4, 2).double()
        x = torch.randn(1, 4, dtype=torch.float64)
        state = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda s: module(x, s), (state,)))

    def test_causal_chunk_and_token_equivalence(self):
        tokens = torch.randint(0, 24, (2, 9))
        for config in ({}, {'gate_mode': 'cfc', 'wiring': 'dense', 'ffn_mult': 0.5},
                       {'gate_mode': 'ltc', 'multi_tau': False, 'memory_slots': 2}):
            with self.subTest(config=config), torch.no_grad():
                model = LiquidLanguageModel(**options(**config)).eval()
                full, full_hidden = model(tokens)
                hidden, parts = None, []
                for chunk in tokens.split(1, dim=1):
                    logits, hidden = model(chunk, hidden)
                    parts.append(logits)
                torch.testing.assert_close(torch.cat(parts, 1), full, rtol=2e-5, atol=2e-6)
                for actual, expected in zip(hidden, full_hidden):
                    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-5)
                altered = tokens.clone()
                altered[:, 5:] = (altered[:, 5:] + 1) % 24
                torch.testing.assert_close(model(altered)[0][:, :5], full[:, :5])
                memory = chunk_memory_reset(model, 2, tokens.device)
                carried = chunk_memory_read(model, hidden, memory)
                torch.testing.assert_close(carried[model.num_layers], hidden[model.num_layers])
                self.assertTrue(all(not h.requires_grad for h in detach_hidden(hidden)))

    def test_checkpointing_gradients_match_and_generation_works(self):
        tokens = torch.randint(0, 24, (2, 7))
        plain = LiquidLanguageModel(**options())
        checkpointed = copy.deepcopy(plain)
        checkpointed.use_checkpoint = True
        for model in (plain, checkpointed):
            logits, hidden = model(tokens)
            F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
            self.assertEqual(hidden[model.num_layers].shape, (2, 16))
        for (name, p), (_, q) in zip(plain.named_parameters(), checkpointed.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            torch.testing.assert_close(p.grad, q.grad)
        for name, param in plain.innovation_memory.named_parameters():
            self.assertGreater(param.grad.abs().sum().item(), 0., name)
        tokenizer = CharTokenizer.train('abcdefghijklmnopqrstuvwx')
        generated = plain.eval().generate('abc', tokenizer, max_new_tokens=4, top_k=0)
        self.assertEqual(len(generated), 7)

    def test_config_roundtrip_and_legacy_default(self):
        model = LiquidLanguageModel(**options())
        config = {key: getattr(model, 'dropout_rate' if key == 'dropout' else key) for key in ARCH_KEYS}
        restored = create_model_from_config(24, config, torch.device('cpu'))
        load_model_state_compat(restored, model.state_dict())
        tokens = torch.randint(0, 24, (1, 6))
        torch.testing.assert_close(model.eval()(tokens)[0], restored.eval()(tokens)[0])
        config.pop('innovation_memory_dim')
        old = create_model_from_config(24, config, torch.device('cpu'))
        self.assertIsNone(old.innovation_memory)
        self.assertFalse(any('innovation_memory' in key for key in old.state_dict()))
        self.assertEqual(len(old.initial_hidden(1, torch.device('cpu'))), old.num_layers)
        load_model_state_compat(old, old.state_dict())
        with self.assertRaises(RuntimeError):
            load_model_state_compat(restored, old.state_dict())
        torch.manual_seed(17)
        a = LiquidLanguageModel(**options(innovation_memory_dim=0))
        torch.manual_seed(17)
        b = LiquidLanguageModel(**{k: v for k, v in options().items() if k != 'innovation_memory_dim'})
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_cli_default_and_invalid_dimension(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.innovation_memory_dim, 0)
        args.innovation_memory_dim = -1
        with self.assertRaisesRegex(ValueError, 'innovation-memory-dim'):
            validate_training_ranges(args)
        with self.assertRaisesRegex(ValueError, 'innovation_memory_dim'):
            LiquidLanguageModel(**options(innovation_memory_dim=-1))

    def test_document_reset_batch_isolation_and_input_state_immutability(self):
        model = LiquidLanguageModel(**options(memory_slots=2)).eval()
        first = torch.randint(0, 24, (2, 11))
        second = torch.randint(0, 24, (2, 7))
        with torch.inference_mode():
            expected, _ = model(second)
            _, old = model(first)
            snapshot = [h.clone() for h in old]
            model(second, old)
            for value, original in zip(old, snapshot):
                torch.testing.assert_close(value, original, rtol=0, atol=0)
            # Calling the model does not leave a hidden global document cache.
            torch.testing.assert_close(model(second)[0], expected, rtol=0, atol=0)
            for index in range(2):
                alone, _ = model(second[index:index + 1])
                torch.testing.assert_close(alone, expected[index:index + 1], rtol=3e-5, atol=1e-5)

    def test_long_streaming_state_stays_finite_and_has_fixed_size(self):
        model = LiquidLanguageModel(**options(num_layers=1)).eval()
        tokens = torch.randint(0, 24, (2, 1024))
        with torch.inference_mode():
            full, expected = model(tokens, last_token_only=True)
            hidden = None
            for chunk in tokens.split(32, dim=1):
                logits, hidden = model(chunk, hidden, last_token_only=True)
                self.assertEqual(hidden[1].shape, (2, 16))
                self.assertTrue(all(torch.isfinite(h).all() for h in hidden))
            torch.testing.assert_close(logits, full, rtol=3e-5, atol=1e-5)
            for actual, target in zip(hidden, expected):
                torch.testing.assert_close(actual, target, rtol=3e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available() and _TRITON_AVAILABLE, 'CUDA/Triton required')
class InnovationMemoryGpuTests(unittest.TestCase):
    def test_triton_amp_carry_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(184)
                reference = LiquidLanguageModel(**options()).cuda().train()
                candidate = copy.deepcopy(reference)
                self.assertEqual(candidate.set_kernel('triton'), 'triton')
                tokens = torch.randint(0, 24, (2, 8), device='cuda')
                outputs = []
                for model in (reference, candidate):
                    with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                        _, hidden = model(tokens[:, :4])
                        logits, hidden = model(tokens[:, 4:], hidden)
                        loss = logits.float().square().mean()
                    self.assertEqual(hidden[model.num_layers].dtype, torch.float32)
                    loss.backward()
                    for name, p in model.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    outputs.append(logits.float())
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0.03, atol=0.003)
                for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=0.08, atol=0.002, msg=name)


if __name__ == '__main__':
    unittest.main()
