"""Native normalization, occupied compact memory and checkpoint compatibility."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from liquid_lm.cli import build_parser
from liquid_lm.layers import (ChunkMemory, CompactChunkMemory, LiquidCell, ResidualFFN,
                              RMSNorm, _TRITON_AVAILABLE,
                              chunk_memory_read, chunk_memory_reset, chunk_memory_write)
from liquid_lm.model import (LiquidLanguageModel, create_model_from_config, detach_hidden,
                             load_model_state_compat, save_checkpoint)
from liquid_lm.tokenizer import CharTokenizer
from liquid_lm.training import restore_resume_data_settings


class NativeRMSNormTests(unittest.TestCase):
    def test_outputs_and_gradients_match_explicit_formula(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    torch.manual_seed(91)
                    module = RMSNorm(19).to(device)
                    module.weight.data.uniform_(0.5, 1.5)
                    source = torch.randn(3, 4, 19, device=device, dtype=dtype)
                    x = source.clone().requires_grad_()
                    reference_x = source.clone().requires_grad_()
                    weight = module.weight.detach().clone().requires_grad_()
                    xf = reference_x.float()
                    reference = (xf * (xf.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
                                 * weight).to(dtype)
                    output = module(x)
                    probe = torch.randn_like(output)
                    (output * probe).float().sum().backward()
                    (reference * probe).float().sum().backward()
                    tolerance = 3e-6 if dtype == torch.float32 else 2e-2
                    torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(x.grad, reference_x.grad, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(module.weight.grad, weight.grad, atol=2e-5, rtol=2e-5)
                    self.assertEqual(output.dtype, dtype)

    def test_double_precision_gradcheck_and_zero_input(self):
        module = RMSNorm(5).double()
        x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(module, (x,)))
        output = module(torch.zeros_like(x))
        torch.testing.assert_close(output, torch.zeros_like(x), rtol=0, atol=0)

class CompactMemoryTests(unittest.TestCase):
    def test_empty_memory_is_an_exact_identity_even_with_invalid_payload(self):
        module = CompactChunkMemory(12, 4, 5)
        state = torch.randn(3, 12, requires_grad=True)
        memory = module.init_memory(3, torch.device("cpu"))
        memory[..., :-1] = float("nan")
        output = module.read(state, memory)
        torch.testing.assert_close(output, state, rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(state.grad, torch.ones_like(state))

    def test_fifo_occupancy_detachment_and_bounded_storage(self):
        module = CompactChunkMemory(12, 3, 5)
        memory = module.init_memory(2, torch.device("cpu"))
        history = []
        for _ in range(12):
            state = torch.randn(2, 12, requires_grad=True)
            history.append(state.detach())
            memory = module.write(state, memory)
            self.assertFalse(memory.requires_grad)
            self.assertEqual(memory.shape, (2, 3, 13))
            self.assertEqual(memory.untyped_storage().nbytes(), 2 * 3 * 13 * 4)
        torch.testing.assert_close(memory[..., :-1], torch.stack(history[-3:], 1))
        self.assertTrue(memory[..., -1].eq(1).all())

    def test_empty_slots_never_dilute_or_influence_a_real_read(self):
        torch.manual_seed(3)
        module = CompactChunkMemory(12, 4, 5)
        state = torch.randn(2, 12)
        memory = module.write(torch.randn(2, 12), module.init_memory(2, state.device))
        with torch.no_grad():
            expected = module.read(state, memory)
            # Identical occupied history, with the invalid slots omitted.
            actual = module.read(state, memory[:, -1:])
            changed = memory.clone()
            changed[:, :-1, :-1] = torch.randn_like(changed[:, :-1, :-1]) * 1000
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(module.read(state, changed), expected)

    def test_read_parameters_learn_without_gradients_into_written_history(self):
        torch.manual_seed(3)
        module = CompactChunkMemory(12, 3, 5)
        memory = module.init_memory(2, torch.device("cpu"))
        history = [torch.randn(2, 12, requires_grad=True) for _ in range(2)]
        for state in history:
            memory = module.write(state, memory)
        query = torch.randn(2, 12, requires_grad=True)
        module.read(query, memory).square().sum().backward()
        self.assertTrue(all(state.grad is None for state in history))
        self.assertIsNotNone(query.grad)
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_amp_state_and_compiled_read(self):
        module = CompactChunkMemory(12, 3, 5)
        memory = module.write(torch.randn(2, 12), module.init_memory(2, torch.device("cpu")))
        for dtype in (torch.float16, torch.bfloat16):
            query = torch.randn(2, 12, dtype=dtype, requires_grad=True)
            output = module.read(query, memory)
            torch.testing.assert_close(output, module.read(query.float(), memory))
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(query.grad).all())
        compiled = torch.compile(module.read, backend="eager", fullgraph=True)
        query = torch.randn(2, 12)
        torch.testing.assert_close(compiled(query, memory), module.read(query, memory))

    def test_parameter_budget_and_slot_independence(self):
        expected = 4 * 384 * 64 + 2 * 64 ** 2 + 2 * 64
        for slots in (2, 8):
            module = CompactChunkMemory(384, slots, 64)
            self.assertEqual(sum(p.numel() for p in module.parameters()), expected)
        self.assertEqual(CompactChunkMemory(12, 2, 64).memory_dim, 12)

    @unittest.skipUnless(torch.cuda.is_available() and _TRITON_AVAILABLE, "CUDA/Triton required")
    def test_full_chunked_cuda_training_with_amp_and_checkpointing(self):
        for gate in ("ltc", "cfc"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(gate=gate, dtype=dtype):
                    model = LiquidLanguageModel(23, 16, 16, 2, dropout=0, zoneout=0,
                        norm_type="rms", memory_slots=3, memory_dim=8,
                        ffn_mult=0.5, gate_mode=gate, use_checkpoint=True).cuda()
                    if gate == "ltc":
                        model.set_kernel("triton")
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        hidden, memory = None, chunk_memory_reset(model, 2, torch.device("cuda"))
                        for _ in range(4):
                            hidden = chunk_memory_read(model, hidden, memory)
                            tokens = torch.randint(0, 23, (2, 3), device="cuda")
                            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                                logits, hidden = model(tokens, hidden)
                                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten())
                            loss.backward()
                            memory = chunk_memory_write(model, hidden, memory)
                            hidden = detach_hidden(hidden)
                        for name, parameter in model.named_parameters():
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        optimizer.step()


class RetainedConfigurationTests(unittest.TestCase):
    def test_all_model_normalizations_are_rms(self):
        model = LiquidLanguageModel(23, 12, 12, 2, ffn_mult=1)
        norms = [module for module in model.modules() if isinstance(module, RMSNorm)]
        self.assertEqual(len(norms), 5)  # Two cells, two FFNs, final readout.
        self.assertFalse(any(isinstance(module, torch.nn.LayerNorm) for module in model.modules()))
        for constructor, arguments in ((LiquidLanguageModel, (23, 12, 12, 1)),
                                       (LiquidCell, (12, 12)), (ResidualFFN, (12, 1))):
            with self.subTest(constructor=constructor.__name__):
                with self.assertRaisesRegex(ValueError, "Only RMSNorm"):
                    constructor(*arguments, norm_type="layer")

    def test_legacy_or_unknown_normalization_metadata_is_rejected(self):
        config = dict(embedding_dim=12, hidden_size=12, num_layers=1)
        for legacy in ({}, {"norm_type": "layer"}, {"norm_type": None}, {"norm_type": "unknown"}):
            with self.subTest(legacy=legacy), self.assertRaisesRegex(ValueError, "not explicitly RMSNorm"):
                create_model_from_config(23, dict(config, **legacy), torch.device("cpu"))

    def test_layernorm_state_rejected_before_modifying_model(self):
        model = LiquidLanguageModel(23, 12, 12, 1)
        before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        fake = {key: tensor + 1 for key, tensor in before.items()}
        fake["cells.0.norm.bias"] = torch.zeros(12)
        with self.assertRaisesRegex(RuntimeError, "LayerNorm state"):
            load_model_state_compat(model, fake)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_saved_normalization_metadata_comes_from_model(self):
        config = dict(embedding_dim=12, hidden_size=12, num_layers=1,
                      dropout=0., zoneout=0., tie_weights=True, residual=True,
                      multi_tau=True, local_conv=True)
        model = LiquidLanguageModel(23, **config).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        tokenizer = CharTokenizer.train(" abcdefghijklmnopqrstuv")
        # Even stale external args cannot label an RMS model as LayerNorm.
        args = SimpleNamespace(**config, norm_type="layer")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rms.pt"
            save_checkpoint(path, model, optimizer, scheduler, None, 0, 0,
                            3., 3., 3., tokenizer, args)
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["config"]["norm_type"], "rms")
        self.assertEqual(args.norm_type, "layer")
        restored = create_model_from_config(23, payload["config"], torch.device("cpu")).eval()
        load_model_state_compat(restored, payload["model_state"])
        tokens = torch.randint(0, 23, (2, 5))
        torch.testing.assert_close(model(tokens)[0], restored(tokens)[0], rtol=0, atol=0)

    def test_training_defaults_use_retained_features(self):
        args = build_parser().parse_args([])
        self.assertEqual((args.norm_type, args.memory_slots, args.memory_dim, args.state_carry),
                         ("rms", 3, 64, 4))

    def test_removed_flags_fail_instead_of_silently_being_ignored(self):
        for option, value in (("--wiring", "ncp"), ("--ode-solver", "heun"),
                              ("--ode-unfolds", "2"), ("--fim-rate", "0.5"),
                              ("--norm-type", "layer"), ("--norm-type", "rms"),
                              ("--ncp-command-frac", "0.4"), ("--ncp-motor-frac", "0.2")):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args([option, value])

    def test_checkpoint_roundtrip_and_missing_memory_dim_use_legacy_layout(self):
        config = dict(embedding_dim=12, hidden_size=12, num_layers=1,
                      dropout=0., zoneout=0., memory_slots=3,
                      tie_weights=True, residual=True, local_conv=True, multi_tau=True)
        for width, expected_class in ((0, ChunkMemory), (5, CompactChunkMemory)):
            cfg = dict(config, memory_dim=width, norm_type="rms")
            original = create_model_from_config(23, cfg, torch.device("cpu"))
            restored = create_model_from_config(23, cfg, torch.device("cpu"))
            restored.load_state_dict(original.state_dict(), strict=True)
            self.assertIsInstance(restored.chunk_memory, expected_class)
            tokens = torch.randint(0, 23, (2, 5))
            torch.testing.assert_close(original(tokens)[0], restored(tokens)[0])
        del cfg["memory_dim"]
        self.assertIsInstance(create_model_from_config(23, cfg, torch.device("cpu")).chunk_memory,
                              ChunkMemory)

    def test_removed_checkpoint_equations_are_rejected_explicitly(self):
        config = dict(embedding_dim=12, hidden_size=12, num_layers=1)
        for change in (dict(wiring="ncp"), dict(ode_solver="heun"), dict(ode_unfolds=2)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "retired"):
                create_model_from_config(23, dict(config, **change), torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "retired FIM"):
            restore_resume_data_settings(SimpleNamespace(), dict(fim_rate=0.5))


if __name__ == "__main__":
    unittest.main()
