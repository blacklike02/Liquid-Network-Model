"""Whole-token CUDA Graph parity, session isolation, lifecycle and fallback."""
import contextlib
import copy
import gc
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import warnings
import weakref

import torch

from liquid_lm.cli import build_parser, main
from liquid_lm.cuda_graphs import CudaGraphDecode, cuda_precision_settings
from liquid_lm.layers import _TRITON_AVAILABLE, chunk_memory_read, chunk_memory_reset, chunk_memory_write
from liquid_lm.model import LiquidLanguageModel
from liquid_lm.tokenizer import CharTokenizer


def make(**changes):
    config = dict(vocab_size=7, embedding_dim=16, hidden_size=16, num_layers=2,
                  dropout=.1, zoneout=.1, ffn_mult=1., memory_slots=3, memory_dim=8)
    config.update(changes)
    return LiquidLanguageModel(**config).eval()


TOKENIZER = CharTokenizer.train(" abcdef")


def generate(model, prompt="abc", **changes):
    options = dict(max_new_tokens=14, top_k=5, top_p=.85, repetition_penalty=1.2,
                   repetition_window=4, memory_chunk_size=3, only_new=True,
                   device=next(model.parameters()).device)
    options.update(changes)
    return model.generate(prompt, TOKENIZER, **options)


class CudaGraphCpuTests(unittest.TestCase):
    def test_precision_signature_supports_modern_and_legacy_api_families(self):
        class Modern(SimpleNamespace):
            @property
            def allow_tf32(self):
                raise AssertionError("Modern precision settings reject this legacy getter")
        matmul = Modern(fp32_precision="tf32")
        cudnn = Modern(fp32_precision="ieee", conv=Modern(fp32_precision="tf32"),
                       rnn=Modern(fp32_precision="ieee"))
        backend = SimpleNamespace(fp32_precision="ieee", cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn)
        with patch.object(torch, "backends", backend):
            before = cuda_precision_settings()
            self.assertEqual(before, ("fp32_precision", "ieee", "tf32", "ieee", "tf32", "ieee"))
            cudnn.conv.fp32_precision = "ieee"
            self.assertNotEqual(before, cuda_precision_settings())
        legacy = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                 cudnn=SimpleNamespace(allow_tf32=True))
        with patch.object(torch, "backends", legacy):
            self.assertEqual(cuda_precision_settings(), ("allow_tf32", False, True))

    def test_cpu_fallback_preserves_sampling_and_does_not_capture(self):
        torch.manual_seed(12)
        model = make()
        keys = list(model.state_dict())
        torch.manual_seed(91)
        expected = generate(model)
        model.set_cuda_graphs(True)
        torch.manual_seed(91)
        with warnings.catch_warnings(record=True) as caught, \
             patch("liquid_lm.model.CudaGraphDecode", side_effect=AssertionError("CPU capture")):
            warnings.simplefilter("always")
            actual = generate(model)
        self.assertEqual(actual, expected)
        self.assertEqual(list(model.state_dict()), keys)
        self.assertEqual(model.cuda_graph_stats()["replays"], 0)
        self.assertEqual(model.cuda_graph_stats()["fallbacks"], 1)
        self.assertTrue(any("CUDA device required" in str(w.message) for w in caught))

    def test_runtime_resources_are_not_copied_and_mode_changes_clear_cache(self):
        model = make()
        model.set_cuda_graphs(True)
        model._cuda_graph_runner = object()
        copied = copy.deepcopy(model)
        self.assertIsNone(copied._cuda_graph_runner)
        self.assertTrue(copied.cuda_graph_stats()["enabled"])
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, copied.state_dict()[key], rtol=0, atol=0)
        for change in (lambda: model.train(), lambda: model.to(dtype=torch.float64),
                       lambda: model.set_kernel("torch"), lambda: model.set_sequence_batching(True)):
            model._cuda_graph_runner = object()
            change()
            self.assertIsNone(model._cuda_graph_runner)
        with self.assertRaisesRegex(ValueError, "Disable CUDA Graphs"):
            model.enable_compiled_step()

    def test_cli_rejects_training_and_nested_compile_modes(self):
        self.assertFalse(build_parser().parse_args([]).cuda_graphs)
        self.assertTrue(build_parser().parse_args(["--cuda-graphs"]).cuda_graphs)
        for options in (["--cuda-graphs"], ["--generate-only", "--cuda-graphs", "--compile"]):
            with patch("sys.argv", ["liquid_text_model.py", *options]), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 patch("liquid_lm.cli.run_training") as train:
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                train.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaGraphGpuTests(unittest.TestCase):
    def setUp(self):
        self.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.previous_tf32

    def test_logits_and_all_states_at_memory_boundaries(self):
        cases = [("ltc", "torch", 0, 0, 0, 1), ("cfc", "torch", 2, 0, 0, 1),
                 ("ltc", "torch", 3, 8, 4, 2)]
        if _TRITON_AVAILABLE:
            cases.append(("ltc", "triton", 3, 8, 4, 2))
        for gate, kernel, slots, dim, innovation, passes in cases:
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                with self.subTest(gate=gate, kernel=kernel, dtype=dtype, passes=passes):
                    torch.manual_seed(61)
                    model = make(gate_mode=gate, memory_slots=slots, memory_dim=dim,
                                 innovation_memory_dim=innovation, recurrent_passes=passes).cuda()
                    self.assertEqual(model.set_kernel(kernel), kernel)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype!=torch.float32):
                        hidden = model.initial_hidden(2, torch.device("cuda"))
                        if model.local_conv:
                            hidden.append(torch.randn(2, model._conv_left_pad, model.embedding_dim, device="cuda"))
                        # Production capture starts after eager prompt prefill;
                        # CfC autocast establishes the carried-state dtype there.
                        _, hidden = model(torch.zeros(2, 1, dtype=torch.long, device="cuda"), hidden)
                        memory = chunk_memory_reset(model, 2, torch.device("cuda"))
                        runner = CudaGraphDecode(model, hidden, memory)
                        fused = [cell.fused_h_weights() for cell in model.cells]
                        for step in range(12):
                            token = torch.tensor([[step%7], [(step+2)%7]], device="cuda")
                            # Exercise all four variants, including an empty
                            # memory read and simultaneous read/write (T1).
                            read = slots > 0 and step%3 == 0
                            write = slots > 0 and (step%3 == 2 or step == 0)
                            if read: hidden = chunk_memory_read(model, hidden, memory)
                            expected, hidden = model(token, hidden, fused=fused, last_token_only=True)
                            if write: memory = chunk_memory_write(model, hidden, memory)
                            actual = runner.step(token, read, write)
                            tol = dict(rtol=2e-4, atol=2e-5) if dtype==torch.float32 else dict(rtol=.02, atol=.015)
                            torch.testing.assert_close(actual, expected[:, -1].float(), **tol)
                            for a, b in zip(runner.hidden, hidden): torch.testing.assert_close(a, b, **tol)
                            if memory is not None: torch.testing.assert_close(runner.memory, memory, **tol)
                        self.assertLessEqual(runner.capture_count, 4)

    def test_seeded_sampling_reuse_eos_and_partial_chunks(self):
        for gate in ("ltc", "cfc"):
            model = make(gate_mode=gate, recurrent_passes=2, innovation_memory_dim=4,
                         local_conv=gate=="cfc", mlp_head=gate=="ltc").cuda()
            model.set_sequence_batching(True)
            reference = copy.deepcopy(model)
            model.set_cuda_graphs(True)
            for dtype in (torch.float32, torch.bfloat16):
                for prompt, chunk in (("a", 3), ("abc", 3), ("abcdefa", 1), ("bc", 3)):
                    with self.subTest(gate=gate, dtype=dtype, prompt=prompt, chunk=chunk):
                        with torch.autocast("cuda", dtype=dtype, enabled=dtype!=torch.float32):
                            torch.manual_seed(817)
                            expected = generate(reference, prompt, memory_chunk_size=chunk)
                            expected_rng = torch.cuda.get_rng_state()
                            torch.manual_seed(817)
                            actual = generate(model, prompt, memory_chunk_size=chunk)
                            actual_rng = torch.cuda.get_rng_state()
                        self.assertEqual(actual, expected)
                        torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
                        self.assertEqual(model.cuda_graph_stats()["fallbacks"], 0)
                        self.assertLessEqual(model.cuda_graph_stats()["cached_graphs"], 4)
            captures = model.cuda_graph_stats()["captures"]
            generate(model, "bc")
            reused = model.cuda_graph_stats()["captures"]
            generate(model, "a")
            self.assertEqual(model.cuda_graph_stats()["captures"], reused)
            self.assertGreaterEqual(reused, captures)
            first = generate(reference, max_new_tokens=1, top_k=1)
            before = model.cuda_graph_stats()["replays"]
            stopped = generate(model, max_new_tokens=30, top_k=1, stop_token_ids=TOKENIZER.encode(first))
            self.assertEqual(stopped, "")
            self.assertEqual(model.cuda_graph_stats()["replays"], before)

    def test_failed_later_capture_restores_session_before_eager_fallback(self):
        model = make().cuda()
        reference = copy.deepcopy(model)
        model.set_cuda_graphs(True)
        original = CudaGraphDecode._transition
        warm_steps = [0]
        def transition(runner, owner, read, write):
            if read and not write:
                warm_steps[0] += 1
            if warm_steps[0] == 2:
                raise RuntimeError("simulated unsupported boundary capture")
            return original(runner, owner, read, write)
        torch.manual_seed(917)
        expected = generate(reference, "ab")
        torch.manual_seed(917)
        with patch.object(CudaGraphDecode, "_transition", transition), self.assertWarnsRegex(RuntimeWarning, "simulated"):
            actual = generate(model, "ab")
        self.assertEqual(actual, expected)
        self.assertEqual(model.cuda_graph_stats()["fallbacks"], 1)
        self.assertGreater(model.cuda_graph_stats()["replays"], 0)
        before = model.cuda_graph_stats()["captures"]
        generate(model, "ab")
        self.assertEqual(model.cuda_graph_stats()["captures"], before)
        model.clear_cuda_graph_cache()
        calls = []
        handle = model.readout.register_forward_hook(lambda module, args, result: calls.append(1))
        try:
            with self.assertWarnsRegex(RuntimeWarning, "forward hooks"):
                generate(model)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 14)
        model.clear_cuda_graph_cache()
        caller_stream = torch.cuda.current_stream()
        def unsupported(runner, owner, read, write):
            result = original(runner, owner, read, write)
            if torch.cuda.is_current_stream_capturing():
                runner.tokens[0, 0].item()  # Real prohibited CPU synchronization.
            return result
        torch.manual_seed(917)
        with patch.object(CudaGraphDecode, "_transition", unsupported), \
             self.assertWarnsRegex(RuntimeWarning, "CUDA Graph capture failed"):
            actual = generate(model, "ab")
        self.assertEqual(actual, expected)
        self.assertEqual(torch.cuda.current_stream(), caller_stream)

    def test_weight_invalidation_memory_bound_and_model_lifetime(self):
        model = make().cuda()
        model.set_cuda_graphs(True)
        generate(model)
        captures = model.cuda_graph_stats()["captures"]
        with torch.no_grad(): model.readout.bias.add_(torch.linspace(-1, 1, 7, device="cuda"))
        reference = copy.deepcopy(model)
        reference.set_cuda_graphs(False)
        torch.manual_seed(81);expected = generate(reference)
        torch.manual_seed(81);actual = generate(model)
        self.assertEqual(actual, expected)
        self.assertGreater(model.cuda_graph_stats()["captures"], captures)
        captures = model.cuda_graph_stats()["captures"]
        model.cells[0].min_tau *= 2
        reference.cells[0].min_tau *= 2
        torch.manual_seed(81);expected = generate(reference)
        torch.manual_seed(81);actual = generate(model)
        self.assertEqual(actual, expected)
        self.assertGreater(model.cuda_graph_stats()["captures"], captures)
        baseline = torch.cuda.memory_allocated()
        captures = model.cuda_graph_stats()["captures"]
        for _ in range(12): generate(model, "abcdefa")
        torch.cuda.synchronize()
        self.assertEqual(model.cuda_graph_stats()["captures"], captures)
        self.assertLessEqual(torch.cuda.memory_allocated()-baseline, 64*1024)
        enabled = gc.isenabled();gc.disable()
        try:
            owner = weakref.ref(model)
            parameter = weakref.ref(model.embedding.weight)
            del model
            self.assertIsNone(owner())
            self.assertIsNone(parameter())
        finally:
            if enabled: gc.enable()


if __name__ == "__main__": unittest.main()
