"""Shared-weight depth keeps independent temporal streams and saved semantics."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from liquid_lm.layers import _TRITON_AVAILABLE, chunk_memory_read, chunk_memory_reset, chunk_memory_write
from liquid_lm.cli import build_parser, validate_training_ranges
from liquid_lm.model import ARCH_KEYS, LiquidLanguageModel, create_model_from_config, load_checkpoint_raw, save_checkpoint
from liquid_lm.tokenizer import CharTokenizer


def make(**changes):
    config = dict(vocab_size=23, embedding_dim=16, hidden_size=16, num_layers=2,
                  recurrent_passes=2, ffn_mult=1., dropout=0., zoneout=0.)
    config.update(changes)
    return LiquidLanguageModel(**config)


class RecurrentDepthTests(unittest.TestCase):
    def test_parameter_sharing_and_independent_temporal_buffers(self):
        torch.manual_seed(98)
        baseline = make(recurrent_passes=1)
        torch.manual_seed(98)
        deep = make(recurrent_passes=3)
        self.assertEqual(list(baseline.state_dict()), list(deep.state_dict()))
        for key, tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor, deep.state_dict()[key], rtol=0, atol=0)
        for batch in (1, 2):
            state = deep.initial_hidden(batch, torch.device("cpu"))
            self.assertEqual(len(state), 6)
            self.assertEqual(len({x.data_ptr() for x in state}), 6)
        calls = [0, 0]
        handles = [block.register_forward_hook(lambda m, a, out, i=i: calls.__setitem__(i, calls[i]+1))
                   for i, block in enumerate(deep.ffns)]
        try:
            deep(torch.randint(0, 23, (2, 4)))
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(calls, [4, 4])  # Once per token, not once per depth pass.

    def test_streaming_and_sequence_batched_outputs_states_and_gradients(self):
        for gate in ("ltc", "cfc"):
            with self.subTest(gate=gate):
                torch.manual_seed(98)
                reference = make(gate_mode=gate).eval()
                batched = copy.deepcopy(reference)
                batched.set_sequence_batching(True)
                tokens = torch.randint(0, 23, (2, 9))
                expected, state = reference(tokens)
                actual, other_state = batched(tokens)
                torch.testing.assert_close(actual, expected)
                for a, b in zip(state, other_state): torch.testing.assert_close(a, b)
                expected.square().mean().backward()
                actual.square().mean().backward()
                for a, b in zip(reference.parameters(), batched.parameters()):
                    self.assertIsNotNone(a.grad)
                    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)
                hidden, pieces = None, []
                with torch.inference_mode():
                    for chunk in tokens.split(3, dim=1):
                        logits, hidden = batched(chunk, hidden)
                        pieces.append(logits)
                torch.testing.assert_close(torch.cat(pieces, 1), expected)
                self.assertEqual(len(hidden), 5)  # Four recurrent streams and convolution.

    def test_checkpointed_backward_and_memory_use_final_depth_stream(self):
        reference = make(memory_slots=3, memory_dim=8, innovation_memory_dim=4)
        checkpointed = copy.deepcopy(reference)
        checkpointed.use_checkpoint = True
        tokens = torch.randint(0, 23, (2, 5))
        a, state = reference(tokens)
        b, _ = checkpointed(tokens)
        a.square().mean().backward(); b.square().mean().backward()
        for x, y in zip(reference.parameters(), checkpointed.parameters()):
            if x.grad is not None: torch.testing.assert_close(x.grad, y.grad)
        memory = chunk_memory_reset(reference, 2, torch.device("cpu"))
        memory = chunk_memory_write(reference, state, memory)
        torch.testing.assert_close(memory[:, -1, :-1], state[3])
        restored = chunk_memory_read(reference, state, memory)
        for index in (0, 1, 2, 4, 5):
            torch.testing.assert_close(restored[index], state[index], rtol=0, atol=0)

    def test_configuration_defaults_roundtrip_and_validation(self):
        config = dict(embedding_dim=16, hidden_size=16, num_layers=2, norm_type="rms")
        old = create_model_from_config(23, config, torch.device("cpu"))
        self.assertEqual(old.recurrent_passes, 1)
        for passes in (2, 3):
            model = create_model_from_config(23, dict(config, recurrent_passes=passes), torch.device("cpu"))
            self.assertEqual(model.recurrent_state_count, 2*passes)
            restored = create_model_from_config(23, dict(config, recurrent_passes=passes), torch.device("cpu"))
            restored.load_state_dict(model.state_dict(), strict=True)
        for passes in (0, -1, 1.5, True):
            with self.subTest(passes=passes), self.assertRaisesRegex(ValueError, "positive integer"):
                make(recurrent_passes=passes)
        with self.assertRaisesRegex(ValueError, "equal|embedding_dim"):
            make(embedding_dim=8, tie_weights=False)
        with self.assertRaisesRegex(ValueError, "recurrent-passes"):
            validate_training_ranges(build_parser().parse_args(["--recurrent-passes", "0"]))
        model = make(recurrent_passes=3).eval()
        saved_args = {key: getattr(model, "dropout_rate" if key == "dropout" else key) for key in ARCH_KEYS}
        saved_args["recurrent_passes"] = 1  # Model semantics are authoritative.
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        tokenizer = CharTokenizer.train("abcdefghijklmnopqrstuvw")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "depth.pt"
            save_checkpoint(path, model, optimizer, scheduler, None, 0, 0, None, None,
                            float("inf"), tokenizer, SimpleNamespace(**saved_args))
            payload = load_checkpoint_raw(path, torch.device("cpu"))
        self.assertEqual(payload["config"]["recurrent_passes"], 3)
        restored = create_model_from_config(23, payload["config"], torch.device("cpu")).eval()
        restored.load_state_dict(payload["model_state"], strict=True)
        tokens = torch.randint(0, 23, (1, 5))
        torch.testing.assert_close(model(tokens)[0], restored(tokens)[0], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and _TRITON_AVAILABLE, "CUDA/Triton required")
    def test_cuda_depth_amp_training_and_memory(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(99)
            model = make(memory_slots=3, memory_dim=8).cuda()
            model.set_kernel("triton")
            model.set_sequence_batching(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            tokens = torch.randint(0, 23, (2, 8), device="cuda")
            state = None
            memory = chunk_memory_reset(model, 2, torch.device("cuda"))
            for _ in range(2):
                state = chunk_memory_read(model, state, memory)
                with torch.autocast("cuda", dtype=dtype, enabled=dtype!=torch.float32):
                    logits, state = model(tokens, state)
                    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), tokens.flatten())
                loss.backward()
                memory = chunk_memory_write(model, state, memory)
                state = [x.detach() for x in state]
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
