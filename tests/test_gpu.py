"""Small CUDA regressions; every test is skipped on CPU-only installations.

Run from the repository root with:
    python -m unittest discover -s tests -p test_gpu.py -v

These check numerical correctness and training state. Timing belongs in the
separate benchmark, because tiny recurrent models are dominated by launch cost.
"""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from liquid_lm.layers import chunk_memory_read, chunk_memory_reset, chunk_memory_write
from liquid_lm.model import (
    LiquidLanguageModel,
    create_model_from_config,
    detach_hidden,
    load_checkpoint_raw,
    save_checkpoint,
)
from liquid_lm.optim import make_optimizer, make_scheduler
from liquid_lm.tokenizer import CharTokenizer
from liquid_lm.training import (
    autocast_context,
    grad_scaler_step_succeeded,
    make_grad_scaler,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-capable PyTorch installation")
class CudaRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        cls.old_threads = torch.get_num_threads()
        cls.old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls.old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        # CPU/GPU parity should compare full FP32 arithmetic. Restore the user's
        # global preferences when the class finishes, including on test failure.
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = cls.old_cudnn_tf32
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(112)

    @staticmethod
    def model_options(**changes):
        options = dict(
            vocab_size=23, embedding_dim=16, hidden_size=16, num_layers=2,
            ode_unfolds=1, dropout=0.0, zoneout=0.0, tie_weights=True,
            residual=True, local_conv=True, multi_tau=True,
            ffn_mult=1.0, norm_type="rms",
        )
        options.update(changes)
        return options

    @staticmethod
    def optimizer_options(name):
        return SimpleNamespace(
            optimizer=name, lr=0.002, weight_decay=0.01,
            muon_lr=0.003, muon_momentum=0.95, muon_ns_steps=3,
        )

    def make_scaler(self):
        scaler = make_grad_scaler(True, self.device)
        self.assertIsNotNone(scaler)
        # A conservative scale keeps the finite-update test independent of
        # GradScaler's expected initial overflow calibration. Overflow itself
        # is exercised explicitly below with the same real CUDA scaler.
        state = scaler.state_dict()
        state["scale"] = 128.0
        scaler.load_state_dict(state)
        return scaler

    def amp_dtypes(self):
        yield torch.float16
        if torch.cuda.is_bf16_supported():
            yield torch.bfloat16

    def assert_finite_gradients(self, model):
        gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients, "backward did not produce gradients")
        for name, gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all().item(), name)
        self.assertTrue(any(torch.count_nonzero(g).item() > 0 for _, g in gradients))

    def assert_gradients_close(self, actual, reference):
        actual_parameters = dict(actual.named_parameters())
        for name, parameter in reference.named_parameters():
            gradient = actual_parameters[name].grad
            if parameter.grad is None:
                self.assertIsNone(gradient, name)
            else:
                self.assertIsNotNone(gradient, name)
                torch.testing.assert_close(
                    gradient.cpu(), parameter.grad.cpu(),
                    rtol=8e-4, atol=3e-5, msg=lambda message: f"{name}: {message}",
                )

    @staticmethod
    def loss(logits, targets):
        return F.cross_entropy(logits.reshape(-1, 23), targets.reshape(-1))

    def training_step(self, model, optimizer, tokens, targets, dtype, scaler=None):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(self.device, dtype):
            logits, _ = model(tokens)
            loss = self.loss(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        self.assert_finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if scaler is None:
            optimizer.step()
        else:
            self.assertTrue(grad_scaler_step_succeeded(optimizer, scaler))
        return loss.detach().item()

    def test_fp32_cpu_cuda_logits_hidden_and_gradients(self):
        tokens = torch.randint(0, 23, (2, 7))
        targets = torch.randint(0, 23, tokens.shape)
        cases = (
            dict(gate_mode="ltc", wiring="dense", multi_tau=False),
            dict(gate_mode="ltc", wiring="dense", memory_slots=2, memory_dim=8),
            dict(gate_mode="ltc", wiring="dense", norm_type="rms", mlp_head=True),
            dict(gate_mode="cfc", wiring="dense", norm_type="rms"),
            dict(gate_mode="cfc", wiring="dense", mlp_head=True),
        )
        for options in cases:
            with self.subTest(**options):
                cpu = LiquidLanguageModel(**self.model_options(**options)).train()
                gpu = copy.deepcopy(cpu).to(self.device)
                expected, expected_hidden = cpu(tokens)
                actual, actual_hidden = gpu(tokens.to(self.device))
                self.assertTrue(actual.is_cuda)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
                self.assertEqual(len(actual_hidden), cpu.num_layers + 1)
                for a, b in zip(actual_hidden, expected_hidden):
                    self.assertTrue(a.is_cuda)
                    torch.testing.assert_close(a.cpu(), b, rtol=2e-4, atol=3e-5)
                self.loss(expected, targets).backward()
                self.loss(actual, targets.to(self.device)).backward()
                self.assert_finite_gradients(gpu)
                self.assert_gradients_close(gpu, cpu)

    def test_cuda_tokenwise_carry_matches_full_sequence(self):
        tokens = torch.randint(0, 23, (2, 9), device=self.device)
        for gate in ("ltc", "cfc"):
            for multi_tau in (False, True):
                with self.subTest(gate=gate, multi_tau=multi_tau):
                    model = LiquidLanguageModel(**self.model_options(
                        gate_mode=gate, multi_tau=multi_tau,
                    )).to(self.device).eval()
                    with torch.inference_mode():
                        expected, expected_hidden = model(tokens)
                        hidden, pieces = None, []
                        for token in tokens.split(1, dim=1):
                            logits, hidden = model(token, hidden=hidden)
                            pieces.append(logits)
                        actual = torch.cat(pieces, dim=1)
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                    for a, b in zip(hidden, expected_hidden):
                        torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-5)

    def test_activation_checkpoint_preserves_cuda_backward(self):
        tokens = torch.randint(0, 23, (2, 5), device=self.device)
        targets = torch.randint(0, 23, tokens.shape, device=self.device)
        for gate in ("ltc", "cfc"):
            for multi_tau in (False, True):
                with self.subTest(gate=gate, multi_tau=multi_tau):
                    reference = LiquidLanguageModel(**self.model_options(
                        gate_mode=gate, multi_tau=multi_tau,
                    )).to(self.device).train()
                    checkpointed = copy.deepcopy(reference)
                    checkpointed.use_checkpoint = True
                    expected, _ = reference(tokens)
                    actual, _ = checkpointed(tokens)
                    self.loss(expected, targets).backward()
                    self.loss(actual, targets).backward()
                    torch.testing.assert_close(actual, expected)
                    self.assert_gradients_close(checkpointed, reference)

    def test_amp_fp16_bf16_train_with_adamw_and_muon(self):
        tokens = torch.randint(0, 23, (2, 6), device=self.device)
        targets = torch.randint(0, 23, tokens.shape, device=self.device)
        for dtype in self.amp_dtypes():
            for optimizer_name in ("adamw", "muon"):
                for gate in ("ltc", "cfc"):
                    with self.subTest(dtype=dtype, optimizer=optimizer_name, gate=gate):
                        model = LiquidLanguageModel(**self.model_options(
                            gate_mode=gate, wiring="dense", norm_type="rms",
                        )).to(self.device).train()
                        optimizer = make_optimizer(model, self.optimizer_options(optimizer_name), self.device)
                        scaler = self.make_scaler() if dtype == torch.float16 else None
                        with torch.no_grad():
                            expected, _ = model(tokens)
                            with autocast_context(self.device, dtype):
                                actual, _ = model(tokens)
                        self.assertEqual(actual.dtype, dtype)
                        # Recurrence can amplify BF16 rounding near zero. Bound
                        # both the total relative error and the largest absolute
                        # error instead of dividing each small logit by itself.
                        error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-8)
                        max_error = (actual.float() - expected).abs().max().item()
                        self.assertLess(error.item(), 0.05 if dtype == torch.bfloat16 else 0.01)
                        self.assertLess(max_error, 0.02 if dtype == torch.bfloat16 else 0.003)
                        self.assertLess(abs(
                            self.loss(actual.float(), targets).item() - self.loss(expected, targets).item()
                        ), 0.01)
                        previous = model.embedding.weight.detach().clone()
                        for _ in range(3):
                            self.training_step(model, optimizer, tokens, targets, dtype, scaler)
                        self.assertFalse(torch.equal(previous, model.embedding.weight))
                        self.assertTrue(all(torch.isfinite(p).all().item() for p in model.parameters()))

    def test_real_grad_scaler_overflow_skips_entire_optimizer(self):
        tokens = torch.randint(0, 23, (2, 4), device=self.device)
        targets = torch.randint(0, 23, tokens.shape, device=self.device)
        for optimizer_name in ("adamw", "muon"):
            # Embedding belongs to AdamW; the recurrent matrix belongs to Muon.
            for poisoned_name in ("embedding.weight", "cells.0.h_proj.weight"):
                with self.subTest(optimizer=optimizer_name, poisoned=poisoned_name):
                    model = LiquidLanguageModel(**self.model_options()).to(self.device).train()
                    optimizer = make_optimizer(model, self.optimizer_options(optimizer_name), self.device)
                    scaler = self.make_scaler()
                    self.training_step(model, optimizer, tokens, targets, torch.float16, scaler)
                    before = {name: p.detach().clone() for name, p in model.named_parameters()}
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(self.device, torch.float16):
                        logits, _ = model(tokens)
                        loss = self.loss(logits, targets)
                    scaler.scale(loss).backward()
                    dict(model.named_parameters())[poisoned_name].grad.view(-1)[0] = float("inf")
                    scaler.unscale_(optimizer)
                    previous_scale = scaler.get_scale()
                    self.assertFalse(grad_scaler_step_succeeded(optimizer, scaler))
                    self.assertLess(scaler.get_scale(), previous_scale)
                    for name, parameter in model.named_parameters():
                        self.assertTrue(torch.equal(parameter, before[name]), name)

    def run_memory_chunks(self, model, tokens, targets, dtype=None):
        device = tokens.device
        hidden = None
        memory = chunk_memory_reset(model, tokens.size(0), device)
        pieces = []
        for start in range(0, tokens.size(1), 3):
            # Keep the same autocast boundary as training/evaluate: memory
            # reads also have to accept a carried BF16/FP16 CfC hidden state.
            hidden = chunk_memory_read(model, hidden, memory)
            with autocast_context(device, dtype):
                logits, hidden = model(tokens[:, start:start + 3], hidden=hidden)
                loss = self.loss(logits, targets[:, start:start + 3]) / 4
            loss.backward()
            pieces.append(logits.detach())
            if start + 3 < tokens.size(1):
                memory = chunk_memory_write(model, hidden, memory).detach()
                self.assertFalse(memory.requires_grad)
            hidden = detach_hidden(hidden)
            self.assertTrue(all(not h.requires_grad for h in hidden))
        self.assert_finite_gradients(model)
        for name in ("chunk_memory.init_slot", "chunk_memory.summarize.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(torch.count_nonzero(gradient).item(), 0, name)
        return torch.cat(pieces, dim=1), hidden, memory

    def test_chunk_memory_carry_cpu_cuda_parity(self):
        tokens = torch.randint(0, 23, (2, 12))
        targets = torch.randint(0, 23, tokens.shape)
        for gate in ("ltc", "cfc"):
            with self.subTest(gate=gate):
                cpu = LiquidLanguageModel(**self.model_options(
                    gate_mode=gate, memory_slots=2,
                )).train()
                gpu = copy.deepcopy(cpu).to(self.device)
                expected, expected_hidden, expected_memory = self.run_memory_chunks(cpu, tokens, targets)
                actual, actual_hidden, actual_memory = self.run_memory_chunks(
                    gpu, tokens.to(self.device), targets.to(self.device),
                )
                torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-5)
                torch.testing.assert_close(actual_memory.cpu(), expected_memory, rtol=3e-4, atol=3e-5)
                for a, b in zip(actual_hidden, expected_hidden):
                    torch.testing.assert_close(a.cpu(), b, rtol=3e-4, atol=3e-5)
                self.assert_gradients_close(gpu, cpu)

    def test_amp_chunk_memory_accepts_carried_low_precision_hidden(self):
        tokens = torch.randint(0, 23, (2, 12), device=self.device)
        targets = torch.randint(0, 23, tokens.shape, device=self.device)
        for dtype in self.amp_dtypes():
            for gate in ("ltc", "cfc"):
                with self.subTest(dtype=dtype, gate=gate):
                    model = LiquidLanguageModel(**self.model_options(
                        gate_mode=gate, memory_slots=2, norm_type="rms",
                    )).to(self.device).train()
                    logits, _, memory = self.run_memory_chunks(model, tokens, targets, dtype)
                    self.assertTrue(torch.isfinite(logits).all().item())
                    self.assertTrue(memory.is_cuda)

    def test_cuda_checkpoint_restores_optimizer_scheduler_scaler_and_cpu_weights(self):
        tokens = torch.randint(0, 23, (2, 6), device=self.device)
        targets = torch.randint(0, 23, tokens.shape, device=self.device)
        tokenizer = CharTokenizer.train(" abcdefghijklmnopqrstuv")
        self.assertEqual(tokenizer.vocab_size, 23)
        for optimizer_name in ("adamw", "muon"):
            with self.subTest(optimizer=optimizer_name), tempfile.TemporaryDirectory() as directory:
                options = self.model_options(gate_mode="cfc", wiring="dense", norm_type="rms", memory_slots=2)
                model = LiquidLanguageModel(**options).to(self.device).train()
                optimizer_args = self.optimizer_options(optimizer_name)
                optimizer = make_optimizer(model, optimizer_args, self.device)
                scheduler = make_scheduler(optimizer, 4, 0, 0.1)
                scaler = self.make_scaler()
                self.training_step(model, optimizer, tokens, targets, torch.float16, scaler)
                scheduler.step()
                args = SimpleNamespace(**options, **vars(optimizer_args))
                path = Path(directory) / "nested" / "cuda.pt"
                save_checkpoint(
                    path, model, optimizer, scheduler, scaler, 1, 1,
                    3.0, 3.1, 3.1, tokenizer, args,
                )
                payload = load_checkpoint_raw(path, self.device)
                resumed = create_model_from_config(23, payload["config"], self.device).train()
                resumed.load_state_dict(payload["model_state"])
                resumed_optimizer = make_optimizer(resumed, optimizer_args, self.device)
                resumed_scheduler = make_scheduler(resumed_optimizer, 4, 0, 0.1)
                # Match run_training's order: constructing LambdaLR after
                # restoring the optimizer would overwrite the saved LR.
                resumed_optimizer.load_state_dict(payload["optimizer_state"])
                resumed_scheduler.load_state_dict(payload["scheduler_state"])
                resumed_scaler = self.make_scaler()
                resumed_scaler.load_state_dict(payload["scaler_state"])
                self.assertEqual(resumed_scaler.state_dict(), scaler.state_dict())
                self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
                cpu_payload = load_checkpoint_raw(path, torch.device("cpu"))
                cpu = create_model_from_config(23, cpu_payload["config"], torch.device("cpu")).eval()
                cpu.load_state_dict(cpu_payload["model_state"])
                with torch.inference_mode():
                    expected, _ = cpu(tokens.cpu())
                    actual, _ = resumed(tokens)
                torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-5)
                # The next identical update must restore moments/momentum, not
                # merely reproduce the inference output from saved weights.
                self.training_step(model, optimizer, tokens, targets, torch.float16, scaler)
                self.training_step(resumed, resumed_optimizer, tokens, targets, torch.float16, resumed_scaler)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        dict(resumed.named_parameters())[name], parameter,
                        rtol=2e-5, atol=2e-6, msg=lambda message: f"{name}: {message}",
                    )

    def test_greedy_generation_matches_cpu_with_and_without_chunk_memory(self):
        tokenizer = CharTokenizer.train(" abcdefghijklmnopqrstuv")
        for gate in ("ltc", "cfc"):
            for slots in (0, 2):
                with self.subTest(gate=gate, memory_slots=slots):
                    cpu = LiquidLanguageModel(**self.model_options(
                        gate_mode=gate, memory_slots=slots,
                    )).eval()
                    gpu = copy.deepcopy(cpu).to(self.device)
                    options = dict(
                        prompt="abcdefabcdef", tokenizer=tokenizer, max_new_tokens=6,
                        top_k=1, top_p=1.0, repetition_penalty=1.0,
                        repetition_window=0, only_new=True, memory_chunk_size=3,
                    )
                    expected = cpu.generate(**options, device=torch.device("cpu"))
                    actual = gpu.generate(**options, device=self.device)
                    self.assertEqual(len(actual), 6)
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
