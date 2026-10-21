"""Inference-only CUDA Graphs for complete one-token liquid transitions.

Sampling stays outside capture, preserving the caller's RNG consumption.
Each runner owns static session buffers and at most four boundary variants.
"""
from __future__ import annotations

from functools import wraps
import weakref

import torch

from .layers import chunk_memory_read, chunk_memory_write


class CudaGraphUnavailable(RuntimeError):
    """Capture failed before a real decode transition was executed."""


def cuda_amp_settings():
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:  # Older supported Torch builds use the CUDA-only API.
        enabled = torch.is_autocast_enabled()
    dtype = torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()
    return enabled, dtype


def cuda_precision_settings():
    """Read one precision API family; modern settings reject legacy getters."""
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    if hasattr(matmul, "fp32_precision"):
        return ("fp32_precision", torch.backends.fp32_precision, matmul.fp32_precision,
                cudnn.fp32_precision, cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision)
    return ("allow_tf32", matmul.allow_tf32, cudnn.allow_tf32)


def serialize_graph_generation(function):
    """Prevent simultaneous generations from sharing mutable graph buffers."""
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        if model._cuda_graphs_enabled:
            with model._cuda_graph_lock:
                return function(model, *args, **kwargs)
        return function(model, *args, **kwargs)
    return wrapped


def decode_signature(model, hidden, memory):
    """Invalidate concatenated weights and captured kernels when inputs change."""
    parameters = []
    cacheable = True
    for parameter in (*model.parameters(), *model.buffers()):
        try:
            version = parameter._version
        except RuntimeError:
            # Inference tensors have no mutation counter. Rebuild each request.
            version = None
            cacheable = False
        parameters.append((id(parameter), parameter.data_ptr(), version, parameter.dtype,
                           parameter.device, tuple(parameter.shape), tuple(parameter.stride())))
    def layout(tensor):
        return None if tensor is None else (tuple(tensor.shape), tensor.dtype, tensor.device)
    settings = ("eps", "min_tau", "multi_tau", "gate_mode", "kernel", "_scale",
                "padding", "stride", "dilation", "groups", "padding_mode", "approximate",
                "_conv_left_pad", "_conv_kernel_size", "_do_residual", "recurrent_state_count")
    modules = tuple((id(module), id(getattr(module.forward, "__func__", module.forward)), module.training,
                     tuple((key, getattr(module, key)) for key in settings if hasattr(module, key)))
                    for module in model.modules())
    return (tuple(parameters), tuple(layout(h) for h in hidden), layout(memory),
            modules,
            model._kernel, model._sequence_batching, id(model._step_fn),
            cuda_amp_settings(),
            cuda_precision_settings(),
            torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
            None if cacheable else object())


class CudaGraphDecode:
    """Private runner; outputs are overwritten by subsequent replays.

    The model owns this object, so the reverse model reference must stay weak.
    CUDA Graph capture requires no concurrent CUDA work in this process.
    """
    @torch.inference_mode()
    def __init__(self, model, hidden, memory):
        self._model = weakref.ref(model)
        self.hidden = [state.clone() for state in hidden]
        self.memory = None if memory is None else memory.clone()
        self.tokens = torch.zeros(hidden[0].shape[0], 1, dtype=torch.long, device=hidden[0].device)
        self.fused = [cell.fused_h_weights() for cell in model.cells]
        self.amp_enabled, self.amp_dtype = cuda_amp_settings()
        self.graphs = {}

    @property
    def capture_count(self):
        return len(self.graphs)

    @torch.inference_mode()
    def reset(self, hidden, memory):
        for destination, source in zip(self.hidden, hidden):
            destination.copy_(source)
        if self.memory is not None:
            self.memory.copy_(memory)

    def _transition(self, model, read_memory, write_memory):
        # Disable the autocast cache so parameter casts belong to the graph,
        # rather than an ambient context that may end before the next request.
        with torch.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype, cache_enabled=False):
            hidden = self.hidden
            if read_memory:
                hidden = chunk_memory_read(model, hidden, self.memory)
            logits, updated = model(self.tokens, hidden=hidden, fused=self.fused, last_token_only=True)
            memory = chunk_memory_write(model, updated, self.memory) if write_memory else None
            for destination, source in zip(self.hidden, updated):
                destination.copy_(source)
            if write_memory:
                self.memory.copy_(memory)
            return logits[:, -1, :].float()

    def _capture(self, key):
        model = self._model()
        if model is None:
            raise CudaGraphUnavailable("The model was released")
        saved_hidden = [state.clone() for state in self.hidden]
        saved_memory = None if self.memory is None else self.memory.clone()
        current = torch.cuda.current_stream(self.tokens.device)
        stream = torch.cuda.Stream(device=self.tokens.device)
        stream.wait_stream(current)
        generator = torch.cuda.default_generators[self.tokens.device.index]
        caller_rng = generator.graphsafe_get_state()
        capture_rng = generator.clone_state()
        failure = None
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._transition(model, *key)
            current.wait_stream(stream)
            # A failed capture can leave its registered RNG state in capture
            # mode. Isolate that registration from the real sampling generator.
            generator.graphsafe_set_state(capture_rng)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self._transition(model, *key)
            self.graphs[key] = (graph, output, capture_rng)
        except (RuntimeError, NotImplementedError) as exc:
            failure = str(exc)
        finally:
            # Warmup operates on scratch state only. Even a failed capture
            # must leave the logical session at its original token position.
            # torch.cuda.graph.__exit__ can raise before restoring its stream.
            def restore():
                torch.cuda.set_stream(current)
                generator.graphsafe_set_state(caller_rng)
                current.wait_stream(stream)
                self.reset(saved_hidden, saved_memory)
            try:
                restore()
            except RuntimeError as exc:
                # An invalidated capture can report its pending CUDA error on
                # the first cleanup copy. That check consumes the error; redo
                # the complete restore. Any other/second failure is fatal.
                if failure is None or "capture" not in str(exc).lower():
                    raise
                restore()
        if failure is not None:
            raise CudaGraphUnavailable(f"CUDA Graph capture failed: {failure}")

    @torch.inference_mode()
    def step(self, token, read_memory=False, write_memory=False):
        self.tokens.copy_(token)
        key = (bool(read_memory), bool(write_memory))
        with torch.cuda.device(self.tokens.device):
            if key not in self.graphs:
                self._capture(key)
            graph, output, _capture_rng = self.graphs[key]
            # A replay failure may have partially updated device state. Do not
            # hide it with an eager retry of an already-consumed token.
            graph.replay()
        return output
