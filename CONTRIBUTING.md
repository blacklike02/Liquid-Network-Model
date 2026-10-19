# Contributing

Thank you for your interest in the project! Changes, ideas, and experiment
results are welcome.

## Documentation language

English is the primary language for public documentation, commit messages and
pull request descriptions. Russian translations are welcome; keep their commands
and reported measurements consistent with the English versions.

## Project layout

Run all commands from the repository root. The `liquid_lm` package uses a flat
module layout:

- `model.py`: language model, generation, checkpoint I/O and architecture keys.
- `layers.py`: LTC/CfC cells, NCP wiring, ChunkMemory and the optional Triton path.
- `training.py`, `optim.py`: training, distillation, evaluation and optimizers.
- `data.py`, `tokenizer.py`: corpus loading and tokenization.
- `cli.py`, `inference.py`, `utils.py`: arguments, inference and shared helpers.
- `liquid_text_model.py`: thin CLI entry point; `python -m liquid_lm` also works.
- `tests/test_regressions.py`: data, masking, resume, recurrent models and Triton.
- `tests/test_audit.py`: KD numerics, safe checkpoint writes, generation prefill,
  nonfinite gradients and carried AMP state.
- `tests/test_gpu.py`: CUDA parity, AMP, optimizer/scaler, memory, resume and
  generation checks.
- `tests/test_triton_fusions.py`: activation dtype/stride and gradient checks,
  native tanh-backward rounding,
  regularized optimizer steps with NCP/checkpointing.
- `tests/check_triton_mixed.py`: standalone 18-case Triton correctness checker
  for mixed dtypes and full-model gradients; run separately from test discovery.
- `benchmark_model.py`: CPU training/backward, generation and KD microbenchmarks.
- `benchmark_gpu.py`: synchronized CPU/CUDA training and generation comparisons.
- `benchmark_triton_tail.py`: isolated LTC-tail forward/backward comparison.
- `research_liquid.py`: context/tau diagnostics, matched-token training-horizon
  experiments and compatible checkpoint averaging.
- `tests/test_research.py`: averaging compatibility, token semantics, nonfinite
  weights, tensor/buffer invariants and overflow protection.

## Tests and reproducible measurements

Use a Python environment with the required dependencies and a CUDA-enabled
PyTorch build for GPU tests. CPU-only PyTorch cannot use CUDA even when the
driver detects a GPU. Check the selected interpreter and build before
comparing results:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m unittest discover -s tests -v
```

The same discovery command runs CPU and CUDA checks. CUDA/Triton tests skip when
their prerequisites are unavailable; report skips alongside passes. For the
optional backends, Triton must support your OS and GPU, and `torch.compile`
needs a working compiler environment. The verified Windows configuration used
`triton-windows 3.8.0.post28`.

Save fresh measurements under the ignored `runs/` directory. These PowerShell
commands reproduce the FP32 comparison workloads:

```powershell
python benchmark_model.py --threads 1 --warmup 3 --repeats 11 --output runs/benchmark_cpu.json
python benchmark_gpu.py --device both --cases train_realistic16 generate128 --precisions fp32 --threads 1 --warmup 3 --repeats 7 --output runs/benchmark_torch.json
$env:TRITON_CACHE_DIR=Join-Path (Get-Location) 'runs/triton_cache'
python tests/check_triton_mixed.py --output runs/triton_correctness.json
python benchmark_gpu.py --device cuda --kernel triton --cases train_realistic16 generate128 --precisions fp32 --threads 1 --warmup 3 --repeats 7 --output runs/benchmark_triton.json
python benchmark_gpu.py --device cuda --kernel triton --cases train_realistic16 train_regularized16 --precisions fp32 bf16 fp16 --threads 1 --warmup 5 --repeats 9 --output runs/benchmark_training.json
$env:TORCHINDUCTOR_CACHE_DIR=Join-Path (Get-Location) 'runs/inductor_cache'
$env:TORCHINDUCTOR_COMPILE_THREADS='1'
python benchmark_gpu.py --device cuda --compile --cases train_realistic16 generate128 --precisions fp32 --threads 1 --warmup 3 --repeats 7 --output runs/benchmark_compile.json
python benchmark_triton_tail.py --device cuda --batch-size 64 --hidden-size 384 --dtype fp32 --warmup 20 --iterations 100
```

`--compile` and `--kernel triton` are separate benchmark modes. To compare AMP,
use `--precisions fp32 bf16 fp16` for eager/Triton and `--precisions fp32 bf16`
for the verified compiled matrix. The standalone Triton checker requires CUDA,
Triton and BF16 support, rejects missing or nonfinite gradients and writes its
declared tolerances and metrics to JSON. Run GPU checks sequentially. The tail
microbenchmark measures only the LTC update kernel; use `benchmark_gpu.py` for
model-level claims. `benchmark_model.py` training timings exclude the optimizer
and have a different scope from `benchmark_gpu.py`.

### Verified results: 19 October 2026

The reference configuration used Windows 11, Python 3.12.10, NVIDIA GeForce RTX 5050
Laptop GPU (8 GB, 8151 MiB reported), driver 616.56 and PyTorch 2.11.0+cu128.
The CPU-only environment used PyTorch 2.14.1+cpu.

| Validation | Result |
|---|---|
| Current full suite, CUDA with Triton available | 53 passed, no skips, including 5 research/averaging tests |
| Current full suite, CPU-only environment | 39 passed, 14 expected CUDA/Triton skips; 53 discovered |
| Initial GPU CLI audit: training, resume, generation, KD and AMP | All 13 passed |
| Initial compiled CLI training and generation | Both passed |
| Activation-fusion follow-up: training, resume, BF16 NCP/memory/checkpointing, generation, selftest | All 5 passed |
| Triton numerical comparisons, including mixed dtypes and full-model backward | All 18 passed |

The final pre-commit rerun after removing local datasets passed all 53 tests in
49.329 s on the CUDA environment. The CPU-only environment passed 39 tests with
14 expected skips in 5.804 s. The suite uses temporary fixtures and does not
require the deleted Shakespeare corpus or local trained checkpoints.

The tests cover the corrected saturated-logit KD calculation, rejection of NaN
gradients before weight updates, checkpoint preservation after failed writes,
last-position-only generation readout and CfC AMP state carried into ChunkMemory.
GPU checks also exercise real GradScaler overflow handling, AdamW/Muon updates,
checkpoint restoration and CPU/CUDA parity. Triton checks reject missing
gradients, and CLI selftest exits with code 2 on a numerical mismatch.

Initial audit before activation fusion: median FP32 times for the
controlled comparison (the Triton path fused only the LTC tail):

| Workload | CPU, 1 thread | CUDA eager | CUDA Triton | CUDA compiled |
|---|---:|---:|---:|---:|
| Training step, B=16, T=64 | 2293.1 ms | 939.9 ms | 204.9 ms | 135.8 ms |
| Generation, prompt=128 + 32 new tokens, B=1 | 484.5 ms | 234.3 ms | 158.2 ms | 104.7 ms |

Each case used 3 warmups and 7 measured iterations on synthetic inputs with
matching seeds and initial weights: LTC/dense, Euler, H=embedding=384, L=2,
V=8000, multi-tau and local convolution enabled, dropout=zoneout=0, no FFN or
ChunkMemory. CPU intra-op and inter-op threads were both 1; TF32 was disabled.
Training includes forward, cross entropy, backward and AdamW. Generation
includes prompt processing, sampling and decoding. CUDA timings synchronize
both boundaries. Imports, model construction, input transfer and JIT warmup
are excluded; first-launch compilation still has a real cost.

These are local measurements from separate backend runs. They do not predict
multithreaded CPU performance or another model's throughput. Smaller workloads
can be faster on CPU, and FP16 showed substantial timing variance; AMP did not
automatically improve speed or memory in this configuration. Short synthetic runs
confirm functionality, not answer quality or long-training stability.

### Training activation fusion follow-up

The retained optimization combines the two LTC add/tanh operations and their
first-order backward into Triton kernels, including the combined gradient of
the recurrent projection slices. It preserves
[native CUDA tanh-backward rounding](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BinaryMiscBackwardOpsKernels.cu)
in FP32/FP16/BF16; this is covered by an exact regression against PyTorch.
Model structure, parameter names, checkpoint architecture keys, normalization
and regularization are unchanged. The multi-tau softmax remains in PyTorch: a
trial fusion amplified arithmetic differences in AMP recurrent training.

Full-step A/B measurements compared a saved pre-change implementation with the
final implementation in one process. Both versions used matching initial weights
and inputs, 3 warmups and 12 measured steps each, alternating ABBA. The workload
matches `train_realistic16` above; `train_regularized16` adds the CLI defaults,
dropout=0.05 and zoneout=0.1. CPU intra-op/inter-op threads were 1; TF32 was off.
Timings include forward, CE, backward and AdamW (`foreach=False`), with CUDA
synchronized at both boundaries. FP16 used GradScaler with initial scale 1024.
Seeds were reset outside the timed region to match regularization masks.

| Dropout/zoneout | Precision | Median before, ms | Median after, ms | Median paired speedup |
|---|---|---:|---:|---:|
| 0 / 0 | FP32 | 255.83 | 221.56 | 1.048× |
| 0 / 0 | BF16 | 599.45 | 576.66 | 1.037× |
| 0 / 0 | FP16 | 650.59 | 627.05 | 1.018× |
| 0.05 / 0.1 | FP32 | 609.86 | 606.32 | 1.003× |
| 0.05 / 0.1 | BF16 | 692.19 | 660.65 | 1.054× |
| 0.05 / 0.1 | FP16 | 781.47 | 752.90 | 1.031× |

Paired speedup is `median(before_i / after_i)` over 12 adjacent A/B pairs,
so it differs from the ratio of the two independent medians. Timing shifts
affected even adjacent measurements; regularized FP32 shows no clear gain.
The larger ratio of independent FP32 medians is not a reliable speedup estimate.
All six cases completed 15 optimizer updates per version without skipped
updates or nonfinite values. Recorded losses matched at every measured step;
the relative L2 difference between final parameter arrays was zero in each case.
The full suite (48 tests), 18-case checker and 5 focused CLI scenarios passed
again after the final kernel changes.

Operator profiling used CPU activities because CUPTI initialization returned
`CUPTI_ERROR_INVALID_DEVICE`; the speed measurements use synchronized wall time.
No CUDA C++ extension is included. The test machine had nvcc 13.4, PyTorch built
for CUDA 12.8 and no MSVC C++ workload, so a matching extension build environment
was unavailable. Matrix multiplications continue to use cuBLAS.

## Training-horizon experiment

`research_liquid.py` runs context/tau diagnostics, matched-token continuation
experiments and compatible checkpoint averaging without changing the architecture.
A 64/128-token training experiment used the 2.47M-parameter Shakespeare checkpoint.
Both arms start from identical weights and receive 200 updates of 8,192 tokens,
with seeds 2026 and 2027. Validation uses fixed windows, separately at lengths
64 and 128. Increasing `state_carry` alone does not increase the BPTT horizon.

Results: extending the training horizon to 128 did not improve final EMA
validation loss over the paired 64-token controls on either seed. Additional
training and uniform averaging of the two 64-token EMA checkpoints reduced
the original fixed-window NLL from 3.473696 to 3.459669 (perplexity 32.2557 to
31.8065, about 1.39% lower), with the same 2,467,846 parameters. This is a small
gain on reused validation, with no independent test-set claim. Both averaged
models and all four individual runs were safely reloaded and re-evaluated.
The current full CUDA suite passed 53 tests without skips; native CLI generation
from the averaged checkpoint passed.

Run `python research_liquid.py --steps 0` for diagnostics and
`python research_liquid.py --steps 200 --skip-diagnostics` for the paired study.
These commands require CUDA, the local source checkpoint at
`runs/shakespeare_2m5/best.pt` and its matching corpus in `data_shakespeare/`.
Neither is bundled; use `--checkpoint` and `--data` for other paths. The script
checks the corpus fingerprint against the checkpoint before running.
Keep research reports, weights and caches local. Experimental output checkpoints
contain inference weights only.

## Most useful contributions right now

1. **Quality and speed A/B runs on a real corpus.** Compare `cfc`, `ncp`,
   `--memory-slots`, `--ode-solver heun`, AMP and `--kernel triton` with
   `ltc + euler + dense` using the same data split, tokenizer, seed and training
   budget. Report validation loss, throughput, peak memory and several full
   epochs. Short functional runs already pass; quality and
   long-context error accumulation remain open.
2. **Data formats and FIM.** Add targeted CSV/Parquet and FIM examples, verify
   special-token stopping, and check that tokenizer-training and validation
   boundaries cannot leak validation data. BPE JSONL dialog training has
   already passed the GPU smoke runs.
3. **Resume and compatibility coverage.** Preserve old checkpoints and token
   mappings, and test interrupted versus uninterrupted training, including
   Python/NumPy/PyTorch/CUDA RNG state and early-stopping state. Add tests for
   new failure cases or meaningful numerical invariants.
4. **Other hardware and PyTorch versions.** Record OS, driver, CUDA, Triton and
   compiler versions, actual selected backend, skips and compilation cost.
   Profile the intended model and CPU thread count before optimizing it.

## Development ideas

- BitNet b1.58-style weight quantization (ternary weights and
  quantization-aware training).
- Mixture-of-Experts over the readout or liquid layers.
- Sparse (NCP) input projections; currently the block structure is applied
  only to the recurrent matrix.
- Batched generation.
- Hybrid liquid layers with a small number of attention blocks.

## How to propose a change

1. Open an Issue describing the idea or bug; for larger changes, it is better
   to discuss them in advance.
2. Fork the repository and create a separate branch from `main`.
3. Preserve backward compatibility: new options should be **disabled** by
   default, and architecture parameters must be added to `ARCH_KEYS` /
   `LEGACY_ARCH_DEFAULTS` in `liquid_lm/model.py` so that old checkpoints
   continue to load. Keep model structure and checkpoint/tokenizer semantics
   intact when fixing bugs. Describe proposed architecture changes separately
   before implementing them. The README section "Adding a custom architecture
   option" lists the integration points.
4. In the Pull Request, state what changed, why, and how it was tested
   (command and result). If you claim a speed or quality improvement, include
   measurements.
5. Do not commit checkpoints, datasets, caches or large raw logs (see
   `.gitignore`). Keep temporary runs in `runs/`; retain compact, documented
   benchmark JSON and commands when needed to support a performance claim.

## Bug reports

Include the command used to run the program, the complete traceback, Python /
PyTorch / CUDA versions, and, if possible, a minimal reproducible example.

By submitting a contribution, you agree that it will be distributed under the
project license (Apache License 2.0).
