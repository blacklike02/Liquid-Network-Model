# Contributing

Thank you for your interest in the project! Changes, ideas, and experiment
results are welcome.

## Documentation language

English is the primary language for public documentation, commit messages and
pull request descriptions. Russian translations are welcome; keep their commands
and reported measurements consistent with the English versions.

## Evaluation priorities

Held-out quality takes priority over training or inference time. A slower
configuration is worthwhile when controlled training demonstrates better
language loss, answer accuracy or retention for the intended task. Report
latency and memory as costs, rather than using speed alone to reject it.

Compare architectures with the same data, tokenizer, initialization where
compatible, training updates and multiple seeds. Select checkpoints on
validation data before inspecting a separate final test. Synthetic recall
establishes a narrow memory capability; conversational quality needs its own
language evaluation. RMSNorm is the only supported normalization. Do not
silently load LayerNorm checkpoints as RMSNorm: their recurrent dynamics differ.
New checkpoint configs must explicitly record `norm_type: rms`; old LayerNorm
weights require an earlier revision or a new training run.

An alternative kernel implementing the same equations must preserve numerical
behavior. Making that kernel slower does not itself add learning capacity.

## Project layout

Run all commands from the repository root. The `liquid_lm` package uses a flat
module layout:

- `model.py`: language model, generation, checkpoint I/O and architecture keys.
- `layers.py`: LTC/CfC cells, native RMSNorm, compact/legacy ChunkMemory and the optional Triton path.
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
  regularized optimizer steps with RMSNorm/checkpointing.
- `tests/check_triton_mixed.py`: standalone 18-case Triton correctness checker
  for mixed dtypes and full-model gradients; run separately from test discovery.
- `benchmark_model.py`: CPU training/backward, generation and KD microbenchmarks.
- `benchmark_gpu.py`: synchronized CPU/CUDA training and generation comparisons.
- `benchmark_compute.py`: adjacent alternating pairs against the previous
  output/input-projection schedule, plus an identical-code timing control.
- `tests/test_compute_schedule.py`: schedule equivalence, RNG/optimizer/carry,
  fullgraph capture, checkpointing and all-precision gradients.
- `benchmark_triton_tail.py`: isolated LTC-tail forward/backward comparison.
- `benchmark_retained_features.py`: native normalization, compact memory and
  a paired delayed-recall learning comparison.
- `tests/test_retained_features.py`: occupied memory, gradients, precision,
  bounded storage and retained/retired configuration behavior.
- `research_liquid.py`: context/tau diagnostics, matched-token training-horizon
  experiments and compatible checkpoint averaging.
- `liquid_lm/associative.py`: experimental innovation-gated matrix memory.
- `benchmark_innovation.py`: on-the-fly associative recall with shared-weight,
  matched-data baseline and FFN parameter controls.
- `tests/test_innovation_memory.py`: equations, causality, state carry,
  checkpoint compatibility, gradients and CUDA/Triton AMP checks.
- `benchmark_memory_validation.py`: ordinary-delta/scalar-decay ablations,
  overwrite/multi-query recall and equal-time training controls.
- `benchmark_memory_language.py`: train-only vocabulary, held-out language NLL.
- `tests/test_memory_benchmarks.py`, `tests/test_memory_language_benchmark.py`:
  benchmark labels, control initialization and split boundaries.
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

### Retained feature revision

Native RMSNorm and compact chunk memory replace the previous experimental
branches. The compact memory uses low-rank projections and explicit occupancy;
empty reads are exact identities. At H384/D64 it adds 106,624 parameters versus
738,432 for the legacy full-width layout. Supported RMSNorm configs lacking
`memory_dim` restore the legacy memory layout. Normalization is fixed to RMSNorm.

NCP, FIM rearrangement, Heun and extra ODE unfolds have been removed. Supported
RMSNorm baseline checkpoints retain their equations and token IDs. LayerNorm,
missing normalization metadata, unsupported retired equations and FIM resume
recipes fail explicitly. Historical results below describe earlier code,
including those options.

Reproduce the focused comparison:

```powershell
python benchmark_retained_features.py --device cuda
```

The script compares normalization kernels, compact/legacy memory, full-model
forward/backward spans and paired delayed recall. Recall uses three seeds,
512 held-out examples per seed, 12.5% chance, identical shared-core weights and
training batches, detached boundaries and a shuffled-memory control. Parameter
and time budgets differ; this is not a conversational benchmark. Results go to
ignored `runs/`.

### RMSNorm-only validation: 20 October 2026

RMSNorm now normalizes every cell, FFN and final readout. LayerNorm construction
and the `--norm-type` CLI selector have been removed. The Python `norm_type`
keyword accepts only `rms`, and new checkpoints always record that value from
the model. Resume and generation reject LayerNorm or missing normalization
metadata explicitly; existing weight files are preserved and require an earlier
revision to run. Metadata edits do not convert weights.

The updated suite passed **118/118 on CUDA/Triton** (38.803 s, PyTorch
2.11.0+cu128, RTX 5050 Laptop). CPU-only passed **98**, with **20 expected skips**
(118 discovered, 18.953 s, PyTorch 2.14.1+cpu). Coverage includes AMP precision,
RMSNorm gradients, recurrent/streaming equivalence, checkpointed backward,
normalization metadata, and rejection of LayerNorm state before model mutation.

Six native BPE CLI scenarios also passed: train, resume and generation on
CPU/Torch and CUDA BF16/Triton, using compact memory and activation checkpointing.
Current RMSNorm FP32 timing on the RTX 5050 Laptop was **3405.25 ms without
memory** and **3603.24 ms with compact memory** for four B8/T128/H384/L2/V4096
chunks and FFN1. Six warm samples per case alternated forward/reverse with
sequence batching and Triton LTC. Optimizer, loading and compilation are
excluded; clocks were unlocked. These are current timings, not a cross-session
speedup or a language-quality result.

<!-- retained-results:start -->
### Historical retained feature results: 19 October 2026

After removing NCP, FIM, Heun and extra ODE unfolds, the final suite passed
**114/114 on CUDA/Triton**. CPU-only passed **94**, with **20 expected skips**.
Six native CLI scenarios passed: BPE train, resume and generation on CPU and
CUDA BF16/Triton. Both retained user checkpoints preserved their file hashes
and exactly matched their recorded CPU logits and carried states.

Compact ChunkMemory uses **106,624 parameters at H384/D64**, compared with
738,432 in the legacy module. It masks unwritten slots and uses a bounded FIFO.
At that revision, training used LayerNorm, three memory slots, width 64 and
four carried chunks; saved architecture settings were restored on resume.
RMSNorm was optional then. The current revision uses RMSNorm exclusively and
rejects those LayerNorm checkpoints. A slower experimental Triton RMSNorm
implementation was removed after measurement.

On a paired three-seed delayed-recall task (600 updates, 512 fresh examples
per seed), mean accuracy was **31.58% without memory** and **75.20%
with compact memory**; shuffling the same trained models' history gave
**23.76%**. Both arms used LayerNorm and identical common-core weights
and batches. Parameter/time budgets differ; this is synthetic recall, not
conversational accuracy or a general language-model benchmark.

Warm RTX 5050 Laptop measurements at B8/T128/H384/L2/V4096, FFN1, four chunks,
zero dropout/zoneout, sequence batching and the Triton LTC backend took median
**3038.31 ms without memory** and **3055.12 ms with compact
memory**, using LayerNorm in both cases. Six samples per configuration alternated
forward/reverse; loading, compilation and optimizer work are excluded. Clocks
were unlocked. Smaller memory projections do not guarantee faster isolated reads.

These measurements describe the pre-RMS-only revision. The current
`python benchmark_retained_features.py --device cuda` evaluates RMSNorm only;
`--skip-learning` runs timing only. It does not reproduce the LayerNorm numbers
above. The earlier 4.54M dialogue weights remain unchanged and require an
earlier revision to run; improved free conversation has not been established.

<!-- retained-results:end -->

### Verified results: 19 October 2026

The reference configuration used Windows 11, Python 3.12.10, NVIDIA GeForce RTX 5050
Laptop GPU (8 GB, 8151 MiB reported), driver 616.56 and PyTorch 2.11.0+cu128.
The CPU-only environment used PyTorch 2.14.1+cpu.

| Validation | Result |
|---|---|
| Earlier feature set, CUDA with Triton available | 101 passed, no skips, in 69.585 s |
| Earlier feature set, CPU-only environment | 82 passed, 19 expected skips; 101 discovered, in 14.222 s |
| Initial GPU CLI audit: training, resume, generation, KD and AMP | All 13 passed |
| Initial compiled CLI training and generation | Both passed |
| Activation-fusion follow-up: training, resume, BF16 NCP/memory/checkpointing, generation, selftest | All 5 passed |
| Triton numerical comparisons, including mixed dtypes and full-model backward | All 18 passed |

After adding opt-in sequence batching, **101 tests passed without skips in
69.585 s** with CUDA/Triton; the CPU-only suite passed **82 with 19 expected
skips in 14.222 s** (101 discovered). Eight new tests cover FP64/FP32/AMP
outputs and gradients, all carried states, causality, last-token prefill,
ChunkMemory/associative memory, stochastic RNG preservation, compiled-step
fallback and checkpoint state-key compatibility.

`--sequence-batching` batches independent upper-layer input projections and
FFNs over time. Recurrence remains sequential. Default execution stays off;
active cell/FFN dropout, zoneout and compiled steps retain their existing
schedule. Pass the runtime flag again when resuming or generating. AMP gradient
rounding differs because weight-gradient reductions are grouped differently.

Reproduce paired measurements with:

```bash
python benchmark_sequence_batching.py --kernel triton --precisions fp32 bf16 fp16 --pairs 9 --warmup 4
python benchmark_sequence_batching.py --kernel triton --precisions fp32 --identical-control --output runs/sequence_batching/control.json
```

H384/L2/V1024, B16/T64, AdamW, FP32 master weights, TF32 off; generation uses
prompt128+32 greedy tokens. Median paired training speedups were
**1.094/1.178/1.185×** (FP32/BF16/FP16) for the 2.47M base model and
**1.696/1.719/1.703×** for the same paired 3.35M architecture with `ffn_mult=1`.
Generation measured **1.104/1.130/1.117×**. All training comparisons used zero
dropout/zoneout. FP32 identical controls measured **0.976/1.001/0.991×** for
base/FFN/generation; the base FP32 result won 7/9 pairs, other training cases
9/9. Clock/power settings were not fixed, so compare adjacent pairs rather
than absolute times between cases. These measurements exclude data loading,
validation, saving and any claim of language-quality improvement. Prefill peak
temporary CUDA memory rose by up to 0.73 MiB; training peaks did not increase
in the tested configurations.

Earlier, after the memory audit, the full suite passed **93 tests in 35.442 s** with
CUDA/Triton; the CPU-only environment passed **75 with 18 skips in 6.711 s**
(93 discovered). Nine new regressions check eager model destruction without
cyclic GC, carried convolution storage and cross-chunk gradients, bounded
sampling-buffer caching, inference checkpoint ownership, dataset temporary
lifetime, logger cleanup after failure and CPU-to-CUDA EMA restoration.

The existing 2.47M checkpoint consumed **51,735,552 → 9,872,384 bytes** of live
CUDA allocation above baseline at generation entry, with generation mocked out
to isolate loading. Model parameters occupy 9,871,384 bytes; allocation rounding
accounts for the small difference. Loading on CPU and discarding training state
and copied weights avoids about 81% of this setup allocation. The B2/T128/E64
convolution tail retained **67,584 → 2,048 bytes** without changing outputs or
cross-chunk gradients. With cyclic GC disabled, 20 eager models previously
remained alive (5,150,720 bytes) until GC; all now release immediately.

Finite stress runs used H64/L2/B2, two T16 subchunks, EMA, 15 training warmups,
80 measured updates, 12 validation calls and 10 generation calls per case.
All sampled live/reserved CUDA allocations were constant within each phase for
LTC/torch FP32, CfC/BF16 with both memory modules and FFN, and LTC/Triton BF16
with NCP, ChunkMemory and gradient checkpointing. CPU runs covered the first
two configurations in FP32. Windows private-memory fluctuations and CUDA's
reserved allocator cache are recorded separately; these measurements do not
prove every workload is leak-free or measure speed improvements.

Three additional temporary CUDA BF16/Triton CLI scenarios passed with NCP,
ChunkMemory, associative memory, EMA and activation checkpointing: fresh
training, resume from a CPU-loaded checkpoint (EMA updates restored from 2
to 4), and top-k distillation from a teacher checkpoint. Data and temporary
checkpoints were removed; the original trained checkpoint was preserved.

Earlier, after the dataset-loading audit, the full suite passed **84 tests in 41.872 s**
with CUDA/Triton, and **67 passed, 17 skipped in 5.807 s** in the CPU-only
environment (84 discovered). The skips are 16 GPU/Triton checks and the optional
real-BPE test. Twelve dataset regressions cover JSON/JSONL extraction and BOM,
strict JSONL UTF-8, legacy encoding fallback, directory pruning/deduplication,
uneven token density, batch isolation and saved split-policy restoration.

A CUDA BF16 smoke trained a temporary 7,151-parameter BPE model and resumed it:
the 2,074/238-token train/validation split and saved fraction 0.1 were preserved
when resume requested 0.3. Temporary data and checkpoints were removed. This is
a functional check, not a language-quality result.

Against HEAD `c4c68b9`, a temporary 32-file, 4,224,374-byte UTF-8 fixture with
1,000 files under an excluded tree loaded in median **175.18 → 124.11 ms**;
median paired speedup was **1.43×**. Single-file UTF-8 reading was
**0.770 → 0.259 ms**, paired speedup **2.99×**. Seven alternating pairs followed
one warmup per implementation, with warm OS cache. Corpus SHA-256 matched in
every pair. These measurements cover loading, not tokenizer fitting, peak RAM,
cold-storage I/O or training throughput. No persistent dataset was added.

Earlier compute scheduling passed 72 tests in 59.129 s with CUDA/Triton and 56 with
16 expected skips in 12.600 s on CPU. The real 2.47M BPE checkpoint retained
its hash/state keys and produced identical greedy 64-token completions for
three prompts against the saved pre-change code. Native Inductor inference
matched eager execution; compiled FP32/BF16 backward produced finite gradients.

Earlier extended memory validation passed all 67 tests in 17.721 s with CUDA/Triton,
and 52 with 15 expected skips in 3.152 s in the CPU-only environment. This
includes document/batch isolation, a 1,024-token finite streaming state,
overwrite/multi-query benchmark labels and held-out language split checks.
Three additional CUDA CLI scenarios passed: BF16/Triton training with NCP,
ChunkMemory, Muon and gradient checkpointing; resume restoring the memory
dimension; and generation from the saved checkpoint. The original 2.47M
checkpoint still loads with the module disabled and its SHA-256 is unchanged.

The earlier pre-commit rerun after removing local datasets passed all 53 tests in
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

## Small English training experiment

The 19 October 2026 local run trained a 4,536,838-parameter model from scratch
on the same RTX 5050 Laptop: LTC Euler, H384/L2, BPE4096, tied embeddings,
multi-tau, local convolution, `ffn_mult=1`, no associative or ChunkMemory.
Execution used Triton and sequence batching, FP32 with TF32, T128/carry2,
AdamW, clipping 1, EMA 0.999, dropout/zoneout 0. Pretraining used LR 0.002;
response tuning used 0.0004 with fresh optimizer/scheduler/EMA state.

Training sources were 50,000 TinyStories GPT-4 v2 stories, 2,065 Everyday
Conversations pairs and 24,000 authored curriculum examples. Story validation
and test each held out 1,000 records; conversation prompts were grouped by
SHA-256 before splitting; curriculum parameter-combination IDs were disjoint
across 24,000/1,000/1,000 training/validation/test examples. BPE fitting used
training text only. Exact deduplication cannot rule out similar story plots.
Source links and licenses appear in README; pinned revisions and corpus hashes
are retained in the local data manifest.

Pretraining completed 3,600 updates, with batch 64 for updates 1–700 and 256
thereafter. Response tuning saved 1,100 updates before a deliberate early stop:
training loss fell while validation NLL and story loss deteriorated. A final
validation-only comparison used 200 balanced questions and 64 stories. The
early 100-update checkpoint scored 88%, versus 91% for the final raw weights;
the latter exceeded the predeclared 1.5× story-perplexity ceiling. The early
checkpoint was selected before the final test was opened.

| Final test, same tokenizer and examples | Pretraining | Selected response model |
|---|---:|---:|
| Curriculum exact match | 166/200 (83%) | 178/200 (89%) |
| Arithmetic exact match | 1/28 | 8/28 |
| Story answer NLL / perplexity | 1.896394 / 6.662 | 1.980482 / 7.246 |

The story protocol covers 256 held-out stories, 52,490 answer targets and
non-overlapping 256-token windows with state reset per window. Curriculum
sampling is balanced across seven task templates with unseen combinations;
only case, whitespace and punctuation are normalized. Twelve held-out
conversation responses, 35 factual paraphrases and fixed demonstrations are
retained as qualitative outputs, not additional accuracy scores. One factual
paraphrase also appears in fixed progress demonstrations; factual probes are
not an independent accuracy benchmark. Free dialogue
remains unreliable, and this is not a general reasoning benchmark.

Both compact exports passed safe loading, finite-weight checks and exact
tensor/tokenizer round trips. Their tokenizer configurations and model state
keys match. The native one-shot chat CLI also passed on CUDA/Triton and CPU/torch
after removing the training datasets. The selected export is 18,272,865 bytes. A synchronized native
generation benchmark (B1, 29 prompt + 64 forced greedy tokens, two warmups,
five repeats, one CPU thread, EOS stopping disabled, loading excluded) measured
median **0.3865 s / 165.6 new tokens/s** on CUDA/Triton and
**0.5013 s / 127.7 new tokens/s** on CPU/torch. GPU clocks/power were unlocked.
Warm CUDA live allocation was unchanged before/after the five repeats:
27,716,608 bytes, with a 32,957,440-byte allocated peak; this excludes driver
and other-process memory. Artifacts and detailed outputs remain local under
the ignored `runs/english_small_4m/` directory.

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
At that revision, the full CUDA suite passed 72 tests without skips; native CLI generation
from the averaged checkpoint passed.

Those results and the script's default Shakespeare checkpoint are historical:
that checkpoint uses LayerNorm and requires the earlier compatible revision.
Its original corpus is not bundled. For a new experiment on the current code,
provide a supported RMSNorm checkpoint without ChunkMemory and its matching
plain-text corpus explicitly:

```powershell
python research_liquid.py --checkpoint path/to/rms_model.pt --data path/to/matching_corpus --steps 0
python research_liquid.py --checkpoint path/to/rms_model.pt --data path/to/matching_corpus --steps 200 --skip-diagnostics
```

These commands require CUDA. The script checks the corpus fingerprint against
the checkpoint before running. New RMSNorm runs do not reproduce the historical
LayerNorm measurements above.
Keep research reports, weights and caches local. Experimental output checkpoints
contain inference weights only.

## Most useful contributions right now

The optional `--innovation-memory-dim` experiment has recall, ablation and
small held-out language checks. Run the original synthetic recall control with:

```bash
python benchmark_innovation.py --device cpu --steps 800 --seeds 11 22 33
```

The benchmark generates examples in memory and saves only compact results
under ignored `runs/`; it does not download or retain a dataset. Synthetic
recall improvements alone do not establish better language generation.

The measured run used one LTC layer, H=64, D=16, B=64, 800 AdamW updates,
LR=0.003, four random key/value pairs and eight distractors. Seeds 11/22/33
each saw the same training examples across arms and 512 fixed held-out examples
per delay. Final weights were evaluated without checkpoint selection.

| Model | Parameters | Accuracy, delay 8 | Delay 32 | Delay 64 |
|---|---:|---:|---:|---:|
| LTC | 32,798 | 23.70% | 15.56% | 12.11% |
| LTC + FFN control | 37,534 | 21.68% | 16.73% | 15.49% |
| LTC + innovation memory | 38,145 | 36.85% | 32.94% | 27.93% |

Values are means across three seeds; chance among the eight values is 12.5%.
At delay 8 the memory arm's per-seed accuracies were 42.58%, 33.79%, 34.18%.
Longer-delay variance remains large; at delay 64 seed 22 slightly lost to FFN.
This is an equal-update comparison. A 30-step CUDA/Triton
cost check measured median full steps of 37.978/42.574/128.070 ms for these
arms. Fixed ordering, few measurements and Windows activity limit timing claims.
The memory uses reference PyTorch operations, adding launch overhead.

The extended 800-update recall comparison reproduced those first three arms.
Ordinary delta writes (without the error-energy multiplier) scored 43.49% at
delay 8 and 37.17% at delay 256; innovation memory scored 36.85% and 20.96%.
On overwritten keys, innovation memory returned the new value 25.65% of the
time and the obsolete value 30.73%; ordinary delta scored 42.25% and 28.19%.
These stress examples were not part of training. The error-energy gate is not
validated as an improvement over the simpler write rule under this protocol.

An additional CUDA/Triton equal-time check allowed 30 measured training seconds
per arm and seed, excluding data preparation, warmup and evaluation. Mean recall
at delay 8 was 23.31% (LTC), 15.04% (FFN), 18.29% (innovation) and 29.04% (delta).
Substantial transient throughput variation, especially on seed 22, prevents a
strict ranking from this small run. The feature remains experimental.

The follow-up language screen trained fresh one-layer H=64 character models on
[Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11), with
train-only vocabulary and contiguous 80/10/10 splits. At B=16, T=64, 300 updates
and seeds 11/22/33, final mean test NLL was **1.9575 (LTC)**, 1.9903 (FFN),
1.9885 (innovation memory) and 1.9843 (ordinary delta writes). Memory lost to LTC
on every seed. No checkpoint selection used validation or test. Evaluation
scored 14,400 nonoverlapping test targets and reset state every 64 characters.
This small budget does not establish a language benefit for the extra state;
the feature remains disabled by default.

`benchmark_memory_validation.py` adds exact ordinary-delta and scalar-decay
ablations, overwrite/multi-query tasks, delays through 256 and a `--seconds`
training budget. `benchmark_memory_language.py --data PATH` runs the language
comparison without downloading data or retaining trained checkpoints. Both
save compact results in ignored `runs/`. The temporary evaluation corpus was
removed after testing; no new trained checkpoint is retained.
The full GPU benchmark accepts `--innovation-memory-dim 32`; all 24 FP32/BF16/FP16
training/generation cases passed in alternating D=0/32/32/0 rounds. Pooled FP32
median training time was 212.75 → 518.12 ms and generation 171.03 → 364.15 ms
for H=384, L=2, V=8000, B16/T64 training and prompt128+32-token generation.

1. **Quality and speed A/B runs on a real corpus.** Compare `cfc`, RMSNorm,
   compact `--memory-slots`, AMP and `--kernel triton` with
   the dense LTC baseline using the same data split, tokenizer, seed and training
   budget. Report validation loss, throughput, peak memory and several full
   epochs. Short functional runs already pass; quality and
   long-context error accumulation remain open.
2. **Data formats and context.** Add targeted CSV/Parquet and multi-turn examples, verify
   special-token stopping, and add record-level train/validation separation.
   New BPE runs now align tokenizer fitting and validation at a character
   boundary; legacy checkpoint resumes retain their token-count split.
   BPE JSONL dialog training has already passed the GPU smoke runs.
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
- Batched generation.
- Hybrid liquid layers with a small number of attention blocks.

## How to propose a change

1. Open an Issue describing the idea or bug; for larger changes, it is better
   to discuss them in advance.
2. Fork the repository and create a separate branch from `main`.
3. Preserve the semantics of supported checkpoints: new experimental options
   should be **disabled** by default, and architecture parameters must be added
   to `ARCH_KEYS` / `LEGACY_ARCH_DEFAULTS` in `liquid_lm/model.py` where applicable.
   Missing fields must never reinterpret trained equations. Intentionally
   retired formats, including LayerNorm, must fail clearly and be documented
   with the compatible earlier revision. Keep model and tokenizer semantics
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
