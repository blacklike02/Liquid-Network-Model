# Echo — a language model based on liquid neural networks (LTC / CfC)

[English](README.md) | [Русский](README.ru.md)

**Echo** is the first model in the planned **Pantheon** family. The
[naming plan](NAMING.md) records the future progression and version conventions.

An autoregressive language model implemented with **Liquid Neural
Networks** (Liquid Time-Constant and Closed-form Continuous-time) in PyTorch.
The code lives in the `liquid_lm` package: corpus loading and cleaning,
tokenizers, the model, training, generation, chat, and supporting utilities.
Everything is launched through the thin entry point `liquid_text_model.py`
(or `python -m liquid_lm`), so all commands in this document work from the
repository root.

Instead of self-attention, the model processes text recurrently: each layer's
hidden state evolves as a dynamical system with learnable time constants. This
provides fixed-size memory and generation cost linear in sequence length.

> Current version: **v3.7**. The change history is at the end of this document.

## Contents

1. [Features](#features)
2. [Requirements and installation](#requirements-and-installation)
3. [Quick start](#quick-start)
4. [Data preparation](#data-preparation)
5. [Tokenizers](#tokenizers)
6. [Model architecture](#model-architecture)
7. [How training works](#how-training-works)
8. [Generation and chat](#generation-and-chat)
9. [Triton kernel](#triton-kernel-experimental)
10. [CLI parameter reference](#cli-parameter-reference)
11. [Ready-to-use recipes](#ready-to-use-recipes)
12. [Files created by the program](#files-created-by-the-program)
13. [Checkpoints, resume, and compatibility](#checkpoints-resume-and-compatibility)
14. [Hyperparameter selection and GPU memory](#hyperparameter-selection-and-gpu-memory)
15. [Known limitations](#known-limitations)
16. [Troubleshooting](#troubleshooting)
17. [Code structure](#code-structure)
18. [Tests and measured results](#tests-and-measured-results)
19. [Change history](#change-history)
20. [Sources and inspiration](#sources-and-inspiration)

## Features

**Model**

- Two cell types: **LTC-inspired** exponential ODE updates and **CfC-style**
  closed-form interpolation, based on Hasani et al., 2022.
- **Multi-tau**: fast, medium, and slow time-constant banks mixed by a learned
  softmax.
- One dense recurrent exponential update per token, with the optional Triton backend.
- A causal depthwise convolution (`k=5`) before the liquid core, carrying
  context across chunks and generation steps.
- Residual connections, native RMSNorm on cell output, zoneout, and locked
  embedding dropout.
- Optional **weight tying** between the embedding and output layer, and an
  optional MLP head.
- **Compact ChunkMemory**: occupied FIFO history, low-rank projections and gated
  retrieval at segment boundaries. New training uses three slots and width 64.

**Data**

- Formats: `.txt .md .rst .log .csv .tsv .json .jsonl .parquet`.
- Automatic encoding detection (`charset-normalizer`, with fallback through
  `utf-8 → cp1251 → cp1252 → latin-1`).
- Automatic conversion of dialogue datasets (`messages`, `conversations`,
  `instruction/output`, `question/answer`, and others) to a common role-tagged
  format.
- Content-hash file deduplication, deterministic file shuffling, and file-size
  limits.
- Byte-level BPE (`tokenizers`) and character tokenizers.

**Training**

- Truncated BPTT with state carry between sub-chunks (`--state-carry`).
- Gradient accumulation and clipping, with cosine scheduling and warmup.
- AdamW (fused on CUDA) and the Muon + AdamW hybrid.
- EMA weights and automatic selection of the better regular or EMA weights on
  validation.
- Mixed precision (bf16 / fp16 with GradScaler), `torch.compile` for the
  recurrent step, and gradient checkpointing.
- Loss masking: for dialogues, only the bot response and end-of-example marker
  contribute to loss.
- Dialogue anchors: some batch windows start exactly at `<|user|>`.
- Knowledge distillation (KD) from a teacher checkpoint.
- Optional TensorBoard / Weights & Biases logging.
- Resume with optimizer, scheduler, EMA, and scaler restoration.

**Inference**

- Temperature, top-k, top-p, and repetition-penalty sampling.
- One-shot chat requests and an interactive REPL with history.
- Optional fused Triton kernels for LTC activations and the sub-step tail, with inference and
  experimental first-order training backward on CUDA.

## Requirements and installation

**Required**

- Python **3.9+**
- `torch` (2.x recommended; `torch.compile` requires 2.0+)
- `numpy`

**Optional**

| Package | Purpose |
|---|---|
| `tokenizers` | BPE tokenizer (without it, the program falls back to char) |
| `pandas` + `pyarrow` | Read `.parquet`, `.csv`, and `.tsv` as tables |
| `charset-normalizer` | Accurate text-file encoding detection |
| `tensorboard` | Logging with `--log-backend tensorboard` |
| `wandb` | Logging with `--log-backend wandb` (`wandb login` required) |
| `triton` | Fused LTC inference/training kernel; a compatible `triton-windows` build was also tested on Windows |

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install PyTorch according to the instructions at pytorch.org for your CUDA version
pip install torch numpy

# Recommended minimum for actual work
pip install tokenizers pandas pyarrow charset-normalizer

# Optional
pip install tensorboard wandb
```

Check that the GPU is visible:

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
```

Use the same interpreter for training and tests, with a CUDA-enabled PyTorch
build for GPU execution. A CPU-only build cannot use CUDA even when the driver
detects the GPU. The CLI prints the device, GPU, VRAM and Tensor Core settings
at startup.

## Quick start

Datasets and trained checkpoints are not bundled with the repository. Prepare
your corpus before training.

1. Put text files in `data/` (subdirectories are supported):

   ```
   data/
   ├── books/
   │   ├── book1.txt
   │   └── book2.txt
   └── dialogs.jsonl
   ```

2. Start training with the defaults:

   ```bash
   python liquid_text_model.py --data data
   ```

3. Generate text with the best model:

   ```bash
   python liquid_text_model.py --generate-only --prompt "Once" --generate 500
   ```

4. Chat with the model (if it was trained on dialogues):

   ```bash
   python liquid_text_model.py --chat-repl
   ```

> The minimum corpus size is **1000 characters** (`--min-chars`). Meaningful
> generation requires megabytes of text; a corpus of a few kilobytes is useful
> only as a smoke test.

## Data preparation

### File discovery

The program recursively scans `--data` and selects files with supported
extensions. It ignores `.git`, `__pycache__`, `node_modules`, `.venv`, `venv`,
`.idea`, `.vscode`, `.mypy_cache`, and `.pytest_cache`.

If the directory does not exist, it is created and the program reports that the
corpus is empty.

### Processing each file

1. Skip files larger than `--max-file-mb` (when > 0) and empty files.
2. Read the file according to its format.
3. Clean the text: normalize line endings (`\r\n`, `\r` → `\n`), trim trailing
   whitespace, and collapse 3+ blank lines into one blank line.
4. Deduplicate files with identical cleaned content.
5. Join the corpus. With `--file-headers`, insert
   `===== filename =====` between files (disabled by default).

The final statistics include files found and used, size by extension, skipped
files (too large, empty, or duplicate), and files read with uncertain encoding.

### Supported formats

| Extension | Reading behavior |
|---|---|
| `.txt .md .rst .log` | UTF-8 first (optional BOM removed), then encoding detection/fallback |
| `.json` | A list or object; dialogue structures become tagged text, otherwise text is extracted from `text/content/article/body/document` |
| `.jsonl` | One JSON record per line; the same rules apply, and unrecognized lines are appended as-is |
| `.csv .tsv` | Through pandas; question–answer columns become dialogue, otherwise a text column is used |
| `.parquet` | Same as tables (requires `pandas` + `pyarrow`) |

Without `pandas`, CSV/TSV files are read as ordinary text and Parquet files
are skipped.

JSON and JSONL accept a UTF-8 BOM. Invalid UTF-8 in JSONL raises an error
instead of silently deleting bytes; convert such files to UTF-8 first.
Excluded directories are pruned before traversal.

Training keeps int32 token data resident on the selected device. Source text
and temporary int64 IDs are released after preparation; this is still an
in-memory pipeline, so very large corpora require sufficient RAM/VRAM.

### Dialogue format

Dialogues are normalized to four special tags:

| Tag | Meaning |
|---|---|
| `<\|user\|>` | User utterance |
| `<\|bot\|>` | Bot response |
| `<\|system\|>` | System instruction |
| `<\|end\|>` | End of example |

Example:

```
<|user|>
What is a liquid neural network?
<|bot|>
It is a recurrent network whose state is described by a differential equation.
<|end|>
```

Recognized input structures:

**Message list** (OpenAI-style datasets):

```json
{"messages": [
  {"role": "system", "content": "Answer briefly."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hello!"}
]}
```

**`conversations` format** (ShareGPT-style):

```json
{"conversations": [
  {"from": "human", "value": "Hello"},
  {"from": "gpt", "value": "Hello!"}
]}
```

**Question–answer pairs** are checked in this order:

| Question key | Answer key |
|---|---|
| `instruction` | `output` |
| `instruction` | `response` |
| `question` | `answer` |
| `prompt` | `response` |
| `prompt` | `completion` |
| `input` | `output` |

When a non-empty adjacent `input` field is present (Alpaca format), it is
appended to the question on a new line.

Role mapping: `user`, `human` → `<|user|>`; `assistant`, `gpt`, `bot`, `model`
→ `<|bot|>`; `system` → `<|system|>`. An unknown role is treated as a user.

### Train / validation split

- For new BPE runs, `--val-split` selects the validation **text tail by character
  count** (default `0.1`, allowed `0.01–0.5`). BPE is fitted only on the training
  prefix; the two parts are encoded separately. Their token fractions can differ.
- Char runs retain a token-count split and a vocabulary built from all characters.
- To avoid a tail consisting systematically of the alphabetically last files,
  `--shuffle-files` is enabled by default. Files are ordered by SHA-256 of their
  relative path, deterministically.
- Both splits must be longer than `seq_len × state_carry + 2` tokens.
- The split policy is saved in checkpoint config (`data_split_policy`). Resuming
  old checkpoints without this field retains the original token-count boundary.
  The split can still cut a document or dialogue; record-level separation and
  near-duplicate removal are not implemented.

## Tokenizers

### BPE (`--tokenizer bpe`, default)

- Byte-level BPE: every Unicode character can be encoded without `<unk>` (all
  256 bytes are in the vocabulary).
- Prefix whitespace is disabled: text is encoded and decoded without added
  spaces.
- Special tokens: `<unk>`, `<pad>`, `<|user|>`, `<|bot|>`, `<|system|>`,
  `<|end|>`.
- Vocabulary size is controlled by `--vocab-size` (default 8000, minimum 300).
  For small corpora it is automatically reduced: the cap is
  `max(300, estimated_tokens / 20)`, where `estimated_tokens = characters / 4`.
- Large corpora are encoded in chunks of about 2 million characters, split at
  `\n\n` boundaries to limit memory use.
- Without the `tokenizers` library, the program falls back to char and reports
  this.

### Char (`--tokenizer char`)

- The vocabulary contains every distinct character in the entire corpus, not
  only the training split.
- Unknown inference characters are replaced with a space.
- Dialogue tags are not separate tokens, so loss masking, dialogue anchors, and
  stop tokens **do not work**. Use BPE for dialogue.

The tokenizer is stored inside the checkpoint; no separate vocabulary files are
needed.

## Model architecture

### Overall structure

```
tokens
  │
  ▼
Embedding ──► LockedDropout
  │
  ▼
[optional] causal depthwise convolution k=5  (x = x + conv(x))
  │
  ▼
┌──────────────── for each step t ───────────────────┐
│  layer 1: LiquidCell(h1, x_t) ──► h1'              │
│  x = x + h1'     (residual, if enabled)            │
│  layer 2: LiquidCell(h2, x)  ──► h2'               │
│  x = x + h2'                                       │
│  ...                                               │
│  final norm(x)                                │
└────────────────────────────────────────────────────┘
  │
  ▼
Readout (Linear or MLP head; weights may be tied to Embedding)
  │
  ▼
logits (vocab_size)
```

Time is processed in a Python loop: each layer's state is updated one token at
a time. Projections depending only on `x` (not on `h`) are computed once for
the whole sequence before the loop.

### LiquidCell — `ltc` mode (default)

The cell models LTC-inspired dynamics with an input/state-dependent time constant:

```
dh/dt = (candidate − h) / τ
candidate = tanh(W_x·x + W_h·h)
τ         = softplus(MLP([x, h])) + min_tau        (min_tau = 0.05)
```

Within a sub-step, `candidate` and `τ` are frozen and the ODE is solved
exactly:

```
decay = exp(−dt / τ)
h_new = decay · h + (1 − decay) · candidate
```

For `dt >= 0` and `τ > 0`, this frozen update is a convex combination of the
old state and candidate. This property alone does not establish stability of
the full nonlinear recurrence with normalization and residual connections.

The candidate and time constant are evaluated once per token. The frozen
exponential update is the supported LTC integrator; extra integration substeps
and the previous predictor/corrector option have been removed.

**Multi-tau (`--multi-tau`, enabled by default).** The `tau_net` output is
tripled into three banks:

| Bank | Formula | Meaning |
|---|---|---|
| fast | `softplus(r₀ − 2) + 0.05` | Fast dynamics |
| medium | `softplus(r₁) + 0.01` | Medium dynamics |
| slow | `softplus(r₂ + 2) + 0.001` | Larger initial offset |

The final `τ = Σ wᵢ · τᵢ`, where `w = softmax(tau_mix)` contains three learned
global (not per-channel) weights. The banks produce one effective time constant
per channel and share one recurrent state. They are not independent memory
states; their ordering and long memory are not guaranteed after training.

### LiquidCell — `cfc` mode

Interpolation inspired by Hasani et al., *Closed-form Continuous-time Neural
Networks* (Nature Machine Intelligence, 2022). The author's
[reference implementation](https://github.com/raminmh/CfC/blob/main/torch_cfc.py)
uses a shared nonlinear backbone before its candidate/time heads. This project
uses direct projections from `[x, h]`, so it is a simplified variant. There is
no numerical ODE solver in this branch.

```
ff1 = tanh(W₁ₓ·x + W₁ₕ·h)
ff2 = tanh(W₂ₓ·x + W₂ₕ·h)
t   = sigmoid(t_a · dt + t_b)           # t_a, t_b are functions of [x, h]
h_new = ff1 · (1 − t) + t · ff2
```

This has less inductive bias toward decay and more freedom in trajectory shape.
`--multi-tau` is ignored in this mode (there is no tau or ODE). `time_net.bias` starts at zero; the gate is `t = 0.5` when its total
preactivation is zero, rather than being fixed at 0.5 for every initial input.

### Cell output and state normalization

The new layer state is `RMSNorm(Dropout(h))`. RMSNorm is the only supported
normalization in cells, FFNs and the final readout; it has a learned scale and
no bias or mean subtraction. FP16/BF16 normalization uses FP32 statistics.
The state carried between time steps is **normalized**, so the full recurrence
also depends on this transformation. There is no `--norm-type` CLI option.

New checkpoints explicitly store `norm_type: rms`. LayerNorm checkpoints and
older checkpoints without normalization metadata are rejected for both resume
and generation. Use an earlier revision to run those weights, or train a new
RMSNorm model. Editing metadata or discarding biases does not convert them.

Feature selection prioritizes held-out quality over runtime. More expensive
configurations are worth retaining when controlled training demonstrates a
quality improvement.

**Zoneout (`--zoneout`).** During training, some output elements are replaced
with the normalized previous state, regularizing abrupt state changes.

### Local causal convolution (`--local-conv`)

A depthwise `Conv1d` with kernel 5 adds local context from the four preceding
tokens to embeddings: `x = x + conv(x)`.

The convolution is strictly causal: padding is on the left only, so position
`t` depends on `x[t−4..t]`. Symmetric padding would reveal future tokens and
leak the answer into loss. The last four inputs are retained in a **convolution
buffer**, carried as an additional element of `hidden`, so context continues
across chunk boundaries and character-by-character generation.

### Residuals, weight tying, and head

- `--residual` (enabled): `x = x + h_new` after each layer. For the first layer
  this works only when `embedding_dim == hidden_size`.
- `--tie-weights` (enabled): the output layer uses the embedding matrix. It
  requires `embedding_dim == hidden_size`.
- `--mlp-head` (disabled): replace one `Linear` with `Linear → GELU → Linear`.
- Initial states `h0` are learned, one parameter per layer.

### Compact ChunkMemory (`--memory-slots N`, `--memory-dim D`)

New CLI training uses three history slots, a 64-channel projection bottleneck
and four carried chunks (`--state-carry 4`). Each slot stores the detached final
raw state of one chunk plus an occupancy flag. Only written slots participate
in attention; an empty buffer is an exact identity. Learned compression and
retrieval operate when the next chunk reads the buffer.

At H384, D64 this adds **106,624 parameters**, versus **738,432** for the legacy
full-width module. The prior 4.54M model's dimensions with compact memory total
**4,641,542 parameters** with RMSNorm, below 5M. These are parameter counts,
not evidence of better conversational quality.

- `--memory-slots 0` disables this side memory.
- `--memory-dim 0` selects the legacy full-width format for compatibility.
- Four chunks expose up to three historical summaries during training. More
  slots require longer training spans to exercise their full capacity.
- Hidden states are detached at each chunk boundary: the BPTT horizon remains
  `--seq-len`, while forward context spans `seq_len * state_carry` tokens.
- Generation/chat read and write at the saved `seq_len` boundaries and reset
  the FIFO for each request. Memory is bounded and does not retain every token.
- RMSNorm checkpoints lacking `memory_dim` restore the legacy memory layout.
  LayerNorm checkpoints require an earlier revision; trained weights are not
  converted to new defaults.

### Innovation-gated associative memory (experimental)

`--innovation-memory-dim 32` adds a 32×32 recurrent matrix alongside the liquid
core, before the final normalization/readout. The LTC/CfC layers still process
every token. The side memory reads old key/value associations, then writes only
the residual between a new value and its stored prediction. A learned write
rate is modulated by that residual's energy. Each value channel has its own
learned continuous decay, initialized to horizons between 32 and 1,024 tokens.

This combines ideas from [LTC](https://arxiv.org/abs/2006.04439) and
[delta-rule memory](https://arxiv.org/abs/2412.06464) in a project-specific
experiment; it is not a claim of scientific priority. There is no attention
softmax or growing token cache. Memory is read **before** the current write,
accumulated in FP32 under AMP, and carried/detached with the other hidden state.
It works in training, generation and chat. State resets with a new sequence.

The default `0` disables the module and preserves old checkpoint behavior.
Enabling it requires training a model with the new parameters; the flag does
not retrofit an old checkpoint during generation. At H=384, D=32 it adds
62,659 parameters and 4 KiB of current state per example. BPTT activations and
optimizer state cost additional memory. The reference memory operations use
PyTorch, including when the LTC core uses Triton; measure the added step cost.

```bash
python liquid_text_model.py --data data --innovation-memory-dim 32
python benchmark_innovation.py --device cpu --steps 800 --seeds 11 22 33
```

The synthetic benchmark compares LTC, an FFN parameter control and the side
memory using identical shared weights, training batches and held-out queries.
It tests associative recall, not language quality; no external dataset is used.

After 800 updates per arm on three seeds, mean held-out accuracy was:

| Model | 8 distractors | 32 distractors | 64 distractors |
|---|---:|---:|---:|
| LTC, 32,798 parameters | 23.70% | 15.56% | 12.11% |
| LTC + FFN, 37,534 parameters | 21.68% | 16.73% | 15.49% |
| LTC + memory, 38,145 parameters | **36.85%** | **32.94%** | **27.93%** |

Training used eight distractors; longer delays test transfer. Chance among the
eight values is 12.5%. The memory arm beat both controls at the trained delay
on all three seeds, but varied substantially at longer delays. This equal-update
comparison gives it more compute; it does not establish a language-quality or
equal-wall-time advantage. A small RTX 5050/Triton cost check measured median
steps of 37.98 / 42.57 / 128.07 ms respectively (B=64, H=64, T=22, 30 steps).
With the same 800-update recall protocol, ordinary delta writes scored 43.49%
at delay 8 and 37.17% at delay 256, versus 36.85% and 20.96% for the error-energy
gate. The extra gate is therefore not supported as an improvement by this test.
With 30 measured training seconds per seed on the RTX 5050, mean recall at
delay 8 was 23.31% for LTC, 18.29% for innovation memory and 29.04% for ordinary
delta. Substantial throughput variation limits this small equal-time comparison.

A follow-up on *Alice's Adventures in Wonderland* did **not** confirm a language
benefit: after 300 updates, three-seed mean held-out character NLL was 1.9575
for LTC, 1.9885 for this memory and 1.9843 for ordinary delta writes (lower is
better). These small fresh models used a train-only vocabulary and an 80/10/10
split, without selecting weights on validation/test. The experiment remains
disabled by default; CONTRIBUTING records the controls and limitations.

### Initialization

Embeddings use `N(0, 0.02)`; recurrent matrices are orthogonal with `gain=0.99`;
input projections use Xavier; biases are zero unless noted. `h0` is initialized
to zero and learned. For `ltc`, `tau_mix` starts uniformly, while the time
network is initialized for a stable, moderately slow starting dynamic.

## How training works

### TBPTT with state carry

The corpus is sampled in windows of `span = seq_len × state_carry`, processed as
`state_carry` sub-chunks of `seq_len` tokens. Hidden state and convolution
buffer are **carried** between sub-chunks but **detached** from the graph,
limiting GPU memory. The effective training context is `span` tokens, while
the gradient depth is `seq_len`.

One iteration (`step`) processes `batch_size × span` tokens. Parameters are
updated once per `--grad-accumulation` iterations.

### Loss function

Cross-entropy is normalized over valid target tokens in the full gradient-
accumulation group. Summed losses are scaled by the maximum target positions;
before clipping, gradients are rescaled by `max_target_positions / valid_target_tokens`.
This keeps masked dialogue targets correctly weighted. With no masking, the
result is equivalent to ordinary mean cross-entropy.

### Loss mask (`--loss-mask`, enabled by default)

When dialogue tags are found (BPE only), loss is calculated only for tokens:

- **after** `<|bot|>` (response text);
- the `<|end|>` token (so the model learns to finish the response).

User tokens, system instructions, and role tags themselves are excluded. The
excluded-token ratio is printed at startup.

### Dialogue anchors (`--dialogue-anchor-ratio`, default `0.7`)

This fraction of each batch's windows starts **exactly at** `<|user|>`; the rest
are random. This exposes the model more often to dialogue starts and teaches
the “question → answer” format. Anchors are built independently for training
and validation. Without dialogues, sampling is fully random.

### Optimizers

**AdamW (`--optimizer adamw`)** uses `betas=(0.9, 0.95)`; CUDA uses the fused
variant when available and otherwise the regular implementation.

**Muon + AdamW (`--optimizer muon`)** is hybrid:

- strictly **2D hidden-layer parameters** use Muon (momentum plus update
  orthogonalization with five Newton–Schulz iterations in bf16);
- embeddings, output layer, biases, RMSNorm, `h0`, `tau_mix`, the 3D
  convolution, and `ChunkMemory.init_slot` use AdamW.

The hybrid has a separate schedule for each part; learning rates are supplied
by `--muon-lr` and `--lr`.

### Learning-rate schedule

Linear warmup (`--warmup-steps`, measured in **optimizer updates**, not
iterations) is followed by cosine decay to `--min-lr-ratio × lr`. Total updates
are `epochs × ceil(steps_per_epoch / grad_accumulation)`.

### EMA (`--ema-decay`, default `0.999`)

Weights use a moving average with warmup
(`min(decay, (1+n)/(10+n))`). After every epoch, validation runs twice with
regular and EMA weights. The lower loss is used; if EMA is better, the best
checkpoint stores EMA weights and the log includes `(EMA)`. Training then
continues with regular weights. `--ema-decay 0` disables EMA.

### Mixed precision (`--amp`)

`auto` selects bf16 when supported and otherwise fp16; `fp16` enables GradScaler;
`off`, or CPU execution, disables mixed precision.

### torch.compile (`--compile`)

Only one recurrent step (`_step_impl`) is compiled, not the whole model.
Input-weight preparation and the final normalization run outside that step;
the latter is batched across time before readout.
Modes are `default`, `reduce-overhead` (CUDA Graphs), and `max-autotune` (not
recommended for these small matrices because it is slow and often provides no
benefit). If compilation fails, the program falls back to `default` and then
eager mode. For `--generate-only`, compilation uses evaluation mode.

### Gradient checkpointing (`--grad-checkpoint`)

Checkpointing is enabled for every cell call. The granularity is very fine, so
overhead may exceed savings: TBPTT memory is primarily the O(T) activations
across the window. Measure on your own dimensions before relying on it.

### Distillation (`--teacher-checkpoint`)

`loss = (1−α)·CE + α·T²·KL(teacher‖student)`, where `α = --kd-alpha` and
`T = --kd-temperature`. Teacher and student tokenizers must have identical token
IDs and configurations. The teacher carries hidden state and ChunkMemory across
sub-chunks, resetting both at the start of each span. Top-k KD computes the tail
in log space to preserve gradients for highly confident predictions.

### Early stopping and saving

- The best checkpoint (by validation loss) is saved immediately on improvement.
- The “last” checkpoint is written every `--save-every` epochs.
- `--patience N` stops after `N` epochs without improvement (`0` disables it).
- A generation preview is printed every 5 epochs, on the final epoch, and after
  every improvement.
- At the end, the best model generates the final text.

### Logging

`--log-backend tensorboard` or `wandb` records:
`train/loss`, `train/loss_running`, `train/perplexity`, `val/loss`,
`val/loss_ema`, `val/perplexity`, `train/lr`, `train/grad_norm`, and
`perf/tokens_per_sec`. If the package is absent, the program warns and
continues without logging.

```bash
tensorboard --logdir runs
```

### Stability safeguards

Loss and gradients are checked for NaN/Inf before unscaled optimizer updates,
including with `--grad-clip 0`. FP16 overflow skips are handled by GradScaler;
the scheduler, global step and EMA advance only after successful updates.
Sampling options are validated before training starts. Checkpoints use a
temporary file and atomic replacement, preserving the previous file on a failed
write and creating missing parent directories.

## Generation and chat

### Ordinary generation

```bash
python liquid_text_model.py --generate-only \
  --prompt "Neural network" --generate 600 \
  --temperature 0.8 --top-k 50 --top-p 0.95
```

The prompt is processed and tokens are selected one at a time. The order is:
repetition penalty → temperature division → top-k → top-p → sampling.
Generation stops at `<|end|>` (when present) or at `--generate`. State and the
convolution buffer are carried, so every new token costs O(1).

Prompt processing projects only the final position into vocabulary logits;
all prompt tokens still pass through the recurrent layers. The regular training
`forward` output is unchanged.

| Parameter | Effect |
|---|---|
| `--temperature` | Lower is more predictable; higher is more diverse |
| `--top-k` | Keep the k most probable tokens (0 disables it) |
| `--top-p` | Nucleus sampling: smallest set with cumulative probability ≥ p |
| `--repetition-penalty` | Values > 1 reduce probability of recent tokens |
| `--repetition-window` | Number of recent tokens considered |

Special role tokens are not printed during decoding.

### One-shot dialogue response

```bash
python liquid_text_model.py --chat --prompt "What is an LTC network?"
```

The prompt is wrapped in `<|user|>\n…\n<|bot|>\n`. Generation stops at
`<|end|>` or at the beginning of a new user turn; only the response is printed.

### Interactive chat

```bash
python liquid_text_model.py --chat-repl
```

An empty line, `exit`, `quit`, or `q` exits. Dialogue history is stored
as text and truncated to `--chat-history-chars` (default 4000) at a turn
boundary.

> `--chat` and `--chat-repl` enable generation themselves; `--generate-only`
> is not needed.

### Which model generation uses

The file from `--checkpoint` (default `liquid_text_best.pt`) is used. The
architecture and tokenizer are read from the checkpoint itself.

## Triton kernel (experimental)

`--kernel triton` fuses the two LTC add/tanh activations and their first-order
backward, including gradients for the two recurrent projection slices. The
Euler update uses a separate fused forward/backward kernel. In multi-tau mode,
each CUDA program writes only three partial mixing-weight gradient sums.
The mixing softmax and normalization remain in PyTorch. RMSNorm uses the native
PyTorch operation when available, with FP32 statistics for FP16/BF16 inputs. Matrix
multiplications remain on cuBLAS. These implementation changes preserve model
parameters, checkpoint keys, dropout, zoneout and normalization behavior.
FP16/BF16 additions and intermediate tanh-backward operations retain PyTorch's
rounding; the implementation does not claim bitwise equivalence across backends.

Requirements: CUDA, a Triton package compatible with the local PyTorch build,
and `--gate-mode ltc`. Selecting Triton uses the eager
recurrent step instead of `torch.compile`; compare full-model tokens/sec for
both paths before long training runs. Run `--selftest-triton` and
`python benchmark_triton_tail.py --batch-size 64 --hidden-size 384 --dtype bf16`
for a forward selftest and an isolated forward/backward tail benchmark. The
selftest exits with code 2 on a numerical mismatch. Run
`python tests/check_triton_mixed.py --output runs/triton_correctness.json` for
mixed-dtype and full-model gradient comparisons, and `benchmark_gpu.py` for
model-level timings. Windows was tested with `triton-windows 3.8.0.post28`.

## CLI parameter reference

### Operating modes

| Flag | Default | Description |
|---|---|---|
| `--generate-only` | — | Generate from a checkpoint only |
| `--chat` | — | One dialogue-format response (enables generation) |
| `--chat-repl` | — | Interactive chat (enables generation) |
| `--chat-history-chars` | `4000` | Maximum REPL history length |
| `--info` | — | Show checkpoint information and exit |
| `--resume` | — | Continue from `--last-checkpoint` |
| `--selftest-triton` | — | Test the Triton kernel and exit |
| `--trust-checkpoint` | — | Allow unsafe loading (`weights_only=False`) |

### Data

| Flag | Default | Description |
|---|---|---|
| `--data` | `data` | Corpus directory |
| `--min-chars` | `1000` | Minimum corpus size |
| `--max-file-mb` | `0.0` | Skip files larger than N MB (0 means unlimited) |
| `--val-split` | `0.1` | Validation fraction (0.01–0.5) |
| `--tokenizer` | `bpe` | `bpe` or `char` |
| `--vocab-size` | `8000` | BPE vocabulary size (≥ 300) |
| `--loss-mask` / `--no-loss-mask` | enabled | Compute loss only on bot responses |
| `--dialogue-anchor-ratio` | `0.7` | Fraction of windows starting at `<\|user\|>` |
| `--shuffle-files` / `--no-shuffle-files` | enabled | Deterministic file shuffling |
| `--file-headers` / `--no-file-headers` | disabled | `===== filename =====` headers between files |

### Training

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `20` | Total epochs (`--resume` uses total, not additional, epochs) |
| `--steps-per-epoch` | `250` | Iterations per epoch |
| `--eval-batches` | `30` | Validation batches |
| `--batch-size` | `16` | Batch size |
| `--seq-len` | `128` | Sub-chunk length (≥ 8) |
| `--state-carry` | `4` | Sub-chunks carrying state |
| `--grad-accumulation` | `4` | Gradient accumulation |
| `--grad-clip` | `1.0` | Maximum gradient norm (0 disables it) |

### Architecture

| Flag | Default | Description |
|---|---|---|
| `--embedding-dim` | `384` | Embedding size |
| `--hidden-size` | `384` | Hidden-state size |
| `--num-layers` | `2` | Number of liquid layers |
| `--dropout` | `0.05` | Dropout, `[0, 1)` |
| `--zoneout` | `0.1` | Zoneout, `[0, 1)` |
| `--tie-weights` / `--no-tie-weights` | enabled | Tie embedding and output |
| `--mlp-head` / `--no-mlp-head` | disabled | MLP head |
| `--residual` / `--no-residual` | enabled | Residual connections |
| `--multi-tau` / `--no-multi-tau` | enabled | Three time-constant banks |
| `--local-conv` / `--no-local-conv` | enabled | Causal `k=5` convolution |
| `--gate-mode` | `ltc` | `ltc` or `cfc` |
| `--ffn-mult` | `0` | SwiGLU width multiplier (0 disables FFN) |
| `--memory-slots` | `3` | Occupied FIFO history slots (0 disables memory) |
| `--memory-dim` | `64` | Compact projection width; 0 selects legacy full-width memory |
| `--innovation-memory-dim` | `0` | Experimental associative matrix width; 0 disables it |
| `--kernel` | `torch` | `torch` or experimental `triton` (LTC inference/training) |
| `--sequence-batching` / `--no-sequence-batching` | disabled | Batch sequence projections/FFNs between layers; stochastic training and compiled steps retain their existing schedule |
| `--grad-checkpoint` / `--no-grad-checkpoint` | disabled | Gradient checkpointing |

### Optimization

| Flag | Default | Description |
|---|---|---|
| `--optimizer` | `adamw` | `adamw` or `muon` |
| `--lr` | `2e-3` | AdamW learning rate |
| `--muon-lr` | `0.02` | Muon learning rate |
| `--muon-momentum` | `0.95` | Muon momentum |
| `--muon-ns-steps` | `5` | Newton–Schulz iterations |
| `--min-lr-ratio` | `0.1` | Final fraction of LR in cosine decay |
| `--warmup-steps` | `100` | Warmup optimizer updates |
| `--weight-decay` | `1e-4` | Weight decay |
| `--ema-decay` | `0.999` | EMA (`0` or `[0.8, 1)`) |

### Distillation

| Flag | Default | Description |
|---|---|---|
| `--teacher-checkpoint` | — | Teacher checkpoint |
| `--kd-alpha` | `0.3` | KD-loss weight |
| `--kd-temperature` | `2.0` | Temperature |

### Device and speed

| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`, … |
| `--amp` | `auto` | `auto`, `fp16`, `bf16`, `off` |
| `--compile` | disabled | Compile recurrent step |
| `--compile-mode` | `default` | `default`, `reduce-overhead`, `max-autotune` |

### Checkpoints and logging

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `liquid_text_best.pt` | Best checkpoint |
| `--last-checkpoint` | `liquid_text_last.pt` | Last checkpoint (for resume) |
| `--save-every` | `1` | Save “last” every N epochs |
| `--patience` | `15` | Early stopping (0 disables it) |
| `--log-backend` | `none` | `none`, `tensorboard`, `wandb` |
| `--log-dir` | `runs` | TensorBoard directory |
| `--wandb-project` | `liquid-lm` | W&B project |
| `--run-name` | checkpoint name | Run name |

### Generation

| Flag | Default | Description |
|---|---|---|
| `--prompt` | `Neural network` | Initial text / question |
| `--generate` | `800` | Maximum new tokens |
| `--temperature` | `0.85` | Temperature |
| `--top-k` | `40` | Top-k |
| `--top-p` | `0.95` | Top-p |
| `--repetition-penalty` | `1.05` | Repetition penalty |
| `--repetition-window` | `128` | Penalty window |
| `--seed` | `1` | Seed |

## Ready-to-use recipes

**Basic text training**

```bash
python liquid_text_model.py --data data --epochs 30
```

**More capacity and speed on a modern GPU**

```bash
python liquid_text_model.py --data data \
  --hidden-size 512 --embedding-dim 512 --num-layers 3 \
  --batch-size 32 --seq-len 256 --amp bf16 --compile
```

**CfC instead of LTC**

```bash
python liquid_text_model.py --data data --gate-mode cfc
```

**Muon + AdamW**

```bash
python liquid_text_model.py --data data --optimizer muon --muon-lr 0.02 --lr 2e-3
```

**Train a chat model on instructions**

```bash
# data/ contains .jsonl/.parquet with instruction/output or messages
python liquid_text_model.py --data data --tokenizer bpe --vocab-size 16000 \
  --loss-mask --dialogue-anchor-ratio 0.7 --epochs 40
python liquid_text_model.py --chat-repl
```

**Long memory with ChunkMemory**

```bash
python liquid_text_model.py --data data --state-carry 4 --memory-slots 3 --memory-dim 64
```

**Resume interrupted training**

```bash
python liquid_text_model.py --data data --resume --epochs 60
```

**Distill a large model into a small one**

```bash
python liquid_text_model.py --data data --hidden-size 256 --embedding-dim 256 \
  --teacher-checkpoint teacher_best.pt --kd-alpha 0.4
```

**Train with TensorBoard**

```bash
python liquid_text_model.py --data data --log-backend tensorboard --run-name exp1
tensorboard --logdir runs
```

**View checkpoint metadata**

```bash
python liquid_text_model.py --info --checkpoint liquid_text_best.pt
```

## Files created by the program

| File | Created | Contents |
|---|---|---|
| `liquid_text_best.pt` | On every validation improvement | Best checkpoint |
| `liquid_text_last.pt` | Every `--save-every` epochs | Last checkpoint (for `--resume`) |
| `training_info.json` | After training | Device, parameter count, tokenizer, corpus size, best epoch, losses, config, and speed |
| `generated.txt` | After training | Final text generated by the best model |
| `runs/<name>/` | With `--log-backend tensorboard` | TensorBoard logs |

## Checkpoints, resume, and compatibility

### What a checkpoint stores

Model weights, optimizer/scheduler/GradScaler/EMA states, epoch, global step,
train/validation loss, best validation score, launch configuration, the
**complete tokenizer**, and the corpus SHA-256.

### Resume

`--resume` reads `--last-checkpoint` **before** loading the corpus and obtains
the file order (`shuffle_files`) and header mode (`file_headers`) from it;
otherwise the train/validation split would shift. Architecture parameters
(`embedding_dim`, `hidden_size`, `gate_mode`, `memory_dim`, `memory_slots`, and
others) also come from the checkpoint. Differences from the command line are
printed as `[RESUME] … (from checkpoint)`. The tokenizer is restored rather
than retrained. A changed corpus produces a warning.
The validation fraction and split policy are also restored from the checkpoint.
Checkpoints are loaded on CPU before copying required tensors to the selected
device. Generation/chat discard optimizer and EMA state and release checkpoint
weight copies after loading the model.

`--epochs` is the **total** epoch count: to train a model that already ran for
20 epochs for 40 more, specify `--epochs 60`.

If optimizer or scheduler state is incompatible (for example, because
`--optimizer` changed), fresh state is used and reported.

### Safe loading

Checkpoints are loaded with `weights_only=True`, without executing arbitrary
pickle code. If safe loading fails, a clear error is shown. For **your own
trusted files**, you may add `--trust-checkpoint`; never use it with untrusted
files.

### Backward compatibility

Only checkpoints explicitly recording `norm_type=rms` can be resumed or used
for generation in this revision. LayerNorm checkpoints and older checkpoints
without normalization metadata require an earlier revision. Their weights and
normalization must not be silently reinterpreted.

Within the supported RMSNorm format, historical defaults still apply to other
missing architecture fields (`gate_mode=ltc`, `memory_slots=0`, and so on).
Supported missing weights (`h0`, `tau_mix`, convolution) are initialized with
a message. Early versions used a **non-causal** convolution; use an earlier
compatible revision for those files and retrain for causal evaluation.

The v3.6 BPE tokenizer differs from earlier versions (no prefix whitespace and
a complete byte alphabet). Supported checkpoints use their saved tokenizer;
retired checkpoints can use theirs with an earlier compatible revision. Retrain
from scratch for fair experiment comparisons.

### Retired experimental options

NCP wiring, FIM data rearrangement, Heun and configurable ODE unfolds have been
removed from the training interface and implementation. Their benefit for this
small conversational workload was not established; this is a scope decision,
not a claim that the methods are universally ineffective. Supported RMSNorm
checkpoints (`dense`, `euler`, one unfold) retain their equations and token IDs.
Checkpoints using retired equations, or resuming a FIM corpus recipe, fail with
an explicit message directing users to the earlier revision. Historical test
results below describe the feature set at the time they were recorded.

## Hyperparameter selection and GPU memory

### Model size

The main parameter consumers are the embedding (`vocab × embedding_dim`, also
the output layer when tied) and each layer's recurrent/input matrices, roughly
`(input + hidden) × hidden × k`. The parameter count is printed at startup as
`[MODEL] Parameters`.

### If memory is insufficient (OOM)

In decreasing order of effectiveness:

1. Reduce `--batch-size` and compensate with `--grad-accumulation` (effective
   batch is their product).
2. Reduce `--seq-len` (memory grows linearly with sub-chunk length; preserve
   context with `--state-carry`).
3. Enable `--amp bf16` (or `fp16`).
4. Reduce `--hidden-size` / `--num-layers`.
5. Reduce `--vocab-size`; `batch × seq × vocab` logits can be the largest tensor.
6. Reduce `--memory-dim`, or disable memory with `--memory-slots 0`.
7. Use `--grad-checkpoint` as a last resort and measure it.

On Linux, `expandable_segments` is already enabled for the CUDA allocator to
reduce fragmentation.

### If training is slow

The model is recurrent, so the time loop is sequential:

- use `--compile`, especially `--compile-mode reduce-overhead`;
- increase `--batch-size` (the recurrent step is nearly batch-size independent
  on GPU);
- use bf16/fp16 instead of fp32;
- benchmark `--kernel triton` and `--sequence-batching` on the intended workload;
- monitor `tok/s` in the epoch log.

### Rules of thumb

- The default `--state-carry 4` with `--seq-len 128` gives a 512-token forward
  training span and a 128-token gradient horizon. Increase carry for longer
  history; increasing slots alone does not extend training examples.
- Start AdamW at `--lr 2e-3`; if it diverges, lower to `1e-3` and strengthen
  `--grad-clip`.
- If validation loss rises while training loss falls, increase data,
  `--dropout`, `--zoneout`, or `--weight-decay`.
- For small (single-digit MB) corpora use `--vocab-size` 4000–8000; for large
  and dialogue corpora use 16000+. The program reduces an oversized vocabulary
  automatically on small corpora.
- Logged perplexity is `exp(loss)`, capped at `exp(20)`.

## Known limitations

- ChunkMemory is active during generation/chat when `--memory-slots > 0`; it
  reads and writes chunk summaries at the checkpoint's `seq_len` boundaries.
  This bounded FIFO is not full-history attention and resets for each request.
- With `--kd-topk > 0`, distillation represents the teacher's probability mass
  outside top-k as a tail bucket; `--kd-topk 0` enables full-vocabulary KL,
  with greater memory and compute cost.
- Triton supports LTC inference and first-order training backward; CfC uses
  the PyTorch path. ChunkMemory is a project-specific summary mechanism; its
  existence does not establish reliable long-term conversational recall.
- Cell-level gradient checkpointing may not save memory.
- The Python recurrent loop limits long-sequence throughput. Measure the intended
  batch size, hidden size and backend when comparing CPU/GPU performance.
- The char tokenizer does not support dialogue tags, loss masking, or anchors.
- Validation is the corpus tail and can be unrepresentative with few files and
  `--no-shuffle-files`.
- This is not a ready-made assistant; quality depends on the amount and quality
  of your data.

## Troubleshooting

| Symptom | Cause and solution |
|---|---|
| `No corpus files in data` | Put files with supported extensions in `--data` |
| `Corpus is too small` | Add text or lower `--min-chars` |
| `Train/Validation portion is too small` | Lower `--seq-len` or `--state-carry`, or add data |
| `CUDA selected, but torch.cuda.is_available() == False` | Install a CUDA PyTorch build or use `--device cpu` |
| `tokenizers is not installed — falling back to char` | `pip install tokenizers` |
| Parquet/CSV is skipped or read incorrectly | `pip install pandas pyarrow` |
| `WARNING: uncertain encoding detection` | Convert files to UTF-8 or install `charset-normalizer` |
| `--tie-weights requires --embedding-dim == --hidden-size` | Make the dimensions equal or add `--no-tie-weights` |
| `Could not safely load … (weights_only=True)` | For your own file, use `--trust-checkpoint` |
| `Loss NaN/Inf` | Lower `--lr`, keep `--grad-clip`, and try `--amp bf16` or `off` |
| `torch.compile failed` | It falls back to eager mode; update PyTorch or omit `--compile` |
| `triton unavailable (…)` | The reason is printed and the PyTorch path is used |
| `Optimizer/scheduler state incompatible` | Optimizer settings changed; fresh state is used |
| Generation loops | Increase `--repetition-penalty` (1.1–1.2), lower `--temperature`, and check `--top-k/--top-p` |
| Chat answers off-topic | Train on dialogue data; use BPE and more data |
| `<\|user\|>` tags are absent from output | Special tokens are hidden during decoding; this is normal |

## Code structure

```
liquid_text_model.py          # thin command-line entry point
liquid_lm/
├── __init__.py               # environment setup before importing torch
├── __main__.py               # python -m liquid_lm
├── cli.py                    # CLI, argument validation, mode dispatch
├── data.py                   # corpus readers, cleaning and dialogue parsing
├── inference.py              # generation-only and interactive chat
├── layers.py                 # liquid cells, native normalization, chunk memory, Triton kernel
├── associative.py            # experimental innovation-gated recurrent memory
├── model.py                  # LiquidLanguageModel, generation and checkpoint I/O
├── optim.py                  # AdamW, Muon, schedulers
├── tokenizer.py              # character and byte-level BPE tokenizers
├── training.py               # batching, losses, AMP, EMA, evaluation and trainer
└── utils.py                  # device helpers, seeding and logging
requirements.txt              # Python dependencies

tests/
├── test_regressions.py       # sequence, masking, resume and numerical regressions
├── test_audit.py             # KD, checkpoint writes, prefill and AMP state
├── test_gpu.py               # CUDA parity, AMP, scaler and checkpoint tests
├── test_innovation_memory.py # memory equations, causality, carry, gradients and AMP
├── test_memory_benchmarks.py # overwrite/query labels and ablation initialization
├── test_memory_language_benchmark.py # vocabulary and held-out split boundaries
├── test_compute_schedule.py   # batching, projection/AMP parity and compiled step
├── test_triton_fusions.py    # fused activations, AMP rounding and optimizer parity
└── check_triton_mixed.py     # standalone mixed-dtype/full-model Triton checks
benchmark_model.py            # CPU forward/backward, generation and KD
benchmark_gpu.py              # synchronized CPU/CUDA/Triton/compiled comparisons
benchmark_triton_tail.py       # isolated LTC tail forward/backward
benchmark_innovation.py       # synthetic associative recall with parameter control
benchmark_memory_validation.py # ablations, overwrite/query tasks and time budgets
benchmark_memory_language.py  # fixed-budget language comparison on supplied text
benchmark_compute.py          # paired before/after compute schedule and A/A control
.github/workflows/
└── tests.yml                 # CPU test workflow
```

Run all commands from the repository root. The optional dependencies
(`pandas`, `charset-normalizer`, `tokenizers`, `tensorboard`, `wandb`,
`triton`) are imported with a fallback in the module that uses them, so a
missing package never breaks unrelated parts of the program.

### Adding a custom architecture option

1. Add the argument to `build_parser()` in `liquid_lm/cli.py`.
2. Implement it in `LiquidLanguageModel` / `LiquidCell` in
   `liquid_lm/model.py` and `liquid_lm/layers.py`, then pass it through
   `run_training` in `liquid_lm/training.py`.
3. Add its key to `ARCH_KEYS` and a default for old checkpoints to
   `LEGACY_ARCH_DEFAULTS` in `liquid_lm/model.py`, so `--resume` and old
   checkpoints remain compatible.
4. Pass the parameter to `create_model_from_config` in `liquid_lm/model.py`.

## Tests and measured results

```bash
python -m unittest discover -s tests -v
python benchmark_gpu.py --device both --cases train_realistic16 generate128 --precisions fp32 --threads 1 --warmup 3 --repeats 7 --output runs/benchmark_torch.json
```

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

Before the feature cleanup on **19 October 2026**, the full suite passed **101 tests without skips** on
Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, driver 616.56 and an
**RTX 5050 Laptop 8 GB**, including five checkpoint-averaging tests. That earlier
CPU-only run with PyTorch 2.14.1 passed 82 and skipped 19 checks
(101 discovered; CUDA/Triton and optional BPE). Both suites pass without local datasets. The initial audit
passed 13 GPU CLI scenarios and 2 compiled
CLI scenarios. Activation fusion was also checked with 18 Triton numerical
comparisons and 5 focused CLI scenarios. These checks cover
LTC/CfC, dense/NCP, AMP, AdamW/Muon, overflow, memory, resume, BPE dialogs and KD.

Dataset loading now has 12 focused regressions, including BOM handling, JSON
text extraction, invalid UTF-8, excluded-directory pruning and split isolation.
On a temporary 4.22 MB UTF-8 fixture, full loading took median 175.18 → 124.11 ms
(median paired speedup 1.43×, seven pairs, warm filesystem cache); corpus hashes
matched. This measures loading, not end-to-end training throughput. A CUDA BF16
BPE training/resume smoke retained the saved split despite a changed CLI fraction.

The memory audit added nine lifetime/storage regressions. For the existing
2.47M checkpoint, live CUDA allocation at generation entry fell from
51.74 to 9.87 MB by excluding training state and checkpoint weight copies
(measured above the process baseline, generation itself excluded). A carried
convolution tail retained 2 KiB instead of 66 KiB in a B2/T128/E64 fixture.
Repeated training, validation and generation showed no growth in live CUDA
allocation after warmup across three tested configurations. These finite checks
do not establish leak freedom for every workload or a multi-hour run.

Opt-in `--sequence-batching` finishes each recurrent layer's sequence before
batching the next layer's input projections and token-wise FFNs. Recurrent
state updates remain sequential. It works with existing checkpoints and
Triton, and is available as `model.set_sequence_batching(True)`. Pass the flag
again for resume, generation or chat. Active cell/FFN dropout, zoneout and
compiled recurrent steps retain the original schedule; eval can still batch
sequences when training regularization is configured. Disabling regularization
for speed has not been evaluated for language quality. AMP gradient rounding
can differ.

Paired RTX 5050/Triton measurements used H384/L2/V1024, B16/T64 training,
prompt128+32-token generation, TF32 off, four warmups and nine alternating pairs:

| Operation | FP32 | BF16 | FP16 |
|---|---:|---:|---:|
| Base model training, dropout/zoneout 0 | 1.09× | 1.18× | 1.18× |
| Training with `ffn_mult=1`, dropout/zoneout 0 | 1.70× | 1.72× | 1.70× |
| Greedy generation | 1.10× | 1.13× | 1.12× |

Each ratio compares the same architecture/weights/data with batching off/on;
the FFN case has 3.35M parameters versus 2.47M in the base case. FP32 base
training is the weakest result (7/9 winning pairs, IQR 1.001–1.128). Identical
FP32 controls were 0.976×/1.001×/0.991× for base/FFN/generation. These are
operation timings, excluding loading, evaluation and saving. Prefill needed
up to 0.73 MiB more peak temporary CUDA memory in these fixtures. Reproduce:
`python benchmark_sequence_batching.py --kernel triton --precisions fp32 bf16 fp16`.

A matched-token continuation experiment measured a small quality
gain at the same 2.47M parameters: additional training plus uniform averaging
reduced fixed-window validation NLL from 3.473696 to 3.459669 (perplexity
32.2557 to 31.8065). Increasing the training horizon from 64 to 128 did not
beat paired controls on either seed. These are results on reused Shakespeare
validation, without an independent test set. CONTRIBUTING describes the
experiment settings and the `research_liquid.py` runner.

Compute scheduling now batches final normalization, shares multi-tau weights
within a forward graph and prepares input projections outside the token loop.
FP32 combines input heads; AMP retains separate head GEMMs to preserve their
gradient-rounding order. Parameters, recurrent equations and checkpoints are
unchanged. Fifteen adjacent alternating pairs on the 2.47M configuration
(B16/T64/H384/L2/V1024, CUDA/Triton, TF32 off) measured:

| Workload | FP32 speedup | BF16 | FP16 |
|---|---:|---:|---:|
| Forward + CE + backward + AdamW | 1.171× | 1.116× | 1.097× |
| Same with dropout=0.05, zoneout=0.1 | 1.122× | 1.109× | 1.082× |
| Prompt 128 + 32 generated tokens | 1.118× | 1.124× | 1.075× |

These are medians of paired time ratios, not ratios of separate medians.
Windows timing varies: identical-code control measured 1.035×, with IQR
0.980–1.052×; FP16 generation was also noisy. Three prompts from the original
BPE checkpoint produced identical greedy 64-token completions before/after;
its SHA-256 and state keys were unchanged. Reproduce with:

```bash
python benchmark_compute.py --pairs 15 --warmup 5 --precisions fp32 bf16 fp16
```

Initial CUDA audit, before activation fusion: FP32 medians after
3 warmups and 7 measured iterations (Triton fused only the LTC tail):

| Workload | CPU, 1 thread | CUDA eager | CUDA Triton | CUDA compiled |
|---|---:|---:|---:|---:|
| Training step, B=16, T=64 | 2293.1 ms | 939.9 ms | 204.9 ms | 135.8 ms |
| Generation, prompt=128 + 32 new tokens, B=1 | 484.5 ms | 234.3 ms | 158.2 ms | 104.7 ms |

Both workloads use synthetic inputs, matching seeds and initial weights,
LTC/dense Euler, H=embedding=384, L=2, V=8000, multi-tau/local convolution on,
dropout=zoneout=0, no FFN or ChunkMemory. CPU intra-op/inter-op threads are 1,
TF32 is off, and CUDA timings synchronize both boundaries. Training includes
forward, CE, backward and AdamW; generation includes prompt processing and decoding.
Imports, setup, transfers and JIT warmup are excluded. First compilation can take
minutes. Smaller models can be faster on CPU; FP16 showed substantial variance,
and AMP did not automatically improve speed or memory. These short runs verify
functionality, not answer quality or long-training stability.

Activation-fusion follow-up on the same GPU: median paired speedups over the
previous tail-only Triton implementation:

| Precision | Dropout/zoneout off | Default dropout=0.05, zoneout=0.1 |
|---|---:|---:|
| FP32 | 1.048× | 1.003× |
| BF16 | 1.037× | 1.054× |
| FP16 | 1.018× | 1.031× |

These compare full training steps at B=16, T=64, H=384, L=2, V=8000, with
3 warmups and 12 measurements per version, alternating in ABBA order in one
process. Losses and final parameters matched after 15 updates in all six cases.
Timing variation remains substantial; the regularized FP32 result is within
noise. See the detailed timings and methodology in CONTRIBUTING.

See [CONTRIBUTING.md](CONTRIBUTING.md) for test coverage, benchmark commands and
guidelines for reporting new measurements. Fresh benchmark results are written
to the ignored `runs/` directory.

### Small English model: local training experiment

On 19 October 2026, a **4,536,838-parameter** LTC model was trained from scratch
on the RTX 5050 Laptop: two H384 layers, 4096-token BPE, tied embeddings,
multi-tau, local convolution and `ffn_mult=1`. Training used Triton,
`--sequence-batching`, FP32 with TF32, T128 and two carried subchunks.

The corpus used 50,000 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
GPT-4 v2 stories (CDLA-Sharing-1.0), 2,065 training pairs from
[Everyday Conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k)
(Apache-2.0), and 24,000 authored curriculum examples. Validation and final-test
records were separated before fitting; BPE was fitted on training text only.
Repeated short-answer examples changed sampling weights, not the split.

| Final held-out evaluation | Text-trained baseline | After response tuning |
|---|---:|---:|
| Exact match, 200 balanced curriculum questions | 166/200 (83%) | 178/200 (89%) |
| Arithmetic subset | 1/28 | 8/28 |
| Story perplexity, 256 stories / 52,490 answer targets | 6.662 | 7.246 |

Story loss uses non-overlapping 256-token windows with state reset per window.
Curriculum questions use held-out parameter combinations from the same task
templates; these scores do not measure general assistant ability. Free dialogue
often gives irrelevant or invented answers, and arithmetic remains weak.

Pretraining completed 3,600 updates (batch 64 for the first 700, then 256).
Response tuning stopped after 1,100 saved updates when validation deteriorated;
the selected response model uses the earlier 100-update checkpoint. Selection
compared validation answer accuracy and story perplexity before the final test.

The exported response model is about **18.3 MB**, including its tokenizer.
Warm native generation measured **165.6 new tokens/s on CUDA/Triton** and
**127.7 on CPU/torch**: B1, 29 prompt tokens plus 64 forced greedy tokens,
two warmups, five repetitions, one CPU thread, loading excluded, EOS stopping
disabled for the benchmark, GPU clocks/power unlocked. Model weights, data
manifests and detailed local results live under the Git-ignored `runs/english_small_4m/`.

## Change history

The current training interface focuses on dense LTC/CfC cells, native RMSNorm
and compact chunk memory. See the compatibility notes for retired options.

### 19 October 2026: numerical fixes and CUDA validation

- Fixed saturated-logit top-k KD gradients, nonfinite-gradient handling and
  CfC AMP state reads through ChunkMemory.
- Checkpoint writes preserve the previous file after a write failure; prompt
  generation avoids projecting unused positions; invalid sampling settings fail early.
- Added CPU/CUDA regressions and reproducible benchmarks without changing the
  model architecture, parameter names or checkpoint architecture keys.
- Verified the CUDA, Triton and compiled paths on an RTX 5050 Laptop GPU.
- Fused LTC activation forward/backward into the Triton
  path; added a full-step benchmark with the default dropout and zoneout.

## Sources and inspiration

- R. Hasani, M. Lechner et al. — *Liquid Time-constant Networks* (AAAI 2021).
- R. Hasani et al. — *Closed-form Continuous-time Neural Networks* (Nature
  Machine Intelligence, 2022).
- M. Lechner et al. — *Neural circuit policies enabling auditable autonomy*
  (Nature Machine Intelligence, 2020).
- Z. Dai et al. — *Transformer-XL* (cached-memory idea used by ChunkMemory).
- K. Jordan et al. — the Muon optimizer.
- Merity et al. — *Regularizing and Optimizing LSTM Language Models*
  (locked dropout); Krueger et al. — *Zoneout*.

This is a research/educational project. It is not an exact reproduction of any
one cited paper; it combines their ideas with the simplifications described
above.
