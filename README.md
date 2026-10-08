# Liquid LM — a language model based on liquid neural networks (LTC / CfC)

[English](README.md) | [Русский](README.ru.md)

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

> Current version: **v3.6**. The change history is at the end of this document.

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
18. [Change history](#change-history)
19. [Sources and inspiration](#sources-and-inspiration)

## Features

**Model**

- Two cell types: **LTC** (exponential ODE integrator) and **CfC** (the
  closed-form formulation from Hasani et al., 2022).
- **Multi-tau**: fast, medium, and slow time-constant banks mixed by a learned
  softmax.
- Euler (default) and Heun (RK2, second-order) sub-step solvers.
- **NCP wiring**: the recurrent matrix is replaced with physically smaller
  block matrices following the Neural Circuit Policies topology, reducing
  parameters and FLOPs.
- A causal depthwise convolution (`k=5`) before the liquid core, carrying
  context across chunks and generation steps.
- Residual connections, LayerNorm on cell output, zoneout, and locked
  embedding dropout.
- Optional **weight tying** between the embedding and output layer, and an
  optional MLP head.
- **ChunkMemory**: FIFO memory of previous chunk summaries with dot-product
  attention, inspired by Transformer-XL.

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
- Optional fused Triton kernel for the LTC sub-step tail (inference only,
  CUDA).

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
| `triton` | Fused inference kernel (`--kernel triton`, Linux + NVIDIA) |

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
python liquid_text_model.py --info --checkpoint nonexistent.pt   # prints Device/GPU/VRAM, then reports the checkpoint error
```

At startup the program prints the device, PyTorch version, GPU name, VRAM,
CUDA version, and Tensor Core status (TF32 / bf16), then reports the missing
checkpoint.

## Quick start

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
| `.txt .md .rst .log` | Text, with automatic encoding detection |
| `.json` | A list or object; dialogue structures become tagged text, otherwise text is extracted from `text/content/article/body/document` |
| `.jsonl` | One JSON record per line; the same rules apply, and unrecognized lines are appended as-is |
| `.csv .tsv` | Through pandas; question–answer columns become dialogue, otherwise a text column is used |
| `.parquet` | Same as tables (requires `pandas` + `pyarrow`) |

Without `pandas`, CSV/TSV files are read as ordinary text and Parquet files
are skipped.

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

- The validation portion is the **tail** of the tokenized corpus; its size is
  controlled by `--val-split` (default `0.1`, allowed `0.01–0.5`).
- To avoid a tail consisting systematically of the alphabetically last files,
  `--shuffle-files` is enabled by default. Files are ordered by SHA-256 of their
  relative path, deterministically.
- Both splits must be longer than `seq_len × state_carry + 2` tokens.
- The BPE tokenizer is trained **only on the training portion**, so the
  validation vocabulary cannot leak into training.

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
│  final LayerNorm(x)                                │
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

The cell models a linear ODE with an input-dependent time constant:

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

This exponential integrator is stable for any `dt` and `τ > 0`.

**Sub-steps (`--ode-unfolds N`).** One `dt=1` tick is divided into `N`
sub-steps of `dt/N`. More sub-steps improve the ODE approximation but increase
cost linearly.

**Solver (`--ode-solver`).**

- `euler`: evaluate `candidate`/`τ` once at the beginning of the sub-step.
- `heun`: predictor-corrector (RK2); evaluate the sub-step twice, at the
  beginning and at the predicted end, and average the values. This provides
  second-order accuracy at roughly twice the computation per sub-step and can
  often reduce `--ode-unfolds` at the same quality.

**Multi-tau (`--multi-tau`, enabled by default).** The `tau_net` output is
tripled into three banks:

| Bank | Formula | Meaning |
|---|---|---|
| fast | `softplus(r₀ − 2) + 0.05` | Fast dynamics |
| medium | `softplus(r₁) + 0.01` | Medium dynamics |
| slow | `softplus(r₂ + 2) + 0.001` | Slow, long memory |

The final `τ = Σ wᵢ · τᵢ`, where `w = softmax(tau_mix)` contains three learned
global (not per-channel) weights.

### LiquidCell — `cfc` mode

Closed-form formula from Hasani et al., *Closed-form Continuous-time Neural
Networks* (Nature Machine Intelligence, 2022). There is no explicit ODE; the
network directly parameterizes the solution.

```
ff1 = tanh(W₁ₓ·x + W₁ₕ·h)
ff2 = tanh(W₂ₓ·x + W₂ₕ·h)
t   = sigmoid(t_a · dt + t_b)           # t_a, t_b are functions of [x, h]
h_new = ff1 · (1 − t) + t · ff2
```

This has less inductive bias toward decay and more freedom in trajectory shape.
`--multi-tau` and `--ode-solver` are ignored in this mode (there is no `τ` or
ODE); `--ode-unfolds > 1` is allowed and composes interpolation over smaller
`dt`. Initializing `time_net.bias = 0` gives the neutral starting value
`t = 0.5`.

### Cell output and state normalization

The new layer state is `LayerNorm(Dropout(h))`. Thus the state carried between
time steps is **normalized**, deliberately stabilizing dynamics on long
sequences.

**Zoneout (`--zoneout`).** During training, some output elements are replaced
with the normalized previous state, regularizing abrupt state changes.

### NCP wiring (`--wiring ncp`)

Neurons are split into **inter**, **command**, and **motor** groups (fractions
come from `--ncp-command-frac` and `--ncp-motor-frac`; the remainder is inter).
Only these paths are allowed:

```
inter   → inter      (recurrent)
inter   → command
command → command    (recurrent)
command → motor
motor   → command    (feedback)
```

Each path is a physically smaller `nn.Linear`, not a mask over a dense
`H×H` matrix. A dense matrix with a mask does not speed up PyTorch matmul; here
nonexistent connections are absent from both weights and computation.

The startup log prints the fraction of the dense recurrent part, for example:
`ncp (recurrent part ~46% of dense by parameters/FLOPs)`.

The original paper's biophysical synapse polarity (reversal potentials and
excitatory/inhibitory connections) is not reproduced. Gradient descent learns
the sign and magnitude of permitted connections. In `cfc` mode, `ff1_h` and
`ff2_h` are independent instances with the same topology. Time gating
(`tau_net` / `time_net`) remains dense.

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

### ChunkMemory (`--memory-slots N`)

An additional slow memory channel on top of `--state-carry`:

- At the end of each sub-chunk, the final state of the last layer is placed in
  an `N`-slot FIFO buffer.
- At the beginning of the next sub-chunk, dot-product attention reads the
  buffer and mixes the result into the last-layer state through a learned gate.
- The buffer stores **raw** states; `summarize` is applied lazily on read so it
  receives gradients.
- Between sub-chunks the buffer is detached, like `hidden`: no gradient crosses
  the chunk boundary. The read/write policy is learned inside each sub-chunk.
- It works only with `--state-carry ≥ 2`.
- It is used only during training and validation; `generate()` and chat do not
  connect this memory.

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

Cross-entropy for the next token, averaged over **valid** tokens. Each
sub-chunk loss is divided by `state_carry × accumulation_group_size`, including
the final incomplete group of an epoch, so contributions are averaged correctly.

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
- embeddings, output layer, biases, LayerNorm, `h0`, `tau_mix`, the 3D
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
Modes are `default`, `reduce-overhead` (CUDA Graphs), and `max-autotune` (not
recommended for these small matrices because it is slow and often provides no
benefit). If compilation fails, the program falls back to `default` and then
eager mode. For `--generate-only`, compilation uses evaluation mode.

### Gradient checkpointing (`--grad-checkpoint`)

Checkpointing is enabled for every cell call. The granularity is very fine, so
overhead may exceed savings: TBPTT memory is primarily the O(T) activations
across the window. Measure on your own dimensions before relying on it.

### Distillation (`--teacher-checkpoint`)

`loss = (1−α)·CE + α·T²·KL(student‖teacher)`, where `α = --kd-alpha` and
`T = --kd-temperature`. Teacher and student vocabularies must match. The teacher
is called **without carrying hidden state** between sub-chunks, so its soft
labels are less informative for the second and later sub-chunks.

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

Loss is checked for NaN/Inf (fp16 skips are handled by GradScaler), gradients
can be clipped by norm (`--grad-clip`, 0 disables clipping), and divergence
causes a clear error.

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

`--kernel triton` replaces the LTC sub-step tail (per-bank softplus, mixing,
`exp`, and the `h` update) with one fused kernel instead of about eight
elementwise PyTorch operations. Matrix multiplications remain on cuBLAS.

Requirements:

- **Inference only** (`generate`, chat, `evaluate`); the kernel has no backward.
- CUDA and an installed `triton`.
- `--gate-mode ltc --ode-solver euler`.

Otherwise the program prints the reason and stays on the PyTorch path. Check
correctness and speed on your GPU first:

```bash
python liquid_text_model.py --selftest-triton
```

The test compares the kernel with PyTorch (single and multi-tau modes,
non-uniform bank weights, and “hard” inputs), uses tolerance `1e-4`, and
measures step time.

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
| `--state-carry` | `2` | Sub-chunks carrying state |
| `--grad-accumulation` | `4` | Gradient accumulation |
| `--grad-clip` | `1.0` | Maximum gradient norm (0 disables it) |

### Architecture

| Flag | Default | Description |
|---|---|---|
| `--embedding-dim` | `384` | Embedding size |
| `--hidden-size` | `384` | Hidden-state size |
| `--num-layers` | `2` | Number of liquid layers |
| `--ode-unfolds` | `1` | Sub-steps per tick |
| `--dropout` | `0.05` | Dropout, `[0, 1)` |
| `--zoneout` | `0.1` | Zoneout, `[0, 1)` |
| `--tie-weights` / `--no-tie-weights` | enabled | Tie embedding and output |
| `--mlp-head` / `--no-mlp-head` | disabled | MLP head |
| `--residual` / `--no-residual` | enabled | Residual connections |
| `--multi-tau` / `--no-multi-tau` | enabled | Three time-constant banks |
| `--local-conv` / `--no-local-conv` | enabled | Causal `k=5` convolution |
| `--gate-mode` | `ltc` | `ltc` or `cfc` |
| `--ode-solver` | `euler` | `euler` or `heun` |
| `--wiring` | `dense` | `dense` or `ncp` |
| `--ncp-command-frac` | `0.4` | Fraction of command neurons |
| `--ncp-motor-frac` | `0.2` | Fraction of motor neurons |
| `--memory-slots` | `0` | ChunkMemory slots (0 disables it) |
| `--kernel` | `torch` | `torch` or `triton` (inference) |
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

**More accurate LTC integration**

```bash
python liquid_text_model.py --data data --ode-solver heun --ode-unfolds 2
```

**Lightweight recurrent part (NCP)**

```bash
python liquid_text_model.py --data data --wiring ncp \
  --ncp-command-frac 0.4 --ncp-motor-frac 0.2
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
python liquid_text_model.py --data data --state-carry 4 --memory-slots 4
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
(`embedding_dim`, `hidden_size`, `gate_mode`, `wiring`, `memory_slots`, and
others) also come from the checkpoint. Differences from the command line are
printed as `[RESUME] … (from checkpoint)`. The tokenizer is restored rather
than retrained. A changed corpus produces a warning.

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

Checkpoints from older versions load with historical defaults for missing
architecture parameters (`gate_mode=ltc`, `ode_solver=euler`, `wiring=dense`,
`memory_slots=0`, and so on). New weights (`h0`, `tau_mix`, convolution) are
initialized again with a message. Early versions used a **non-causal**
convolution, so old weights with `local_conv` should be retrained.

The v3.6 BPE tokenizer differs from earlier versions (no prefix whitespace and
a complete byte alphabet). Old checkpoints continue to work with their own
tokenizer, but retrain from scratch for fair experiment comparisons.

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
6. Try `--wiring ncp`.
7. Use `--grad-checkpoint` as a last resort and measure it.

On Linux, `expandable_segments` is already enabled for the CUDA allocator to
reduce fragmentation.

### If training is slow

The model is recurrent, so the time loop is sequential:

- use `--compile`, especially `--compile-mode reduce-overhead`;
- increase `--batch-size` (the recurrent step is nearly batch-size independent
  on GPU);
- use bf16/fp16 instead of fp32;
- use `--ode-unfolds 1` and `--ode-solver euler` (Heun and additional sub-steps
  cost more);
- monitor `tok/s` in the epoch log.

### Rules of thumb

- `--state-carry 2` with `--seq-len 128` gives an effective 256-token training
  context. Increase `--state-carry` for longer dependencies and try
  `--memory-slots`.
- Start AdamW at `--lr 2e-3`; if it diverges, lower to `1e-3` and strengthen
  `--grad-clip`.
- If validation loss rises while training loss falls, increase data,
  `--dropout`, `--zoneout`, or `--weight-decay`.
- For small (single-digit MB) corpora use `--vocab-size` 4000–8000; for large
  and dialogue corpora use 16000+. The program reduces an oversized vocabulary
  automatically on small corpora.
- Logged perplexity is `exp(loss)`, capped at `exp(20)`.

## Known limitations

- **ChunkMemory is not used for generation or chat.** A model trained with
  `--memory-slots > 0` runs without it at inference, creating a small
  train/inference mismatch.
- Distillation calls the teacher without carrying state between sub-chunks.
- The Triton kernel supports inference and `ltc + euler` only.
- NCP reproduces connection topology, not the original paper's synapse
  biophysics; ChunkMemory follows the general M-CfC idea, not its formulas.
- Cell-level gradient checkpointing may not save memory.
- Sequential processing is slower than a transformer on long training batches
  (but faster during generation).
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
liquid_text_model.py          # thin entry point: python liquid_text_model.py ...
liquid_lm/
├── __init__.py               # sets PYTORCH_CUDA_ALLOC_CONF before torch is imported
├── __main__.py               # python -m liquid_lm ...
├── cli.py                    # main(): argument validation and mode dispatch
├── args.py                   # build_parser()
├── constants.py              # supported suffixes, excluded dirs, text encodings
├── device.py                 # set_seed, choose_device, configure_tensor_cores,
│                             #   print_device_info, cuda_memory_report
├── checkpoint.py             # save_checkpoint, load_checkpoint_raw,
│                             #   print_checkpoint_info, set_trust_checkpoints
├── train_logger.py           # TrainLogger (TensorBoard / Weights & Biases)
├── inference.py              # run_generate_only, run_chat_repl
├── selftest.py               # selftest_triton_tail
├── data/
│   ├── tags.py               # dialogue tags, role and question-answer key tables
│   ├── readers.py            # clean_text, dialogue_obj_to_text, json_to_text,
│   │                         #   jsonl_to_text, table readers, read_text_robust
│   ├── corpus.py             # load_texts
│   ├── tokenization.py       # CharTokenizer, BPETokenizer, build_tokenizer,
│   │                         #   tokenizer_from_checkpoint
│   └── batching.py           # get_batch, build_dialogue_anchors, build_train_mask
├── model/
│   ├── liquid_lm.py          # LiquidLanguageModel (including generate)
│   ├── liquid_cell.py        # LiquidCell (ltc / cfc gates, dense / ncp wiring)
│   ├── ncp.py                # NCPRecurrent
│   ├── chunk_memory.py       # ChunkMemory and the chunk_memory_* helpers
│   ├── triton_kernel.py      # _ltc_tail_kernel and its wrapper
│   ├── dropout.py            # LockedDropout
│   ├── ema.py                # EMA
│   ├── config.py             # ARCH_KEYS, LEGACY_ARCH_DEFAULTS,
│   │                         #   create_model_from_config, load_model_state_compat
│   └── utils.py              # get_base_model, detach_hidden, EMA/backup helpers
└── training/
    ├── trainer.py            # run_training
    ├── evaluate.py           # evaluate
    ├── optim.py              # Muon, MuonAdamWHybrid, MultiScheduler,
    │                         #   make_scheduler, make_optimizer
    ├── losses.py             # masked_cross_entropy
    └── amp.py                # resolve_amp_dtype, make_grad_scaler, autocast_context
```

Run all commands from the repository root. The optional dependencies
(`pandas`, `charset-normalizer`, `tokenizers`, `tensorboard`, `wandb`,
`triton`) are imported with a fallback in the module that uses them, so a
missing package never breaks unrelated parts of the program.

### Adding a custom architecture option

1. Add the argument to `build_parser()` in `liquid_lm/args.py`.
2. Implement it in `LiquidLanguageModel` / `LiquidCell`
   (`liquid_lm/model/`), and pass it through where the model is built in
   `run_training` (`liquid_lm/training/trainer.py`).
3. Add its key to `ARCH_KEYS` and a default for old checkpoints to
   `LEGACY_ARCH_DEFAULTS` in `liquid_lm/model/config.py`, so `--resume` and
   old files remain compatible.
4. Pass the parameter to `create_model_from_config` (same file).

## Change history

The project is currently **v3.6**. See the repository history and checkpoint
compatibility notes above for details of behavior that changed between versions.

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
