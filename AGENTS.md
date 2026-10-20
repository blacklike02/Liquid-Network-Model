# Working on Echo (Liquid LM)

This file applies to the entire repository. Follow the user's current request
when it changes the scope or decisions recorded here. Read `README.md` and the
relevant parts of `CONTRIBUTING.md` before changing behavior.

## Project intent and communication

- The chosen model name is **Echo**. Its planned family is **Pantheon**, with
  progression **Echo -> Iris -> Hermes -> Athena -> Hera**. See `NAMING.md` for
  naming/version rules. Future tiers are plans, not implemented capabilities.
  Preserve existing package, entry-point and checkpoint names when applying
  the public model name.
- This is an experimental liquid recurrent language model, not a standard
  Transformer. Preserve the liquid architecture and existing behavior when
  fixing bugs or optimizing computation. Present new architecture proposals
  separately from maintenance changes.
- The project owner communicates in Russian. Public documentation, code
  comments, commit messages and PR descriptions should be in English. Keep
  `README.ru.md` consistent with the English README when behavior changes.
- Held-out quality matters more than training time. A slower architecture can
  be worthwhile when controlled evaluation demonstrates better quality.
  A slower implementation of identical equations does not add model capacity.
- There is no fixed parameter-count ceiling. The earlier 5M target was a small
  experiment, not a project limit. Use inexpensive prototypes to isolate ideas,
  then select scale based on quality, available data, memory and compute.
- Do not describe synthetic recall accuracy as conversational intelligence.
  Select checkpoints using validation data before evaluating a separate test;
  compare multiple seeds and disclose unequal parameter, token or time budgets.

## Code map

- `liquid_text_model.py`, `liquid_lm/__main__.py`: CLI entry points.
- `liquid_lm/cli.py`: arguments and execution dispatch.
- `liquid_lm/model.py`: model, generation, architecture metadata, checkpoint I/O.
- `liquid_lm/layers.py`: LTC/CfC cells, RMSNorm, FFNs, chunk memory, Triton kernels.
- `liquid_lm/associative.py`: experimental innovation-gated associative memory.
- `liquid_lm/training.py`, `optim.py`: training, evaluation, KD, optimizers.
- `liquid_lm/data.py`, `tokenizer.py`: corpus loading, split policy, tokenization.
- `liquid_lm/inference.py`: generation and interactive chat.
- `tests/`: unittest regression and optional CUDA/Triton checks.
- `benchmark_*.py`, `research_liquid.py`: reproducible measurements and experiments.

## Architecture and checkpoint invariants

- **RMSNorm is the only supported normalization**, in cells, FFNs and the final
  readout. `--norm-type` was removed. The Python `norm_type` keyword accepts
  only `rms`; new checkpoint configs explicitly record that value.
- LayerNorm checkpoints and configs without normalization metadata are rejected
  for resume and generation. Use an earlier compatible revision for those
  weights. Never relabel metadata or discard biases to claim a conversion.
  The `layer` entry in `LEGACY_ARCH_DEFAULTS` records historical semantics;
  it is not permission to restore LayerNorm or reinterpret missing metadata.
  The retained English 4.54M and Shakespeare checkpoints were trained with
  LayerNorm; they are historical artifacts, not current RMSNorm baselines.
- NCP wiring, FIM augmentation, Heun and configurable ODE substeps were removed.
  Compatibility arguments accept only dense wiring, Euler and one unfold.
  Reject retired equations and nonzero FIM resume recipes explicitly. Keep
  reserved historical FIM token IDs; removing them would change token meanings.
- CLI defaults for new training are three chunk-memory slots, projection width
  64 and four carried chunks. Python model constructors default to memory off.
  Preserve this distinction. For supported RMS checkpoints, absent/zero
  `memory_dim` means the legacy full-width memory layout.
- Compact memory stores detached raw chunk-final states plus occupancy flags.
  Unwritten slots must not affect reads; an empty buffer is an exact identity.
  Carrying more chunks extends forward context, not the gradient horizon.
- Keep convolution causal and carry its tail across chunks. Preserve streaming,
  full-sequence, sequence-batched and activation-checkpointed equivalence.
- Keep tokenizer IDs, saved architecture settings, data split semantics and
  checkpoint tensor meanings stable. Integrate architecture fields through
  `ARCH_KEYS`, defaults, CLI, training and save/load paths where applicable.
  Reject unsupported semantics instead of silently loading different equations.
- The existing Triton path accelerates LTC operations and supports first-order
  training gradients. CfC uses the Torch path. Do not assume a custom kernel is
  faster: compare outputs, gradients and synchronized full-model timings.

## Verification

Run commands from the repository root. Check the interpreter before GPU work:
the local `.venv` was CPU-only, while system Python had CUDA-enabled PyTorch.
Do not assume this remains true on another machine.

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:TRITON_CACHE_DIR=Join-Path (Get-Location) 'runs/agent_checks/cache/triton'
$env:TORCHINDUCTOR_CACHE_DIR=Join-Path (Get-Location) 'runs/agent_checks/cache/inductor'
python -m unittest discover -s tests -v
git diff --check
```

- Use a fresh task-specific output directory instead of overwriting earlier
  measurements. Keep compiler caches inside the writable workspace.
- Run checks appropriate to the change. Report passes, skips, environment and
  scope; GPU skips do not establish GPU correctness. Do not rerun expensive
  training or benchmarks for documentation-only edits.
- For LTC kernel changes, also run `tests/check_triton_mixed.py` separately;
  it is not included in unittest discovery. See CONTRIBUTING for benchmark
  commands. Run GPU workloads sequentially to avoid distorting timing.
- Separate warm execution from compilation, loading and optimizer overhead.
  Report which costs are included. Prefer full-model results to microkernel
  speed claims and do not compare unlocked-clock sessions as a clean speedup.
- Reference results on **20 October 2026**: 118/118 CUDA/Triton tests passed on
  RTX 5050 Laptop with PyTorch 2.11.0+cu128; CPU-only passed 98 with 20 expected
  skips on PyTorch 2.14.1+cpu. Six BPE CLI train/resume/generate scenarios passed
  across CPU and CUDA BF16/Triton. These are historical results, not evidence
  that a later change was tested. Earlier LayerNorm results are historical too.

## Local files, cleanup and handoff

- `docs/` is the owner's private working research folder, intentionally ignored.
  It is not disposable. `runs/` contains trained weights and experiment records.
  Preserve both, along with the installed environment and editor settings.
- Do not overwrite or delete user checkpoints, datasets or reports during
  cleanup. Remove identifiable task-owned temporary outputs only; verify exact
  resolved paths stay inside the workspace before recursive deletion, and
  reject reparse points or symlinks in deletion targets.
- Keep datasets, checkpoints, caches and raw logs out of Git. Public README and
  CONTRIBUTING must not depend on links to ignored local reports. Important
  public results need their commands, conditions and limitations in public docs.
- CLI training/generation can write `generated.txt` and `training_info.json` to
  the working directory. Run smoke tests from a temporary directory using an
  absolute script path, and clean their corpora, weights and caches afterward.
- Record substantial experiments and fixes in an English report under `docs/`;
  place proposed next experiments at its end. If that folder is absent in a
  fresh clone, do not assume its private scripts or data are available.
- `COMMIT_MESSAGE.txt` is an ignored English draft. Update it when preparing a
  commit, but do not commit, push or publish unless the user requests it.
- Preserve unrelated working-tree changes. Do not reset or stage the owner's
  work as part of inspection, testing or cleanup.
