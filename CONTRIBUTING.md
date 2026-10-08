# Contributing

Thank you for your interest in the project! Changes, ideas, and experiment
results are welcome.

## Project layout

The code lives in the `liquid_lm` package (`model/`, `data/`, `training/`, plus
the CLI and inference modules), launched through the thin entry point
`liquid_text_model.py`. See the "Code structure" section of the README for a
file-by-file map. Run all commands from the repository root.

## Most useful contributions right now

1. **A/B benchmarks.** The `cfc`, `ncp`, `--memory-slots`, `--ode-solver heun`,
   and `--kernel triton` modes have not yet been compared with the baseline
   `ltc + euler + dense`. Running two configurations on the same data with the
   same `--seed`, then comparing `val loss` and speed, would be especially
   valuable. Please include the commands used, hardware, and PyTorch version.
2. **Tests.** There are currently no automated tests (a top-level `tests/`
   directory would be a good home for them). Unit tests are especially needed
   for the cell mathematics: equivalence between `_ltc_step` and the Triton
   tail (`liquid_lm/model/liquid_cell.py`, `liquid_lm/model/triton_kernel.py`),
   equivalence between block-based `NCPRecurrent` and a dense matrix with zero
   blocks (`liquid_lm/model/ncp.py`), and the FIFO behavior of `ChunkMemory`
   (`liquid_lm/model/chunk_memory.py`).
3. **Testing on other hardware and PyTorch versions**, especially
   `--kernel triton` and `--compile`.
4. **Ideas from the list below.**

## Development ideas

- BitNet b1.58-style weight quantization (ternary weights and
  quantization-aware training).
- Mixture-of-Experts over the readout or liquid layers.
- Sparse (NCP) input projections; currently the block structure is applied
  only to the recurrent matrix.
- Using `ChunkMemory` in `generate()` and chat.
- Batched generation.
- Hybrid liquid layers with a small number of attention blocks.

## How to propose a change

1. Open an Issue describing the idea or bug; for larger changes, it is better
   to discuss them in advance.
2. Fork the repository and create a separate branch from `main`.
3. Preserve backward compatibility: new options should be **disabled** by
   default, and architecture parameters must be added to `ARCH_KEYS` /
   `LEGACY_ARCH_DEFAULTS` (in `liquid_lm/model/config.py`) so that old
   checkpoints continue to load. The README section "Adding a custom
   architecture option" lists every place that needs to change.
4. In the Pull Request, state what changed, why, and how it was tested
   (command and result). If you claim a speed or quality improvement, include
   measurements.
5. Do not commit checkpoints, datasets, or logs (see `.gitignore`).

## Bug reports

Include the command used to run the program, the complete traceback, Python /
PyTorch / CUDA versions, and, if possible, a minimal reproducible example.

By submitting a contribution, you agree that it will be distributed under the
project license (Apache License 2.0).
