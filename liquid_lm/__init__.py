"""Liquid RNN language model (LTC/CfC) — package."""

import os

# Must run before the first torch import, which is why it lives here.
if os.name == "posix":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
