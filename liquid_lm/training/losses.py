"""Loss functions."""

from __future__ import annotations

import torch.nn.functional as F

from ..data.batching import IGNORE_INDEX


def masked_cross_entropy(logits, targets):
    valid_count = (targets != IGNORE_INDEX).sum()
    sum_loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, reduction="sum")
    loss = sum_loss / valid_count.clamp(min=1)
    return loss, sum_loss, valid_count
