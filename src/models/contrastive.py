"""Supervised contrastive loss (Khosla et al. 2020), for use alongside (not instead
of) a classification cross-entropy loss -- see models/span_type_classifier.py's
ContrastiveSpanTypeTrainer.

Known limitation, disclosed rather than silently absorbed: this loss only sees
in-batch positives. CO/ST have only 38-50 total span instances across the whole
612-paragraph corpus, so at ordinary batch sizes many batches will contain zero
same-label pairs for those two labels, and this loss contributes nothing for them
on those batches. No cross-batch memory bank or class-balanced sampler is
implemented here -- a deliberate scope cut for a first pass, not an oversight.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """embeddings: (B, D). labels: (B,) integer class ids.

        Returns a scalar 0.0 (not NaN) if no anchor in the batch has any in-batch
        positive (e.g. batch size 1, or every label in the batch is unique).
        """
        n = embeddings.shape[0]
        device = embeddings.device
        if n < 2:
            return embeddings.new_zeros(())

        z = F.normalize(embeddings, dim=-1)
        sim = (z @ z.T) / self.temperature  # (B, B)

        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        positive_mask = same_label & ~self_mask

        has_positive = positive_mask.any(dim=1)  # (B,)
        if not has_positive.any():
            return embeddings.new_zeros(())

        # log-sum-exp over all other samples (denominator), self excluded.
        sim_masked = sim.masked_fill(self_mask, float("-inf"))
        log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)  # (B, 1)
        log_prob = sim - log_denom  # (B, B)

        # mean log-prob over each anchor's positives, anchors with none excluded.
        pos_count = positive_mask.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_count

        return -mean_log_prob_pos[has_positive].mean()
