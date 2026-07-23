"""Task 1 losses: weighted BCE (baseline) and Asymmetric Loss (boosted).

Both expose the same (logits, targets) -> loss interface via get_task1_loss_fn's
factory, so the training script never needs to branch on which one it got.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Ridnik et al. asymmetric loss with margin-clipped negatives.

    L = -y(1-p)^gamma_pos log(p) - (1-y) p_m^gamma_neg log(1-p_m), p_m = max(p - clip, 0)
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_pos = p.clamp(min=self.eps, max=1 - self.eps)
        p_neg = (p - self.clip).clamp(min=0.0).clamp(max=1 - self.eps)

        loss_pos = targets * torch.log(p_pos) * torch.pow(1 - p_pos, self.gamma_pos)
        loss_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=self.eps)) * torch.pow(p_neg, self.gamma_neg)
        return -(loss_pos + loss_neg).mean()


def weighted_bce_pos_weight(rows: list[dict], labels: list[str], clip: float = 8.0) -> torch.Tensor:
    """Inverse label frequency, clipped at `clip`x, for nn.BCEWithLogitsLoss(pos_weight=...)."""
    n_total = len(rows)
    weights = []
    for label in labels:
        n_pos = sum(1 for r in rows if label in r["labels"])
        n_pos = max(n_pos, 1)  # guard against a label with zero positives in this split
        weights.append(min(clip, (n_total - n_pos) / n_pos))
    return torch.tensor(weights, dtype=torch.float32)


def get_task1_loss_fn(
    loss_name: str,
    pos_weight: torch.Tensor | None = None,
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Factory: config.model.loss selects the loss, callers never see which class it is."""
    if loss_name == "asymmetric":
        asl = AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
        return lambda logits, targets: asl(logits, targets)
    if loss_name == "weighted_bce":
        # pos_weight is a plain tensor, not a module buffer -- it won't follow the
        # model to its training device (e.g. cuda) automatically, so move it at each
        # call rather than baking a fixed-device BCEWithLogitsLoss into the closure.
        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            pw = pos_weight.to(logits.device) if pos_weight is not None else None
            return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

        return loss_fn
    raise ValueError(f"Unknown loss: {loss_name!r} (expected 'asymmetric' or 'weighted_bce')")
