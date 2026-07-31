"""CRF-wrapped token classifier, shared by Task 2's Track A (13-tag BIO) and Track B's
Stage A boundary tagger (3-tag O/B-ARG/I-ARG) -- parameterized purely by num_labels, no
subclassing needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, Trainer


class TokenClassifierWithCRF(nn.Module):
    def __init__(self, backbone_id_or_path: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_id_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # **kwargs absorbs anything a tokenizer/collator might add (e.g. token_type_ids) that this head doesn't use.
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        mask = attention_mask.bool()

        if labels is not None:
            # CRF requires non-negative tag ids and the first timestep to be unmasked;
            # -100 (ignored positions, incl. special tokens) is remapped to 0 ("O") and excluded via `mask` instead.
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            log_likelihood = self.crf(emissions, safe_labels, mask=mask, reduction="mean")
            loss = -log_likelihood
            return {"loss": loss, "logits": emissions}
        return {"logits": emissions}

    def decode(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(outputs.last_hidden_state)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask), emissions


def bio_class_weights_from_span_weights(span_weights: torch.Tensor, labels: list[str], bio2id: dict[str, int]) -> torch.Tensor:
    """Expands models/span_scorer.py's span_class_weights (indexed ["O"] + labels,
    length len(labels)+1) into a BIO_TAGS-indexed tensor (length 2*len(labels)+1):
    "O" keeps span_weights[0], both "B-<label>" and "I-<label>" get the same
    per-label weight span_weights[i+1] -- the CRF's tag-transition structure
    already distinguishes B/I, this only weights which TYPE of content a tag is."""
    weights = torch.ones(len(bio2id), dtype=span_weights.dtype)
    weights[bio2id["O"]] = span_weights[0]
    for i, label in enumerate(labels):
        weights[bio2id[f"B-{label}"]] = span_weights[i + 1]
        weights[bio2id[f"I-{label}"]] = span_weights[i + 1]
    return weights


class WeightedCRFTrainer(Trainer):
    """TokenClassifierWithCRF's own loss (self.crf's sequence log-likelihood) has no
    per-tag class weighting -- rare labels (CO/ST) get diluted gradient signal
    purely from rarity. Adds a second, per-tag-weighted token-level cross-entropy
    term over the SAME emissions already computed in one forward pass (no extra
    encoder pass) alongside the CRF's own loss, rather than replacing it."""

    def __init__(self, *args, bio_class_weights: torch.Tensor, aux_ce_weight: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.bio_class_weights = bio_class_weights
        self.aux_ce_weight = aux_ce_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        emissions = outputs["logits"]

        aux_ce = nn.functional.cross_entropy(
            emissions.view(-1, emissions.size(-1)),
            labels.view(-1),
            weight=self.bio_class_weights.to(emissions.device),
            ignore_index=-100,
        )
        loss = outputs["loss"] + self.aux_ce_weight * aux_ce
        return (loss, outputs) if return_outputs else loss
