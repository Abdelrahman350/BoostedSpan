"""CRF-wrapped token classifier, shared by Task 2's Track A (13-tag BIO) and Track B's
Stage A boundary tagger (3-tag O/B-ARG/I-ARG) -- parameterized purely by num_labels, no
subclassing needed.
"""

from __future__ import annotations

import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel


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
