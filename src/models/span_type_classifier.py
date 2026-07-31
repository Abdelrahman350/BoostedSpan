"""Track B Stage B: span-type classifier.

Shared encoder, mean-pools token representations inside a gold span, concatenates with
the [CLS] representation, linear layer over the 6 ADU types. Trained via teacher
forcing on gold span offsets with inverse-frequency class-weighted cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, Trainer

from models.contrastive import SupConLoss
from text.cues import build_input_text


class SpanTypeClassifier(nn.Module):
    def __init__(self, backbone_id_or_path: str, num_types: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_id_or_path)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden * 2, num_types)  # [CLS] repr concat mean-pooled span repr

    def forward(self, input_ids, attention_mask, span_start, span_end, type_labels=None, class_weights=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_repr = hidden_states[:, 0]
        span_reprs = []
        for i in range(hidden_states.shape[0]):
            s, e = span_start[i].item(), max(span_end[i].item(), span_start[i].item() + 1)
            span_reprs.append(hidden_states[i, s:e].mean(dim=0))
        span_repr = torch.stack(span_reprs, dim=0)
        combined_repr = self.dropout(torch.cat([cls_repr, span_repr], dim=-1))
        logits = self.classifier(combined_repr)

        loss = None
        if type_labels is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights)(logits, type_labels)
        return {"loss": loss, "logits": logits, "combined_repr": combined_repr}


def char_span_to_token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int):
    """(None, None) if the span fell entirely outside a truncated window."""
    tok_start = tok_end = None
    for ti, (ts, te) in enumerate(offsets):
        if ts == te:
            continue
        if ts < char_end and te > char_start:
            if tok_start is None:
                tok_start = ti
            tok_end = ti
    return tok_start, tok_end


def build_span_type_examples(tokenizer, rows: list[dict], label2id: dict[str, int], max_len: int) -> list[dict]:
    examples = []
    for r in rows:
        text = build_input_text(r["text"], r["type"])
        prefix_len = len(text) - len(r["text"])
        enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True, padding=False)
        offsets = enc["offset_mapping"]
        for s in r["labels"]:
            tok_start, tok_end = char_span_to_token_span(
                offsets, s["start_offset"] + prefix_len, s["end_offset"] + prefix_len
            )
            if tok_start is None:
                continue  # span truncated out of the max_len window (rare, long-paragraph tail)
            examples.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "span_start": tok_start,
                    "span_end": tok_end + 1,
                    "type_labels": label2id[s["label"]],
                }
            )
    return examples


def make_span_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = nn.utils.rnn.pad_sequence(
            [torch.tensor(b["input_ids"]) for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            [torch.tensor(b["attention_mask"]) for b in batch], batch_first=True, padding_value=0
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_start": torch.tensor([b["span_start"] for b in batch]),
            "span_end": torch.tensor([b["span_end"] for b in batch]),
            "type_labels": torch.tensor([b["type_labels"] for b in batch]),
        }

    return collate_fn


class ClassWeightedSpanTrainer(Trainer):
    """Injects class_weights (bound via constructor, not a module-level global) into
    SpanTypeClassifier.forward at every compute_loss call."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        type_labels = inputs.pop("type_labels")
        outputs = model(**inputs, type_labels=type_labels, class_weights=self.class_weights.to(model.classifier.weight.device))
        loss = outputs["loss"]
        # combined_repr is an internal-only key (used by ContrastiveSpanTypeTrainer
        # below): Trainer.prediction_step gathers every non-"loss" key of a dict
        # `outputs` into the eval `logits` tuple it hands to compute_metrics, so
        # leaving this in would silently turn compute_span_type_metrics' `logits,
        # labels = eval_pred` into unpacking a 2-tuple instead of the logits array.
        outputs.pop("combined_repr", None)
        return (loss, outputs) if return_outputs else loss


class ContrastiveSpanTypeTrainer(ClassWeightedSpanTrainer):
    """ClassWeightedSpanTrainer + a supervised contrastive term (models/contrastive.py)
    over combined_repr, the same pre-classifier representation the CE loss is computed
    from. See models/contrastive.py's docstring for the known in-batch-positives
    limitation on CO/ST."""

    def __init__(self, *args, contrastive_weight: float, contrastive_temperature: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.supcon = SupConLoss(temperature=contrastive_temperature)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        type_labels = inputs.pop("type_labels")
        outputs = model(**inputs, type_labels=type_labels, class_weights=self.class_weights.to(model.classifier.weight.device))
        contrastive_loss = self.supcon(outputs["combined_repr"], type_labels)
        loss = outputs["loss"] + self.contrastive_weight * contrastive_loss
        outputs.pop("combined_repr", None)  # see ClassWeightedSpanTrainer.compute_loss's comment
        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def predict_span_types(text: str, domain: str, spans: list[dict], model, tokenizer, labels: list[str], max_len: int) -> list[dict]:
    if not spans:
        return []
    full_text = build_input_text(text, domain)
    prefix_len = len(full_text) - len(text)
    enc = tokenizer(full_text, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    device = next(model.parameters()).device
    input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    typed = []
    for s in spans:
        tok_start, tok_end = char_span_to_token_span(
            offsets, s["start_offset"] + prefix_len, s["end_offset"] + prefix_len
        )
        if tok_start is None:
            continue  # span fell outside the truncated window
        span_start_t = torch.tensor([tok_start]).to(device)
        span_end_t = torch.tensor([tok_end + 1]).to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, span_start=span_start_t, span_end=span_end_t)
        pred_label = labels[out["logits"].argmax(dim=-1).item()]
        typed.append({"label": pred_label, "start_offset": s["start_offset"], "end_offset": s["end_offset"]})
    return typed


@torch.no_grad()
def retype_spans_with_confidence(
    text: str, domain: str, spans: list[dict], model, tokenizer, labels: list[str], max_len: int, confidence_threshold: float
) -> list[dict]:
    """Like predict_span_types, but spans already carry a label (e.g. from a CRF
    ensemble's boundary+type decode) -- only overrides it with this classifier's
    own prediction when that prediction's softmax confidence clears
    confidence_threshold, otherwise keeps the original label unchanged. A span
    that falls outside a truncated window (tok_start is None) also keeps its
    original label -- unlike predict_span_types, which can afford to just drop an
    untyped span, here the span already has a usable label worth preserving."""
    if not spans:
        return []
    full_text = build_input_text(text, domain)
    prefix_len = len(full_text) - len(text)
    enc = tokenizer(full_text, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    device = next(model.parameters()).device
    input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    retyped = []
    for s in spans:
        tok_start, tok_end = char_span_to_token_span(
            offsets, s["start_offset"] + prefix_len, s["end_offset"] + prefix_len
        )
        if tok_start is None:
            retyped.append(dict(s))
            continue
        span_start_t = torch.tensor([tok_start]).to(device)
        span_end_t = torch.tensor([tok_end + 1]).to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, span_start=span_start_t, span_end=span_end_t)
        probs = out["logits"].softmax(dim=-1)[0]
        best_idx = int(probs.argmax().item())
        best_conf = probs[best_idx].item()
        new_label = labels[best_idx] if best_conf >= confidence_threshold else s["label"]
        retyped.append({"label": new_label, "start_offset": s["start_offset"], "end_offset": s["end_offset"]})
    return retyped
