"""Task 2 span_scorer variant: enumerate-and-classify span scoring, an alternative to
BIO+CRF decode (models/crf_tagger.py).

Scores ALL enumerated candidate (start, end) spans up to max_span_width in one forward
pass, via the same mean-pool-span-tokens + concat-[CLS] + linear pattern already
proven in models/span_type_classifier.py's SpanTypeClassifier -- except that classifier
only ever scores one gold span per example (teacher forcing); this one scores every
candidate at once, so span pooling must be vectorized (pool_span_means's cumsum trick)
rather than a per-span Python loop.

Unlike BIO+CRF's single tag sequence, candidate spans carry independent scores, so
inference-time decode (decode_spans_from_scores) can represent overlapping spans of
different labels -- the ~1% of paragraphs with overlapping gold spans that BIO cannot
represent at all (see CLAUDE.md section 3). Output is the same generic
{"label", "start_offset", "end_offset"} span dicts as the CRF path, so
postprocessing/spans.py, evaluation/scoring.py, and evaluation/submission.py all work
unmodified.
"""

from __future__ import annotations

import collections
import random

import torch
import torch.nn as nn
from transformers import AutoModel, Trainer

from data.loading import LABELS
from models.span_type_classifier import char_span_to_token_span
from text.cues import build_input_text

# id 0 = null/background ("not a real span"), ids 1..6 = LABELS in order.
SPAN_SCORER_LABELS = ["O"] + LABELS
span_label2id = {label: i for i, label in enumerate(SPAN_SCORER_LABELS)}


def enumerate_candidate_spans(seq_len: int, max_width: int) -> list[tuple[int, int]]:
    """All (start, end) with 0 <= start < end <= seq_len and end - start <= max_width,
    end EXCLUSIVE. Pure/combinatorial -- callers offset into the full tokenized
    sequence (e.g. skipping CLS/SEP) themselves.
    """
    candidates = []
    for start in range(seq_len):
        max_end = min(start + max_width, seq_len)
        for end in range(start + 1, max_end + 1):
            candidates.append((start, end))
    return candidates


def pool_span_means(hidden_states: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Vectorized mean-pool of hidden_states[:, start:end] for every (start, end) in
    starts/ends, via a cumsum prefix-sum trick -- turns "loop over N candidate spans"
    into a handful of gather/subtract ops on one (B, L, H) tensor.

    hidden_states: (B, L, H). starts, ends: (B, N) long tensors, end EXCLUSIVE.
    Returns (B, N, H).
    """
    B, L, H = hidden_states.shape
    zeros = hidden_states.new_zeros(B, 1, H)
    cs = torch.cat([zeros, hidden_states.cumsum(dim=1)], dim=1)  # (B, L+1, H); cs[:, i] = sum of [:i]

    start_idx = starts.unsqueeze(-1).expand(-1, -1, H)
    end_idx = ends.unsqueeze(-1).expand(-1, -1, H)
    sum_start = torch.gather(cs, 1, start_idx)
    sum_end = torch.gather(cs, 1, end_idx)
    span_sum = sum_end - sum_start

    lengths = (ends - starts).clamp(min=1).unsqueeze(-1).to(hidden_states.dtype)
    return span_sum / lengths


class SpanScorerModel(nn.Module):
    def __init__(self, backbone_id_or_path: str, num_labels: int = len(SPAN_SCORER_LABELS)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_id_or_path)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden * 2, num_labels)  # [CLS] repr concat mean-pooled span repr

    def forward(self, input_ids, attention_mask, candidate_starts, candidate_ends, type_labels=None, class_weights=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, H)
        span_repr = pool_span_means(hidden_states, candidate_starts, candidate_ends)  # (B, N, H)
        n_candidates = span_repr.shape[1]
        cls_repr = hidden_states[:, 0:1, :].expand(-1, n_candidates, -1)  # (B, N, H)
        logits = self.classifier(self.dropout(torch.cat([cls_repr, span_repr], dim=-1)))  # (B, N, num_labels)

        loss = None
        if type_labels is not None:
            # Padded candidate slots carry label -100 (same ignore-index convention as
            # BIO's ignored positions elsewhere in this repo) -- CrossEntropyLoss's
            # default ignore_index=-100 excludes them from the mean automatically, no
            # separate mask tensor needed.
            loss = nn.CrossEntropyLoss(weight=class_weights)(logits.reshape(-1, logits.shape[-1]), type_labels.reshape(-1))
        return {"loss": loss, "logits": logits}


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def sample_training_candidates(
    gold_spans_tok: list[tuple[int, int, str]],
    all_candidates: list[tuple[int, int]],
    negative_sampling_ratio: float,
    hard_negative_fraction: float,
    rng: random.Random,
) -> list[dict]:
    """Always keeps every gold-positive candidate (never subsampled -- protects rare
    labels like CO/ST, with only ~38-50 total instances in the whole corpus, from
    being starved). Negatives are sampled at negative_sampling_ratio per positive,
    split between hard negatives (partial IoU 0.1-0.5 against some gold span -- teach
    boundary precision) and random/easy negatives (basic null-vs-content signal).
    """
    gold_by_span = {(s, e): label for s, e, label in gold_spans_tok}
    gold_set = set(gold_by_span.keys())
    non_gold = [c for c in all_candidates if c not in gold_set]

    hard, easy = [], []
    for c in non_gold:
        best_iou = max((_iou(c, g) for g in gold_set), default=0.0)
        (hard if 0.1 <= best_iou <= 0.5 else easy).append(c)

    n_pos = max(len(gold_set), 1)
    n_neg = int(round(negative_sampling_ratio * n_pos))
    n_hard = int(round(n_neg * hard_negative_fraction))
    n_easy = n_neg - n_hard

    sampled_hard = rng.sample(hard, min(n_hard, len(hard)))
    sampled_easy = rng.sample(easy, min(n_easy, len(easy)))

    result = [{"start": s, "end": e, "label_id": span_label2id[label]} for (s, e), label in gold_by_span.items()]
    for s, e in sampled_hard + sampled_easy:
        result.append({"start": s, "end": e, "label_id": span_label2id["O"]})
    return result


def build_span_scorer_examples(
    tokenizer, rows: list[dict], max_len: int, stride: int, max_span_width: int,
    negative_sampling_ratio: float, hard_negative_fraction: float, discourse_cues: bool,
    rng: random.Random | None = None,
) -> list[dict]:
    """Chunk-aware (mirrors train_task2.py's _bio_tags_for_chunks): enumerates
    candidates per chunk, only relevant for the ~1% of long paragraphs needing
    sliding-window chunking."""
    rng = rng or random.Random(0)
    examples = []
    for r in rows:
        text = build_input_text(r["text"], r["type"], discourse_cues)
        prefix_len = len(text) - len(r["text"])
        enc = tokenizer(
            text, truncation=True, max_length=max_len, stride=stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding=False,
        )
        for chunk_i in range(len(enc["input_ids"])):
            offsets = enc["offset_mapping"][chunk_i]
            content_idxs = [i for i, (s, e) in enumerate(offsets) if s != e]
            if not content_idxs:
                continue
            lo, hi = min(content_idxs), max(content_idxs) + 1
            local_candidates = enumerate_candidate_spans(hi - lo, max_span_width)
            all_candidates = [(lo + s, lo + e) for s, e in local_candidates]

            gold_spans_tok = []
            for s in r["labels"]:
                tok_start, tok_end = char_span_to_token_span(offsets, s["start_offset"] + prefix_len, s["end_offset"] + prefix_len)
                if tok_start is None:
                    continue
                tok_end_excl = tok_end + 1
                if tok_end_excl - tok_start > max_span_width:
                    continue  # excluded by the width cap -- a disclosed, real limitation
                gold_spans_tok.append((tok_start, tok_end_excl, s["label"]))

            sampled = sample_training_candidates(gold_spans_tok, all_candidates, negative_sampling_ratio, hard_negative_fraction, rng)
            if not sampled:
                continue
            examples.append(
                {
                    "input_ids": enc["input_ids"][chunk_i],
                    "attention_mask": enc["attention_mask"][chunk_i],
                    "candidate_starts": [c["start"] for c in sampled],
                    "candidate_ends": [c["end"] for c in sampled],
                    "type_labels": [c["label_id"] for c in sampled],
                }
            )
    return examples


def make_span_scorer_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = nn.utils.rnn.pad_sequence(
            [torch.tensor(b["input_ids"]) for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            [torch.tensor(b["attention_mask"]) for b in batch], batch_first=True, padding_value=0
        )
        max_n = max(len(b["candidate_starts"]) for b in batch)
        starts = torch.zeros(len(batch), max_n, dtype=torch.long)
        ends = torch.zeros(len(batch), max_n, dtype=torch.long)
        labels = torch.full((len(batch), max_n), -100, dtype=torch.long)  # pad = ignored, not the null class
        for i, b in enumerate(batch):
            n = len(b["candidate_starts"])
            starts[i, :n] = torch.tensor(b["candidate_starts"], dtype=torch.long)
            ends[i, :n] = torch.tensor(b["candidate_ends"], dtype=torch.long)
            labels[i, :n] = torch.tensor(b["type_labels"], dtype=torch.long)
        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "candidate_starts": starts, "candidate_ends": ends, "type_labels": labels,
        }

    return collate_fn


def span_class_weights(rows: list[dict], clip: float) -> torch.Tensor:
    """Inverse-frequency weights over SPAN_SCORER_LABELS (null class fixed at 1.0 --
    its abundance is already controlled by negative_sampling_ratio, not loss
    weighting). Clip default (20x) is higher than Task 1's 8x: span-level CO/ST
    imbalance (~35x vs AS) is far more extreme than Task 1's paragraph-level
    imbalance -- see utils/config.py's SpanScorerConfig docstring."""
    counts = collections.Counter(s["label"] for r in rows for s in r["labels"])
    total = sum(counts.values())
    weights = [1.0]
    for label in LABELS:
        n = max(counts.get(label, 0), 1)
        weights.append(min(clip, total / (len(LABELS) * n)))
    return torch.tensor(weights, dtype=torch.float32)


class SpanScorerTrainer(Trainer):
    """Injects class_weights (bound via constructor, not a module-level global) into
    SpanScorerModel.forward at every compute_loss call -- same pattern as
    span_type_classifier.py's ClassWeightedSpanTrainer."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        type_labels = inputs.pop("type_labels")
        outputs = model(**inputs, type_labels=type_labels, class_weights=self.class_weights.to(model.classifier.weight.device))
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def decode_spans_from_scores(
    candidates: list[dict], score_threshold: float, overlap_iou_threshold: float
) -> list[dict]:
    """Greedy score-ranked NMS over candidate {"label","start_offset","end_offset","score"}
    dicts. Same-label overlap (any overlap) suppresses the lower-scoring candidate.
    Different-label overlap suppresses the lower-scoring one only if IoU is AT/ABOVE
    overlap_iou_threshold; below it, both are kept -- this is what lets the model
    represent the ~1% overlapping-gold-span case BIO+CRF cannot, while still
    defaulting to non-overlapping output (matching ~99% of gold) for everything else.
    """
    kept = [c for c in candidates if c["score"] >= score_threshold]
    kept.sort(key=lambda c: c["score"], reverse=True)

    accepted: list[dict] = []
    for c in kept:
        suppressed = False
        for a in accepted:
            inter = max(0, min(c["end_offset"], a["end_offset"]) - max(c["start_offset"], a["start_offset"]))
            if inter <= 0:
                continue
            if c["label"] == a["label"]:
                suppressed = True
                break
            union = (c["end_offset"] - c["start_offset"]) + (a["end_offset"] - a["start_offset"]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= overlap_iou_threshold:
                suppressed = True
                break
        if not suppressed:
            accepted.append(c)

    return [{"label": c["label"], "start_offset": c["start_offset"], "end_offset": c["end_offset"]} for c in accepted]


@torch.no_grad()
def predict_span_scorer_paragraph(
    text: str, domain: str, model, tokenizer, max_span_width: int, max_len: int, stride: int,
    discourse_cues: bool, score_threshold: float, overlap_iou_threshold: float,
) -> list[dict]:
    full_text = build_input_text(text, domain, discourse_cues)
    prefix_len = len(full_text) - len(text)
    enc = tokenizer(
        full_text, truncation=True, max_length=max_len, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt",
    )
    offset_mapping_batches = enc.pop("offset_mapping")
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    mask_np = attention_mask.cpu().numpy()

    all_candidates: list[dict] = []
    for chunk_i in range(input_ids.shape[0]):
        offsets = offset_mapping_batches[chunk_i].tolist()
        mask_row = mask_np[chunk_i]
        content_idxs = [i for i, (m, (s, e)) in enumerate(zip(mask_row, offsets)) if m == 1 and s != e]
        if not content_idxs:
            continue
        lo, hi = min(content_idxs), max(content_idxs) + 1
        local_candidates = enumerate_candidate_spans(hi - lo, max_span_width)
        if not local_candidates:
            continue
        cand_starts = [lo + s for s, e in local_candidates]
        cand_ends = [lo + e for s, e in local_candidates]

        out = model(
            input_ids=input_ids[chunk_i : chunk_i + 1],
            attention_mask=attention_mask[chunk_i : chunk_i + 1],
            candidate_starts=torch.tensor([cand_starts], dtype=torch.long, device=device),
            candidate_ends=torch.tensor([cand_ends], dtype=torch.long, device=device),
        )
        probs = out["logits"].softmax(dim=-1)[0]  # (N, num_labels)
        real_probs = probs[:, 1:]  # exclude the null class from label selection
        best_label_idx = real_probs.argmax(dim=-1)
        best_score = real_probs.gather(1, best_label_idx.unsqueeze(-1)).squeeze(-1)

        for (tok_s, tok_e), label_idx, score in zip(local_candidates, best_label_idx.tolist(), best_score.tolist()):
            g_s, g_e = lo + tok_s, lo + tok_e
            char_s, char_e = offsets[g_s][0], offsets[g_e - 1][1]
            s2, e2 = max(0, char_s - prefix_len), max(0, char_e - prefix_len)
            if e2 <= s2:
                continue
            all_candidates.append({"label": LABELS[label_idx], "start_offset": s2, "end_offset": e2, "score": score})

    return decode_spans_from_scores(all_candidates, score_threshold, overlap_iou_threshold)
