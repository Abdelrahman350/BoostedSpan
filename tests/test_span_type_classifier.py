"""Tests for models/span_type_classifier.py's retype_spans_with_confidence -- the
confidence-gated override used by train_task2_retype.py's boundary+retype hybrid.
No real backbone/GPU needed: a char-level fake tokenizer + a fixed-logits fake
model exercise the real char_span_to_token_span / build_input_text machinery.
"""

import torch

from data.loading import LABELS
from models.span_type_classifier import retype_spans_with_confidence
from text.cues import build_input_text

_TEXT = "hello world foo bar"
_DOMAIN = "editorial"
_FULL_TEXT = build_input_text(_TEXT, _DOMAIN)  # "[EDITORIAL] hello world foo bar", no cue matches
_PREFIX_LEN = len(_FULL_TEXT) - len(_TEXT)


class _CharTokenizer:
    """Treats every character as its own token -- offsets are (i, i+1) pairs over
    whatever prefix of the text survives `max_length` truncation."""

    def __call__(self, text, truncation=True, max_length=None, return_offsets_mapping=True, return_tensors=None):
        n = len(text) if max_length is None else min(len(text), max_length)
        return {
            "input_ids": torch.tensor([list(range(n))]),
            "attention_mask": torch.tensor([[1] * n]),
            "offset_mapping": torch.tensor([[(i, i + 1) for i in range(n)]]),
        }


class _FixedLogitsModel(torch.nn.Module):
    def __init__(self, logits: list[float]):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self._logits = torch.tensor(logits)

    def forward(self, input_ids, attention_mask, span_start, span_end):
        return {"logits": self._logits.unsqueeze(0)}


def _span(word: str, label: str) -> dict:
    start = _TEXT.index(word)
    return {"label": label, "start_offset": start, "end_offset": start + len(word)}


def test_high_confidence_disagreement_overrides_label():
    # "AN" should win overwhelmingly: softmax of [0,0,0,0,10,0] over
    # LABELS=["AS","AN","ST","TE","CO","OT"] puts ~1.0 confidence on AN (index 1).
    model = _FixedLogitsModel([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    spans = [_span("world", "AS")]

    result = retype_spans_with_confidence(_TEXT, _DOMAIN, spans, model, _CharTokenizer(), LABELS, max_len=64, confidence_threshold=0.6)

    assert result == [{"label": "AN", "start_offset": _TEXT.index("world"), "end_offset": _TEXT.index("world") + 5}]


def test_low_confidence_disagreement_keeps_original_label():
    # Near-uniform logits -> top class confidence well under 0.6 -- original label
    # ("AS") must survive even though the classifier's argmax disagrees with it.
    model = _FixedLogitsModel([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    spans = [_span("world", "AS")]

    result = retype_spans_with_confidence(_TEXT, _DOMAIN, spans, model, _CharTokenizer(), LABELS, max_len=64, confidence_threshold=0.6)

    assert result[0]["label"] == "AS"


def test_high_confidence_agreement_keeps_label():
    # Classifier agrees with the original label -- no change expected either way.
    model = _FixedLogitsModel([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    spans = [_span("world", "AS")]

    result = retype_spans_with_confidence(_TEXT, _DOMAIN, spans, model, _CharTokenizer(), LABELS, max_len=64, confidence_threshold=0.6)

    assert result[0]["label"] == "AS"


def test_truncated_span_keeps_original_label_without_calling_model():
    # max_len cuts the tokenized sequence off before "bar" -- char_span_to_token_span
    # returns (None, None) for it, so it must keep its original label untouched
    # (never handed to the model at all).
    bar_start = _TEXT.index("bar")
    max_len = _PREFIX_LEN + bar_start  # token window ends exactly before "bar" starts

    class _ExplodingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))  # retype_spans_with_confidence needs a param to infer device

        def forward(self, *args, **kwargs):
            raise AssertionError("model should never be called for a truncated-out span")

    spans = [_span("bar", "TE")]
    result = retype_spans_with_confidence(_TEXT, _DOMAIN, spans, _ExplodingModel(), _CharTokenizer(), LABELS, max_len, confidence_threshold=0.6)

    assert result == [{"label": "TE", "start_offset": bar_start, "end_offset": bar_start + 3}]


def test_empty_spans_returns_empty():
    model = _FixedLogitsModel([0.0] * 6)
    assert retype_spans_with_confidence(_TEXT, _DOMAIN, [], model, _CharTokenizer(), LABELS, max_len=64, confidence_threshold=0.6) == []
