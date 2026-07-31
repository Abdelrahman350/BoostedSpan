"""Tests for train_task1_generative.py's class-balanced SFT weighting
(build_generative_sft_dataset's weight field, make_generative_collate_fn's
include_weight gate, WeightedQLoRATrainer's manual-shift weighted loss). No real
tokenizer/model/GPU needed -- matches test_train_task1_generative.py's existing
fake-stub style.
"""

import math

import torch

from data.loading import LABELS
from models.losses import weighted_bce_pos_weight
from train_task1_generative import (
    WeightedQLoRATrainer,
    build_generative_sft_dataset,
    make_generative_collate_fn,
)


class _FakeWordTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, **kwargs):
        ids = [sum(ord(c) for c in w) % 1000 + 1 for w in text.split()]
        if add_special_tokens:
            ids = [0] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


_ROWS = [
    {"paragraph_id": 1, "text": "نص تجريبي واحد", "type": "editorial", "labels": ["AS"]},
    {"paragraph_id": 2, "text": "نص تجريبي اثنان", "type": "debate", "labels": ["AS", "OT"]},
    {"paragraph_id": 3, "text": "نص تجريبي ثلاثة", "type": "editorial", "labels": ["AS"]},
    {"paragraph_id": 4, "text": "نص تجريبي أربعة", "type": "debate", "labels": []},
]


def test_class_balanced_false_all_weights_one():
    examples = build_generative_sft_dataset(_FakeWordTokenizer(), _ROWS, max_len=64, discourse_cues=False, class_balanced=False)
    assert all(e["weight"] == 1.0 for e in examples)


def test_class_balanced_true_matches_weighted_bce_pos_weight():
    pos_weight = weighted_bce_pos_weight(_ROWS, LABELS, clip=8.0)
    examples = build_generative_sft_dataset(_FakeWordTokenizer(), _ROWS, max_len=64, discourse_cues=False, class_balanced=True)

    # one example per (row, label), in that nested order
    idx = 0
    for row in _ROWS:
        for label_idx, label in enumerate(LABELS):
            example = examples[idx]
            is_yes = label in row["labels"]
            expected = float(pos_weight[label_idx]) if is_yes else 1.0
            assert math.isclose(example["weight"], expected, rel_tol=1e-6)
            idx += 1


def test_collate_fn_omits_weight_key_by_default():
    tok = _FakeWordTokenizer()
    examples = build_generative_sft_dataset(tok, _ROWS[:1], max_len=64, discourse_cues=False)
    batch = make_generative_collate_fn(tok, include_weight=False)(examples)
    assert "weight" not in batch


def test_collate_fn_includes_weight_when_requested():
    tok = _FakeWordTokenizer()
    examples = build_generative_sft_dataset(tok, _ROWS[:1], max_len=64, discourse_cues=False, class_balanced=True)
    batch = make_generative_collate_fn(tok, include_weight=True)(examples)
    assert "weight" in batch
    assert batch["weight"].shape == (len(examples),)


class _FakeCausalOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeCausalModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids, attention_mask):
        return _FakeCausalOutput(self._logits)


def test_weighted_qlora_trainer_hand_derived_loss():
    # 2 examples, seq_len=3, vocab=3, uniform logits everywhere (CE = ln(3) at
    # every supervised position). Example 0: 1 supervised position, weight=1.0.
    # Example 1: 2 supervised positions, weight=2.0.
    logits = torch.zeros(2, 3, 3)
    inputs = {
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 1], [-100, 1, 2]]),
        "weight": torch.tensor([1.0, 2.0]),
    }

    loss = WeightedQLoRATrainer.compute_loss(None, _FakeCausalModel(logits), inputs)

    expected = 1.5 * math.log(3)  # mean(1*ln3, 2*ln3)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)


def test_weighted_qlora_trainer_pops_weight_from_inputs():
    inputs = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "labels": torch.tensor([[-100, 1]]),
        "weight": torch.tensor([1.0]),
    }
    WeightedQLoRATrainer.compute_loss(None, _FakeCausalModel(torch.zeros(1, 2, 3)), inputs)
    assert "weight" not in inputs
