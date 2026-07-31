"""Tests for models/crf_tagger.py's aux-CE weighting (bio_class_weights_from_span_weights,
WeightedCRFTrainer). No real CRF/backbone/GPU needed for either.
"""

import math

import torch

from data.loading import BIO_TAGS, LABELS, bio2id
from models.crf_tagger import WeightedCRFTrainer, bio_class_weights_from_span_weights


def test_bio_class_weights_expansion():
    # span_weights indexed ["O"] + LABELS = ["O","AS","AN","ST","TE","CO","OT"]
    span_weights = torch.tensor([1.0, 0.5, 2.0, 8.0, 3.0, 8.0, 1.5])

    bio_weights = bio_class_weights_from_span_weights(span_weights, LABELS, bio2id)

    assert bio_weights.shape == (len(BIO_TAGS),)
    assert bio_weights[bio2id["O"]] == 1.0
    for i, label in enumerate(LABELS):
        assert bio_weights[bio2id[f"B-{label}"]] == span_weights[i + 1]
        assert bio_weights[bio2id[f"I-{label}"]] == span_weights[i + 1]
        # B and I of the same label always get the identical weight
        assert bio_weights[bio2id[f"B-{label}"]] == bio_weights[bio2id[f"I-{label}"]]


class _FakeCRFOutput(dict):
    pass


class _FakeCRFModel:
    """Mimics TokenClassifierWithCRF's forward() dict shape: {"loss": ..., "logits": ...}."""

    def __init__(self, crf_loss: torch.Tensor, emissions: torch.Tensor):
        self._crf_loss = crf_loss
        self._emissions = emissions

    def __call__(self, **kwargs):
        return _FakeCRFOutput(loss=self._crf_loss, logits=self._emissions)


def test_weighted_crf_trainer_adds_scaled_aux_ce():
    # 1 example, seq_len=2, num_labels=3. Position 0 supervised (label=1, weight=2.0
    # on that class), position 1 ignored (-100). Uniform emissions -> CE=ln(3).
    crf_loss = torch.tensor(0.7)
    emissions = torch.zeros(1, 2, 3)
    labels = torch.tensor([[1, -100]])
    class_weights = torch.tensor([1.0, 2.0, 1.0])

    trainer = WeightedCRFTrainer.__new__(WeightedCRFTrainer)  # bypass Trainer.__init__
    trainer.bio_class_weights = class_weights
    trainer.aux_ce_weight = 0.5

    loss = trainer.compute_loss(_FakeCRFModel(crf_loss, emissions), {"labels": labels})

    # weighted CE with class_weights=[1,2,1] over a single supervised example at
    # class 1: loss = weight[1] * ln(3) / weight[1] (CrossEntropyLoss's default
    # mean reduction normalizes by the SUM of weights of contributing samples, which
    # here is just weight[1] itself) = ln(3).
    expected = crf_loss + 0.5 * math.log(3)
    assert torch.isclose(loss, expected, atol=1e-4)


def test_weighted_crf_trainer_zero_aux_weight_equals_crf_loss_only():
    crf_loss = torch.tensor(1.234)
    emissions = torch.zeros(1, 2, 3)
    labels = torch.tensor([[0, -100]])

    trainer = WeightedCRFTrainer.__new__(WeightedCRFTrainer)
    trainer.bio_class_weights = torch.ones(3)
    trainer.aux_ce_weight = 0.0

    loss = trainer.compute_loss(_FakeCRFModel(crf_loss, emissions), {"labels": labels})

    assert torch.isclose(loss, crf_loss, atol=1e-6)
