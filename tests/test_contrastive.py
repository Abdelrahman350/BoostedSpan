import math

import torch

from models.contrastive import SupConLoss


def test_identical_embeddings_same_label_near_zero_loss():
    # Only one other sample in the batch, and it's the positive -- softmax assigns
    # it full probability trivially, so log_prob is exactly 0 and loss is exactly 0.
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0])

    loss = SupConLoss(temperature=1.0)(embeddings, labels)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_no_in_batch_positives_returns_zero_not_nan():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
    labels = torch.tensor([0, 1, 2])  # every label unique -> no anchor has a positive

    loss = SupConLoss(temperature=0.1)(embeddings, labels)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    assert not torch.isnan(loss)


def test_single_sample_batch_returns_zero():
    loss = SupConLoss(temperature=0.1)(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_matches_hand_derived_value_for_two_class_batch():
    # 4 embeddings, 2 orthonormal classes of 2 (labels [0, 0, 1, 1]), temperature=1.0.
    # Hand-derived: every anchor's single in-batch positive shares sim=1, its two
    # negatives share sim=0, so per-anchor loss = -(1 - log(e^1 + e^0 + e^0))
    # = log(e + 2) - 1 ~= 0.551444, identical for all 4 anchors by symmetry.
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])

    loss = SupConLoss(temperature=1.0)(embeddings, labels)

    expected = math.log(math.e + 2) - 1
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)


def test_temperature_changes_loss_magnitude():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])

    loss_t1 = SupConLoss(temperature=1.0)(embeddings, labels)
    loss_t_sharp = SupConLoss(temperature=0.5)(embeddings, labels)

    assert not torch.isclose(loss_t1, loss_t_sharp, atol=1e-4)


def test_gradients_flow_through_embeddings():
    embeddings = torch.tensor([[1.0, 0.1], [0.9, 0.2], [0.1, 1.0], [0.2, 0.9]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])

    loss = SupConLoss(temperature=0.2)(embeddings, labels)
    loss.backward()

    assert embeddings.grad is not None
    assert not torch.isnan(embeddings.grad).any()
