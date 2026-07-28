"""BestStateTracker regression tests: keeps the highest-metric state, first-seen wins
ties, and the stored state is a true clone (later mutation of the source dict never
corrupts it). Pure Python + tiny tensors -- no GPU, no downloads."""

import torch

from utils.best_checkpoint import BestStateTracker


def test_keeps_highest_metric_state():
    tracker = BestStateTracker()
    assert tracker.update(0.5, {"w": torch.tensor([1.0])}) is True
    assert tracker.update(0.3, {"w": torch.tensor([2.0])}) is False
    assert tracker.update(0.9, {"w": torch.tensor([3.0])}) is True

    assert tracker.best_metric == 0.9
    assert torch.equal(tracker.best_state["w"], torch.tensor([3.0]))


def test_ties_keep_first_seen():
    tracker = BestStateTracker()
    tracker.update(0.5, {"w": torch.tensor([1.0])})
    updated = tracker.update(0.5, {"w": torch.tensor([2.0])})

    assert updated is False
    assert torch.equal(tracker.best_state["w"], torch.tensor([1.0]))


def test_stored_state_is_a_real_clone():
    tracker = BestStateTracker()
    state = {"w": torch.tensor([1.0, 2.0])}
    tracker.update(0.5, state)

    state["w"][0] = 999.0  # mutate the original after storing

    assert tracker.best_state["w"][0].item() == 1.0


def test_empty_tracker_has_no_best_state():
    tracker = BestStateTracker()
    assert tracker.best_metric == float("-inf")
    assert tracker.best_state is None
