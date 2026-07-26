"""Tests for models/span_scorer.py -- the enumerate-and-classify alternative to
BIO+CRF (see CLAUDE.md's span_scorer variant). No GPU/network/real backbone required:
torch tensor ops here run on tiny hand-built CPU tensors.
"""

import random

import torch

from models.span_scorer import (
    decode_spans_from_scores,
    enumerate_candidate_spans,
    pool_span_means,
    sample_training_candidates,
)


def test_enumerate_candidate_spans_count_and_bounds():
    seq_len, max_width = 10, 3
    candidates = enumerate_candidate_spans(seq_len, max_width)

    expected_count = sum(min(max_width, seq_len - start) for start in range(seq_len))
    assert len(candidates) == expected_count
    for start, end in candidates:
        assert 0 <= start < end <= seq_len
        assert end - start <= max_width


def test_enumerate_candidate_spans_width_wider_than_sequence():
    candidates = enumerate_candidate_spans(3, 10)
    # every (start, end) pair with start < end <= 3
    assert set(candidates) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


def test_pool_span_means_matches_naive_loop():
    torch.manual_seed(0)
    hidden = torch.randn(2, 6, 4)
    starts = torch.tensor([[0, 2], [1, 3]])
    ends = torch.tensor([[3, 5], [4, 6]])

    pooled = pool_span_means(hidden, starts, ends)

    naive = torch.zeros(2, 2, 4)
    for b in range(2):
        for n in range(2):
            s, e = starts[b, n].item(), ends[b, n].item()
            naive[b, n] = hidden[b, s:e].mean(dim=0)

    assert torch.allclose(pooled, naive, atol=1e-5)


def test_pool_span_means_handles_single_token_span():
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    starts = torch.tensor([[2]])
    ends = torch.tensor([[3]])
    pooled = pool_span_means(hidden, starts, ends)
    assert torch.allclose(pooled[0, 0], hidden[0, 2])


def test_negative_sampling_always_keeps_all_gold_positives():
    gold = [(0, 3, "AS"), (5, 8, "OT")]
    all_candidates = enumerate_candidate_spans(10, 4)
    for seed in range(10):
        rng = random.Random(seed)
        sampled = sample_training_candidates(gold, all_candidates, negative_sampling_ratio=5, hard_negative_fraction=0.5, rng=rng)
        sampled_positive_spans = {(c["start"], c["end"]) for c in sampled if c["label_id"] != 0}
        assert sampled_positive_spans == {(0, 3), (5, 8)}


def test_negative_sampling_ratio_and_hard_negative_split():
    gold = [(0, 3, "AS")]
    all_candidates = enumerate_candidate_spans(20, 4)
    rng = random.Random(1)
    sampled = sample_training_candidates(gold, all_candidates, negative_sampling_ratio=4, hard_negative_fraction=1.0, rng=rng)

    negatives = [c for c in sampled if c["label_id"] == 0]
    assert len(negatives) <= 4  # capped by ratio (and by available hard negatives)
    # hard_negative_fraction=1.0 -- every sampled negative must have partial IoU
    # (0.1-0.5) against the gold span, i.e. genuinely overlap it a bit.
    for c in negatives:
        s, e = c["start"], c["end"]
        inter = max(0, min(e, 3) - max(s, 0))
        union = (e - s) + 3 - inter
        iou = inter / union if union > 0 else 0.0
        assert 0.1 <= iou <= 0.5


def test_decode_suppresses_duplicate_same_label():
    candidates = [
        {"label": "AS", "start_offset": 0, "end_offset": 10, "score": 0.9},
        {"label": "AS", "start_offset": 2, "end_offset": 12, "score": 0.5},
    ]
    decoded = decode_spans_from_scores(candidates, score_threshold=0.4, overlap_iou_threshold=0.3)
    assert decoded == [{"label": "AS", "start_offset": 0, "end_offset": 10}]


def test_decode_allows_low_overlap_different_label():
    candidates = [
        {"label": "AS", "start_offset": 0, "end_offset": 20, "score": 0.9},
        {"label": "OT", "start_offset": 18, "end_offset": 30, "score": 0.8},  # small overlap, low IoU
    ]
    decoded = decode_spans_from_scores(candidates, score_threshold=0.4, overlap_iou_threshold=0.3)
    assert len(decoded) == 2
    assert {d["label"] for d in decoded} == {"AS", "OT"}


def test_decode_suppresses_high_overlap_different_label():
    candidates = [
        {"label": "AS", "start_offset": 0, "end_offset": 20, "score": 0.9},
        {"label": "OT", "start_offset": 2, "end_offset": 18, "score": 0.5},  # heavy overlap, high IoU
    ]
    decoded = decode_spans_from_scores(candidates, score_threshold=0.4, overlap_iou_threshold=0.3)
    assert decoded == [{"label": "AS", "start_offset": 0, "end_offset": 20}]


def test_decode_respects_score_threshold():
    candidates = [{"label": "AS", "start_offset": 0, "end_offset": 10, "score": 0.2}]
    decoded = decode_spans_from_scores(candidates, score_threshold=0.5, overlap_iou_threshold=0.3)
    assert decoded == []


def test_decode_merges_candidates_from_multiple_chunks():
    # Simulates predict_span_scorer_paragraph's flow: candidates pooled from several
    # chunks (already mapped to global character offsets) are decoded together in one
    # NMS pass -- a genuine improvement over BIO's "later chunk wins" heuristic.
    chunk1_candidates = [{"label": "AS", "start_offset": 0, "end_offset": 10, "score": 0.9}]
    chunk2_candidates = [
        {"label": "AS", "start_offset": 5, "end_offset": 15, "score": 0.4},  # overlaps chunk1's span, same label
        {"label": "TE", "start_offset": 100, "end_offset": 110, "score": 0.7},  # independent
    ]
    decoded = decode_spans_from_scores(chunk1_candidates + chunk2_candidates, score_threshold=0.3, overlap_iou_threshold=0.3)
    assert decoded == [
        {"label": "AS", "start_offset": 0, "end_offset": 10},
        {"label": "TE", "start_offset": 100, "end_offset": 110},
    ]
