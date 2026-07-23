"""Regression tests for corpus_partial_overlap_f1 -- the internal-only approximation
(CLAUDE.md section 8 bug 5). These verify the metric's arithmetic in isolation; they
must never be mistaken for a replacement of the organizers' official scorer.
"""

from evaluation.scoring import corpus_partial_overlap_f1


def test_known_partial_overlap_value():
    gold = {"p1": [{"label": "AS", "start_offset": 0, "end_offset": 10}]}
    pred = {"p1": [{"label": "AS", "start_offset": 5, "end_offset": 15}]}
    # intersection = [5,10) = 5 chars; precision = 5/10 (pred len), recall = 5/10 (gold len)
    f1 = corpus_partial_overlap_f1(gold, pred)
    assert abs(f1 - 0.5) < 1e-9


def test_perfect_match_gives_f1_one():
    gold = {"p1": [{"label": "AS", "start_offset": 0, "end_offset": 10}]}
    pred = {"p1": [{"label": "AS", "start_offset": 0, "end_offset": 10}]}
    f1 = corpus_partial_overlap_f1(gold, pred)
    assert abs(f1 - 1.0) < 1e-9


def test_no_predictions_gives_f1_zero_without_crash():
    gold = {"p1": [{"label": "AS", "start_offset": 0, "end_offset": 10}]}
    pred: dict = {}
    f1 = corpus_partial_overlap_f1(gold, pred)
    assert f1 == 0.0


def test_no_gold_and_no_pred_gives_f1_zero_without_crash():
    f1 = corpus_partial_overlap_f1({}, {})
    assert f1 == 0.0


def test_label_mismatch_contributes_zero():
    # Same offsets, different label -- must never count as a match.
    gold = {"p1": [{"label": "AS", "start_offset": 0, "end_offset": 10}]}
    pred = {"p1": [{"label": "OT", "start_offset": 0, "end_offset": 10}]}
    f1 = corpus_partial_overlap_f1(gold, pred)
    assert f1 == 0.0


def test_label_filter_restricts_to_one_label():
    gold = {"p1": [{"label": "AS", "start_offset": 0, "end_offset": 10}, {"label": "OT", "start_offset": 20, "end_offset": 30}]}
    pred = {"p1": [{"label": "AS", "start_offset": 0, "end_offset": 10}]}
    f1_as = corpus_partial_overlap_f1(gold, pred, label_filter="AS")
    assert abs(f1_as - 1.0) < 1e-9
    f1_ot = corpus_partial_overlap_f1(gold, pred, label_filter="OT")
    assert f1_ot == 0.0
