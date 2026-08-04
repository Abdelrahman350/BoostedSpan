from postprocessing.spans import (
    cluster_ensemble_decode_spans,
    is_word_char,
    merge_adjacent_same_label,
    min_weight_for_fraction,
    postprocess_spans,
    snap_to_word_boundary,
    spans_from_char_to_tag,
    strip_non_content_spans,
)


def test_cluster_decode_recovers_boundary_jittered_short_span():
    # 3 runs each predict roughly the same short span with slightly different
    # boundaries (the exact failure mode char-voting misses) -- combined weight
    # (3.0) clears min_weight (2.0), so the union of all three should survive as
    # one recovered span, not vanish.
    run0 = [{"label": "AS", "start_offset": 10, "end_offset": 20}]
    run1 = [{"label": "AS", "start_offset": 12, "end_offset": 22}]
    run2 = [{"label": "AS", "start_offset": 8, "end_offset": 18}]
    run3 = []  # this run predicted nothing here

    result = cluster_ensemble_decode_spans([run0, run1, run2, run3], min_weight=2.0, weights=[1.0, 1.0, 1.0, 1.0])

    assert result == [{"label": "AS", "start_offset": 8, "end_offset": 22}]


def test_cluster_decode_drops_cluster_below_min_weight():
    run0 = [{"label": "AS", "start_offset": 10, "end_offset": 20}]
    run1 = []
    run2 = []

    result = cluster_ensemble_decode_spans([run0, run1, run2], min_weight=2.0, weights=[1.0, 1.0, 1.0])

    assert result == []


def test_cluster_decode_same_run_contributing_twice_counts_weight_once():
    # A single run's own decode is internally non-overlapping, but even if it
    # contributed two touching pieces of the same cluster, that run's weight must
    # only count once toward the cluster total -- not be double-counted.
    run0 = [
        {"label": "AS", "start_offset": 10, "end_offset": 15},
        {"label": "AS", "start_offset": 14, "end_offset": 20},
    ]
    run1 = []

    result = cluster_ensemble_decode_spans([run0, run1], min_weight=1.5, weights=[1.0, 1.0])

    # run0's total contribution is weight 1.0 (not 2.0), below min_weight 1.5
    assert result == []


def test_cluster_decode_non_overlapping_same_label_spans_stay_separate():
    run0 = [{"label": "AS", "start_offset": 0, "end_offset": 10}]
    run1 = [{"label": "AS", "start_offset": 100, "end_offset": 110}]

    result = cluster_ensemble_decode_spans([run0, run1], min_weight=0.5, weights=[1.0, 1.0])

    assert sorted(result, key=lambda s: s["start_offset"]) == [
        {"label": "AS", "start_offset": 0, "end_offset": 10},
        {"label": "AS", "start_offset": 100, "end_offset": 110},
    ]


def test_cluster_decode_cross_label_nms_high_iou_suppresses_lower_weight():
    run0 = [{"label": "AS", "start_offset": 0, "end_offset": 20}]  # weight 2.0 (2 runs)
    run1 = [{"label": "AS", "start_offset": 0, "end_offset": 20}]
    run2 = [{"label": "OT", "start_offset": 2, "end_offset": 18}]  # weight 1.0, heavy overlap with AS

    result = cluster_ensemble_decode_spans([run0, run1, run2], min_weight=0.5, weights=[1.0, 1.0, 1.0], cross_label_iou=0.3)

    assert result == [{"label": "AS", "start_offset": 0, "end_offset": 20}]


def test_cluster_decode_cross_label_low_iou_keeps_both():
    run0 = [{"label": "AS", "start_offset": 0, "end_offset": 20}]
    run1 = [{"label": "OT", "start_offset": 18, "end_offset": 30}]  # small overlap, low IoU

    result = cluster_ensemble_decode_spans([run0, run1], min_weight=0.5, weights=[1.0, 1.0], cross_label_iou=0.3)

    assert {(s["label"], s["start_offset"], s["end_offset"]) for s in result} == {("AS", 0, 20), ("OT", 18, 30)}


def test_cluster_decode_short_span_threshold_defaults_off():
    # short_span_max_length/short_span_min_weight both default to None -- exact
    # original single-threshold behavior, a low-weight short cluster still dropped.
    run0 = [{"label": "AS", "start_offset": 0, "end_offset": 10}]
    result = cluster_ensemble_decode_spans([run0], min_weight=2.0, weights=[1.0])
    assert result == []


def test_cluster_decode_short_span_lenient_threshold_recovers_short_cluster():
    # Same single-run, weight-1.0 cluster as above, now with a lenient short-span
    # threshold (max_length=15, min_weight=0.5) -- the 10-char cluster qualifies as
    # "short" and is accepted against 0.5, not the strict global min_weight=2.0.
    run0 = [{"label": "AS", "start_offset": 0, "end_offset": 10}]
    result = cluster_ensemble_decode_spans(
        [run0], min_weight=2.0, weights=[1.0], short_span_max_length=15, short_span_min_weight=0.5
    )
    assert result == [{"label": "AS", "start_offset": 0, "end_offset": 10}]


def test_cluster_decode_short_span_threshold_does_not_affect_long_clusters():
    # A cluster longer than short_span_max_length still uses the strict global
    # min_weight, even when the lenient short-span params are set.
    run0 = [{"label": "AS", "start_offset": 0, "end_offset": 100}]  # 100 chars, not "short"
    result = cluster_ensemble_decode_spans(
        [run0], min_weight=2.0, weights=[1.0], short_span_max_length=15, short_span_min_weight=0.5
    )
    assert result == []


def test_cluster_decode_short_span_boundary_is_inclusive():
    run0 = [{"label": "AS", "start_offset": 0, "end_offset": 15}]  # exactly 15 chars
    result = cluster_ensemble_decode_spans(
        [run0], min_weight=2.0, weights=[1.0], short_span_max_length=15, short_span_min_weight=0.5
    )
    assert result == [{"label": "AS", "start_offset": 0, "end_offset": 15}]


def test_min_weight_for_fraction_default_matches_prior_strict_majority():
    # fraction=0.5 (EnsemblingConfig's default) must reproduce the old hardcoded
    # `sum(weights) / 2` exactly -- a no-op change for every existing config.
    assert min_weight_for_fraction([1.0, 1.0, 1.0, 1.0], 0.5) == 2.0


def test_min_weight_for_fraction_lower_fraction_admits_more_spans():
    assert min_weight_for_fraction([1.0, 1.0, 1.0, 1.0], 0.25) == 1.0


def test_min_weight_for_fraction_respects_unequal_weights():
    assert min_weight_for_fraction([0.6, 0.9], 0.5) == 0.75


def test_is_word_char_arabic_letters():
    assert is_word_char("ا") is True
    assert is_word_char("ب") is True
    assert is_word_char("م") is True


def test_is_word_char_arabic_diacritics():
    # Tanwin fatha (combining mark, category Mn) -- attaches to a word, must count as content.
    assert is_word_char("ً") is True


def test_is_word_char_arabic_punctuation_regression():
    # This is the exact bug-4 regression case: the old ؀-ۿ range wrongly
    # returned True for these since they sit inside the Arabic Unicode block despite
    # not being letters.
    assert is_word_char("،") is False  # Arabic comma
    assert is_word_char("؛") is False  # Arabic semicolon
    assert is_word_char("؟") is False  # Arabic question mark


def test_is_word_char_latin_punctuation_and_whitespace():
    assert is_word_char(".") is False
    assert is_word_char(",") is False
    assert is_word_char(" ") is False


def test_strip_non_content_spans():
    text = "مرحبا، كيف حالك؟"
    spans = [
        {"label": "AS", "start_offset": 0, "end_offset": 6},  # "مرحبا،" has real content
        {"label": "OT", "start_offset": 6, "end_offset": 7},  # just "،"
    ]
    kept = strip_non_content_spans(text, spans)
    assert len(kept) == 1
    assert kept[0]["label"] == "AS"


def test_snap_to_word_boundary():
    text = "hello world"
    # span starts/ends mid-word ("ell wor")
    start, end = snap_to_word_boundary(text, 1, 9)
    assert text[start:end] == "hello world"


def test_merge_adjacent_same_label():
    text = "AS، BS"
    spans = [
        {"label": "AS", "start_offset": 0, "end_offset": 2},
        {"label": "AS", "start_offset": 3, "end_offset": 6},
    ]
    merged = merge_adjacent_same_label(spans, text)
    assert len(merged) == 1
    assert merged[0] == {"label": "AS", "start_offset": 0, "end_offset": 6}


def test_merge_adjacent_different_labels_not_merged():
    text = "AS, BS"
    spans = [
        {"label": "AS", "start_offset": 0, "end_offset": 2},
        {"label": "OT", "start_offset": 4, "end_offset": 6},
    ]
    merged = merge_adjacent_same_label(spans, text)
    assert len(merged) == 2


def test_merge_adjacent_beyond_max_gap_not_merged():
    text = "AS     BS"  # 5-char gap, default max_gap=2
    spans = [
        {"label": "AS", "start_offset": 0, "end_offset": 2},
        {"label": "AS", "start_offset": 7, "end_offset": 9},
    ]
    merged = merge_adjacent_same_label(spans, text)
    assert len(merged) == 2


def test_postprocess_spans_pipeline():
    text = "مرحبا بالعالم، كيف حالك؟"
    spans = [{"label": "AS", "start_offset": 1, "end_offset": 4}]  # mid-word cut
    result = postprocess_spans(text, spans)
    assert len(result) == 1
    s, e = result[0]["start_offset"], result[0]["end_offset"]
    assert text[s:e] == "مرحبا"


def test_spans_from_char_to_tag_dangling_i_recovers():
    # An I-AS with no matching preceding B-AS/I-AS (starts a chunk mid-span) must
    # still produce a span, not be silently dropped -- "dangling I-tag recovers as
    # B-tag" per CLAUDE.md's baseline decode spec.
    char_to_tag = {(0, 1): "I-AS", (1, 2): "I-AS", (2, 3): "O"}
    spans = spans_from_char_to_tag(char_to_tag)
    assert spans == [{"label": "AS", "start_offset": 0, "end_offset": 2}]


def test_spans_from_char_to_tag_later_chunk_wins_on_overlap():
    # Simulates two overlapping chunks where the later chunk's entry for the same
    # offset key overwrites the earlier chunk's (dict update order = insertion order).
    char_to_tag = {}
    char_to_tag.update({(0, 1): "B-AS", (1, 2): "I-AS"})  # chunk 1
    char_to_tag.update({(1, 2): "O", (2, 3): "B-OT"})  # chunk 2, overlapping offset (1,2)
    spans = spans_from_char_to_tag(char_to_tag)
    assert spans == [{"label": "AS", "start_offset": 0, "end_offset": 1}, {"label": "OT", "start_offset": 2, "end_offset": 3}]
