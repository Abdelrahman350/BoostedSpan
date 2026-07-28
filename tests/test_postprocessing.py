from postprocessing.spans import (
    filter_spans_by_class_rules,
    is_word_char,
    merge_adjacent_same_label,
    postprocess_spans,
    snap_to_word_boundary,
    spans_from_char_to_tag,
    strip_non_content_spans,
)


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


def test_filter_spans_by_class_rules_drops_short_spans_for_configured_labels():
    spans = [
        {"label": "CO", "start_offset": 0, "end_offset": 3},  # length 3
        {"label": "CO", "start_offset": 10, "end_offset": 30},  # length 20
        {"label": "AS", "start_offset": 40, "end_offset": 43},  # length 3, no rule for AS
    ]
    filtered = filter_spans_by_class_rules(spans, min_length_per_class={"CO": 10})
    assert filtered == [
        {"label": "CO", "start_offset": 10, "end_offset": 30},
        {"label": "AS", "start_offset": 40, "end_offset": 43},
    ]


def test_filter_spans_by_class_rules_noop_when_empty():
    spans = [{"label": "AS", "start_offset": 0, "end_offset": 3}]
    assert filter_spans_by_class_rules(spans, min_length_per_class=None) == spans
    assert filter_spans_by_class_rules(spans, min_length_per_class={}) == spans
