"""Regression tests for CLAUDE.md section 8 bug 1: CRF decode / offset-mapping
misalignment. The fix is to filter offsets by attention_mask==1 FIRST (the same
filter a CRF's .decode() applies internally), THEN skip (0,0) "special token" entries
inside the loop -- never filter offsets by "looks like a special token" as a
standalone proxy, since that silently desynchronizes token/offset pairs whenever a
batch has padding.
"""

from postprocessing.spans import offsets_kept_by_mask, spans_from_char_to_tag


def test_offsets_kept_by_mask_matches_decoded_path_length():
    # A padded sequence: CLS, one real token, SEP are unmasked (mask=1); two pad
    # positions follow (mask=0). A CRF's .decode() would return exactly 3 tag ids for
    # this sequence -- one per unmasked position.
    offsets = [(0, 0), (0, 5), (0, 0), (0, 0), (0, 0)]  # CLS, tok1, SEP, pad, pad
    mask_row = [1, 1, 1, 0, 0]
    decoded_path = [0, 3, 0]  # O, B-<something>, O -- one id per unmasked position

    kept_offsets = offsets_kept_by_mask(offsets, mask_row)

    assert len(kept_offsets) == len(decoded_path)
    assert kept_offsets == [(0, 0), (0, 5), (0, 0)]


def test_naive_offset_first_filtering_would_diverge():
    # Demonstrates why mask-first filtering matters: naively dropping (0,0) entries
    # BEFORE consulting attention_mask produces a shorter, differently-aligned list
    # than the CRF's decoded path, which would silently mis-pair tags with tokens if
    # zipped directly (the bug this regression test guards against).
    offsets = [(0, 0), (0, 5), (0, 0), (0, 0), (0, 0)]
    mask_row = [1, 1, 1, 0, 0]
    decoded_path = [0, 3, 0]

    correct = offsets_kept_by_mask(offsets, mask_row)
    naive_buggy = [o for o in offsets if o != (0, 0)]

    assert len(correct) == len(decoded_path)
    assert len(naive_buggy) != len(decoded_path)  # the bug: length mismatch vs. the CRF's own output


def test_char_to_tag_alignment_across_padded_batch():
    # Two-sequence batch, sequence 0 padded (5 slots, 3 real), sequence 1 unpadded (3 slots).
    # Each sequence's own mask_row must be used independently -- no cross-contamination
    # between batch-mates.
    id2bio = {0: "O", 3: "B-AS", 4: "I-AS"}

    seq0_offsets = [(0, 0), (0, 5), (5, 6), (0, 0), (0, 0)]
    seq0_mask = [1, 1, 1, 0, 0]
    seq0_path = [0, 3, 4]  # O, B-AS, I-AS for CLS/tok1/tok2 (SEP position folded into tok2 here for simplicity)

    seq1_offsets = [(0, 0), (0, 4), (0, 0)]
    seq1_mask = [1, 1, 1]
    seq1_path = [0, 0, 0]

    char_to_tag = {}
    for offsets, mask_row, path in [(seq0_offsets, seq0_mask, seq0_path), (seq1_offsets, seq1_mask, seq1_path)]:
        kept_offsets = offsets_kept_by_mask(offsets, mask_row)
        assert len(kept_offsets) == len(path)
        for (s, e), tag_id in zip(kept_offsets, path):
            if s == e:
                continue
            char_to_tag[(s, e)] = id2bio[tag_id]

    assert char_to_tag == {(0, 5): "B-AS", (5, 6): "I-AS", (0, 4): "O"}
    spans = spans_from_char_to_tag(char_to_tag)
    assert spans == [{"label": "AS", "start_offset": 0, "end_offset": 6}]
