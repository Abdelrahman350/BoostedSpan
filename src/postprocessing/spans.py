"""Turn raw per-token model output into character-offset spans, and clean up the result.

Includes the bug-1 regression fix (CLAUDE.md section 8, item 1): offsets must be
filtered by attention_mask==1 FIRST -- the same filter a CRF's .decode() applies
internally -- and only then should (0,0) "special token" entries be skipped inside the
loop. Filtering by "looks like a special token" as a standalone proxy silently
desynchronizes token/offset pairs whenever a batch has padding.
"""

from __future__ import annotations

import collections
import unicodedata


def offsets_kept_by_mask(offsets: list[tuple[int, int]], mask_row) -> list[tuple[int, int]]:
    """Bug-1-safe offset filtering: keep offsets at positions where attention_mask==1,
    in order -- the exact filter a CRF's .decode() applies internally, so the returned
    list aligns 1:1 with the decoded tag sequence for this same sequence."""
    return [o for o, m in zip(offsets, mask_row) if m == 1]


def is_word_char(c: str) -> bool:
    """Category-based, not a hand-picked Unicode range: \\u0600-\\u06FF ("the Arabic
    block") looks like it should mean "Arabic letters" but it also contains Arabic
    punctuation (، ؛ ؟) and other non-letter symbols, which would wrongly count as
    "word content". Excluding Punctuation (P*), Separator/whitespace (Z*), and Control
    (C*) categories -- while keeping combining marks (Mn, e.g. Arabic tashkeel
    diacritics) as word content, since they attach to a word rather than separating one
    -- works correctly for Arabic and any script.
    """
    cat = unicodedata.category(c)
    return not (cat.startswith("P") or cat.startswith("Z") or cat.startswith("C"))


def strip_non_content_spans(text: str, spans: list[dict]) -> list[dict]:
    return [s for s in spans if any(is_word_char(c) for c in text[s["start_offset"] : s["end_offset"]])]


def snap_to_word_boundary(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and is_word_char(text[start - 1]) and is_word_char(text[start]):
        start -= 1
    while end < len(text) and is_word_char(text[end - 1]) and is_word_char(text[end]):
        end += 1
    return start, end


def merge_adjacent_same_label(spans: list[dict], text: str, max_gap: int = 2) -> list[dict]:
    if not spans:
        return spans
    spans_sorted = sorted(spans, key=lambda s: s["start_offset"])
    merged = [dict(spans_sorted[0])]
    for s in spans_sorted[1:]:
        last = merged[-1]
        gap_text = text[last["end_offset"] : s["start_offset"]]
        if (
            s["label"] == last["label"]
            and s["start_offset"] - last["end_offset"] <= max_gap
            and gap_text.strip(" ،.,") == ""
        ):
            last["end_offset"] = max(last["end_offset"], s["end_offset"])
        else:
            merged.append(dict(s))
    return merged


def postprocess_spans(text: str, spans: list[dict]) -> list[dict]:
    spans = strip_non_content_spans(text, spans)
    snapped = []
    for s in spans:
        ns, ne = snap_to_word_boundary(text, s["start_offset"], s["end_offset"])
        snapped.append({"label": s["label"], "start_offset": ns, "end_offset": ne})
    return merge_adjacent_same_label(snapped, text)


def min_weight_for_fraction(weights: list[float], fraction: float) -> float:
    """Total-weight threshold a character's majority-voted label must clear to
    survive ensemble_decode_spans -- fraction=0.5 (the long-standing default) is a
    strict majority; lower values admit spans only a minority of runs agree on."""
    return sum(weights) * fraction


def ensemble_decode_spans(
    all_runs_spans: list[dict], text_len: int, min_weight: float, weights: list[float] | None = None
) -> list[dict]:
    """Character-level majority vote across runs' decoded spans. Weighted by `weights`
    (e.g. each run's internal validation F1) if given, uniform (all 1.0) otherwise."""
    n = len(all_runs_spans)
    weights = weights if weights is not None else [1.0] * n
    votes = [collections.Counter() for _ in range(text_len)]
    for spans, w in zip(all_runs_spans, weights):
        for s in spans:
            for i in range(s["start_offset"], min(s["end_offset"], text_len)):
                votes[i][s["label"]] += w

    char_label = []
    for v in votes:
        if v:
            label, weight_sum = v.most_common(1)[0]
            char_label.append(label if weight_sum >= min_weight else None)
        else:
            char_label.append(None)

    spans, cur_label, cur_start = [], None, None
    for i, lab in enumerate(char_label + [None]):
        if lab != cur_label:
            if cur_label is not None:
                spans.append({"label": cur_label, "start_offset": cur_start, "end_offset": i})
            cur_label, cur_start = lab, i
    return spans


def cluster_ensemble_decode_spans(
    all_runs_spans: list[list[dict]], min_weight: float, weights: list[float], cross_label_iou: float = 0.3
) -> list[dict]:
    """Span-level alternative to ensemble_decode_spans's per-character majority
    vote, fixing a real failure mode diagnosed against enhanced_track_a_weighted:
    each individual run recalls 96-99% of gold spans regardless of length, but
    char-voting drops ~26% of SHORT spans entirely on the ensembled output.
    Mechanism: independently boundary-jittered predictions across runs can
    fragment a short span's per-character vote so no single character clears the
    majority threshold, even though every run found something roughly there. A
    2-char symmetric dilation before char-voting was tried and made things worse
    (F1 0.687 vs baseline 0.706) -- it bridges gaps between separate, correctly-
    predicted nearby spans as a side effect, not just recovering missed ones.

    This clusters each label's candidate spans by transitive character overlap
    FIRST (classic interval-merge sweep), counts each contributing run's weight
    ONCE per cluster (not diluted across characters), and outputs the UNION of a
    cluster's member spans once its total weight clears min_weight -- so partial,
    boundary-jittered agreement across runs recovers one span instead of vanishing.
    A separate cross-label NMS pass (same IoU-threshold pattern as
    models/span_scorer.py's decode_spans_from_scores) resolves overlapping
    different-label clusters afterward, since char-voting no longer does that for
    free. Empirically (val sweep against enhanced_track_a_weighted's checkpoints):
    F1 0.730 vs. char-voting's 0.706, with precision/recall much more balanced.
    """
    candidates: list[tuple[int, int, str, int]] = []
    for run_idx, spans in enumerate(all_runs_spans):
        for s in spans:
            candidates.append((s["start_offset"], s["end_offset"], s["label"], run_idx))

    by_label: dict[str, list[tuple[int, int, str, int]]] = collections.defaultdict(list)
    for c in candidates:
        by_label[c[2]].append(c)

    clusters: list[tuple[str, int, int, float]] = []  # (label, start, end, total_weight)
    for label, items in by_label.items():
        items.sort()
        n = len(items)
        i = 0
        while i < n:
            cur_end = items[i][1]
            member_runs = {items[i][3]}
            members = [items[i]]
            j = i + 1
            while j < n and items[j][0] < cur_end:
                cur_end = max(cur_end, items[j][1])
                member_runs.add(items[j][3])
                members.append(items[j])
                j += 1
            total_weight = sum(weights[r] for r in member_runs)
            if total_weight >= min_weight:
                start = min(m[0] for m in members)
                end = max(m[1] for m in members)
                clusters.append((label, start, end, total_weight))
            i = j

    clusters.sort(key=lambda c: c[3], reverse=True)
    accepted: list[tuple[str, int, int, float]] = []
    for label, start, end, w in clusters:
        suppressed = False
        for al, astart, aend, _ in accepted:
            inter = max(0, min(end, aend) - max(start, astart))
            if inter <= 0:
                continue
            if label == al:
                suppressed = True
                break
            union = (end - start) + (aend - astart) - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= cross_label_iou:
                suppressed = True
                break
        if not suppressed:
            accepted.append((label, start, end, w))

    return [{"label": l, "start_offset": s, "end_offset": e} for l, s, e, _ in accepted]


def spans_from_char_to_tag(char_to_tag: dict[tuple[int, int], str]) -> list[dict]:
    """Reconstruct spans from a {(start, end): "O"|"B-<label>"|"I-<label>"} map, keyed
    by character offset and merged across chunks (a later chunk's entry for the same
    offset key naturally overwrites an earlier chunk's -- "later chunk wins" on
    sliding-window overlap).

    This single reconstruction rule serves BOTH the CRF path (predict_task2_paragraph,
    predict_boundary_spans) and the baseline's greedy argmax decode: an I-<label> only
    continues the currently open span if that span is the SAME label; otherwise
    (a B-tag, no open span, or a different label) it starts a new span. That is
    already exactly "dangling I-tag recovers as B-tag" -- the CRF path doesn't need a
    separate recovery rule, it uses the identical logic.
    """
    tok_spans = sorted(char_to_tag.keys())
    spans: list[list] = []
    open_span: list | None = None
    for s, e in tok_spans:
        tag = char_to_tag[(s, e)]
        if tag == "O":
            if open_span:
                spans.append(open_span)
            open_span = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or open_span is None or open_span[0] != label:
            if open_span:
                spans.append(open_span)
            open_span = [label, s, e]
        else:
            open_span[2] = e
    if open_span:
        spans.append(open_span)
    return [{"label": label, "start_offset": s, "end_offset": e} for label, s, e in spans]


def decode_bio_greedy(
    tag_ids: list[int], id2bio: dict[int, str], offsets: list[tuple[int, int]], mask_row
) -> dict[tuple[int, int], str]:
    """Baseline decode: independent per-token argmax (no CRF/Viterbi). Returns a
    {(start, end): tag} map for THIS chunk, using the same bug-1-safe mask-first
    offset filtering as the CRF path -- callers merge this into a global char_to_tag
    dict across chunks and reconstruct spans via spans_from_char_to_tag, exactly
    mirroring how the CRF decode path already handles chunk overlap.

    Written from CLAUDE.md's prose spec (section 7); no notebook source exists since
    the baseline was dropped from all three source notebooks in favor of CRF from v2
    onward.
    """
    kept_offsets = offsets_kept_by_mask(offsets, mask_row)
    kept_tag_ids = [t for t, m in zip(tag_ids, mask_row) if m == 1]
    assert len(kept_offsets) == len(kept_tag_ids), "mask/tag length mismatch"

    chunk_char_to_tag: dict[tuple[int, int], str] = {}
    for (s, e), tag_id in zip(kept_offsets, kept_tag_ids):
        if s == e:
            continue
        chunk_char_to_tag[(s, e)] = id2bio[tag_id]
    return chunk_char_to_tag
