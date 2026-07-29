"""Boundary-jitter augmentation for Task 2's enhanced_track_a variant.

A data-level proxy for "boundary smoothing" (see CLAUDE.md section 7's note on why the
published boundary-smoothing technique, defined for span-matrix classifiers, doesn't
transplant literally onto a CRF's global sequence likelihood).
"""

from __future__ import annotations

import random


def jitter_spans(
    spans: list[dict], text_len: int, max_shift: int = 2, rng: random.Random | None = None
) -> list[dict]:
    rng = rng or random
    jittered = []
    for s in spans:
        start = s["start_offset"] + rng.randint(-max_shift, max_shift)
        end = s["end_offset"] + rng.randint(-max_shift, max_shift)
        start = max(0, min(start, text_len - 1))
        end = max(start + 1, min(end, text_len))
        jittered.append({"label": s["label"], "start_offset": start, "end_offset": end})
    return jittered
