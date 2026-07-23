"""Discourse-marker lexical cues and domain-prefix input composition.

Regex-matches Arabic surface cues for TE (reporting verbs), ST (numerals/statistics),
AN (anecdote markers); folds matches into the model input as a "[CUES:...]" prefix,
alongside a "[EDITORIAL]"/"[DEBATE]" domain-prefix tag. Shared, byte-identical logic
between Task 1 and Task 2 in the source notebooks.
"""

from __future__ import annotations

import re

CUE_PATTERNS: dict[str, re.Pattern] = {
    "TE": re.compile(r"قال|صرَّح|صرح|أكد|حسب|وفقًا ل|وفقا ل|بحسب|أفاد|ذكر أن"),
    "ST": re.compile(r"\d+\s*%|\d+\s*٪|دراسة|إحصائي|بحث علمي|نسبة"),
    "AN": re.compile(r"أتذكر|حدث معي|ذات مرة|قصتي|تجربتي الشخصية"),
}


def cue_tag(text: str, enabled: bool = True) -> str:
    """Returns "" immediately when disabled -- a passthrough, so callers never need
    to branch on discourse_cues themselves."""
    if not enabled:
        return ""
    hits = [label for label, pattern in CUE_PATTERNS.items() if pattern.search(text)]
    return f"[CUES:{','.join(hits)}] " if hits else ""


def build_domain_prefix(domain: str) -> str:
    return "[EDITORIAL] " if domain == "editorial" else "[DEBATE] "


def build_input_text(text: str, domain: str, discourse_cues: bool = True) -> str:
    """Domain prefix is always applied; only the cue tag is gated by discourse_cues."""
    return build_domain_prefix(domain) + cue_tag(text, enabled=discourse_cues) + text
