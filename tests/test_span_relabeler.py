"""span_relabeler regression tests, GPU/model-independent parts only: mark_span offset
correctness and build_relabel_prompt structure. Training/evaluation functions need a
real 7B model download and are exercised via scripts/d9_train_and_eval_relabeler.py
instead, consistent with this repo's existing choice not to unit-test
train_task1_generative.py's heavy paths either."""

from models.span_relabeler import LABEL_DESCRIPTIONS, SPAN_CLOSE, SPAN_OPEN, build_relabel_prompt, mark_span


def test_mark_span_wraps_exact_offsets():
    text = "هذا نص فيه جزء محدد للاختبار"
    start, end = text.index("جزء محدد"), text.index("جزء محدد") + len("جزء محدد")

    marked = mark_span(text, start, end)

    assert marked == text[:start] + SPAN_OPEN + "جزء محدد" + SPAN_CLOSE + text[end:]


def test_mark_span_at_text_boundaries():
    text = "نص قصير"
    assert mark_span(text, 0, len(text)) == SPAN_OPEN + text + SPAN_CLOSE
    assert mark_span(text, 0, 0) == SPAN_OPEN + SPAN_CLOSE + text


def test_build_relabel_prompt_contains_marked_span_and_label_description():
    span = {"start_offset": 4, "end_offset": 7, "label": "AS"}
    text = "نص هنا جزء آخر"

    prompt = build_relabel_prompt(text, "editorial", span, "AS")

    assert SPAN_OPEN in prompt and SPAN_CLOSE in prompt
    assert LABEL_DESCRIPTIONS["AS"] in prompt


def test_build_relabel_prompt_varies_by_candidate_label():
    span = {"start_offset": 0, "end_offset": 3, "label": "AS"}
    text = "نص هنا جزء آخر"

    prompt_as = build_relabel_prompt(text, "debate", span, "AS")
    prompt_te = build_relabel_prompt(text, "debate", span, "TE")

    assert prompt_as != prompt_te
    assert LABEL_DESCRIPTIONS["TE"] in prompt_te
    assert LABEL_DESCRIPTIONS["TE"] not in prompt_as
