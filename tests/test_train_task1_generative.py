"""Tests for train_task1_generative.py -- the QLoRA ALLaM-7B rank-classification
variant. No real 7B model or network access is used: build_sft_example/score_labels_via_logits
are tested against small hand-built fake tokenizer/model stubs matching just the API
surface those functions actually call, matching CLAUDE.md's own honesty framing that
"whether QLoRA actually fits T4 VRAM" and "actual downstream F1" genuinely need a real
GPU run and are NOT claimed to be covered here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from data.loading import LABELS
from train_task1 import RunResult, ensemble_and_score
from train_task1_generative import (
    _BestAdapterState,
    build_generative_prompt,
    build_sft_example,
    score_labels_via_logits,
)


def test_build_generative_prompt_includes_domain_and_cues():
    text = "قال الخبير إن نسبة البطالة ارتفعت 5% هذا العام"
    prompt = build_generative_prompt(text, "editorial", "ST", discourse_cues=True)

    assert "افتتاحية صحفية" in prompt  # editorial domain framing
    assert "نقل كلام عن مصدر آخر" in prompt  # TE cue ("قال") fired
    assert "أرقام أو إحصاءات" in prompt  # ST cue ("%") fired
    assert text in prompt


def test_build_generative_prompt_debate_domain_no_cues():
    prompt = build_generative_prompt("نص عادي بدون أي مؤشرات خاصة", "debate", "AS", discourse_cues=True)
    assert "نقاش أو مناظرة" in prompt
    assert "مؤشرات لغوية" not in prompt  # no cue patterns matched


def test_build_generative_prompt_cues_disabled():
    text = "قال الخبير إن نسبة البطالة ارتفعت 5%"
    prompt = build_generative_prompt(text, "editorial", "ST", discourse_cues=False)
    assert "مؤشرات لغوية" not in prompt


class _FakeWordTokenizer:
    """Deterministic, network-free stand-in: splits on whitespace, maps each word to
    a stable id, prepends a fake BOS. No trailing special token, so a prompt's ids are
    always an exact prefix of (prompt + completion)'s ids -- exercising build_sft_example's
    real masking logic without needing a real tokenizer download."""

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, **kwargs):
        ids = [sum(ord(c) for c in w) % 1000 + 1 for w in text.split()]
        if add_special_tokens:
            ids = [0] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_build_sft_example_masks_prompt_tokens():
    tok = _FakeWordTokenizer()
    prompt = "hello world this is a prompt"
    completion = " yes"

    example = build_sft_example(tok, prompt, completion, max_len=512)

    prompt_ids = tok(prompt)["input_ids"]
    n_prompt = len(prompt_ids)
    assert example["labels"][:n_prompt] == [-100] * n_prompt
    assert example["labels"][n_prompt:] != [-100] * (len(example["labels"]) - n_prompt)
    assert example["input_ids"][:n_prompt] == prompt_ids
    assert len(example["attention_mask"]) == len(example["input_ids"])


class _FakeScoringTokenizer:
    def __call__(self, text, add_special_tokens=True, return_tensors=None, truncation=False, max_length=None):
        stripped = text.strip()
        if stripped == "نعم":
            ids = [111]
        elif stripped == "لا":
            ids = [222]
        else:
            ids = [1, 2, 3]
        if return_tensors == "pt":
            data = {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([[1] * len(ids)])}

            class _Enc(dict):
                def to(self, device):
                    return self

            return _Enc(data)
        return {"input_ids": ids}


class _FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        logits = torch.zeros(1, seq_len, 300)
        logits[0, -1, 111] = 2.0  # "yes" token gets the higher logit
        logits[0, -1, 222] = 0.5
        return type("Out", (), {"logits": logits})()


def test_score_labels_via_logits_shape_and_range():
    model = _FakeCausalLM()
    tok = _FakeScoringTokenizer()

    probs = score_labels_via_logits("some paragraph text", "editorial", model, tok, max_len=512, discourse_cues=True)

    assert probs.shape == (len(LABELS),)
    assert np.all((probs >= 0) & (probs <= 1))
    # "yes" logit (2.0) > "no" logit (0.5) at every probed position -> P(yes) > 0.5 for every label
    assert np.all(probs > 0.5)


def test_best_adapter_state_tracks_highest_f1():
    tracker = _BestAdapterState()
    assert tracker.maybe_update(0.5, {"w": torch.tensor([1.0])}) is True
    assert tracker.maybe_update(0.3, {"w": torch.tensor([2.0])}) is False
    assert tracker.maybe_update(0.7, {"w": torch.tensor([3.0])}) is True

    assert tracker.best_f1 == 0.7
    assert torch.equal(tracker.best_state["w"], torch.tensor([3.0]))


def test_best_adapter_state_ties_dont_overwrite():
    tracker = _BestAdapterState()
    tracker.maybe_update(0.5, {"w": torch.tensor([1.0])})
    assert tracker.maybe_update(0.5, {"w": torch.tensor([99.0])}) is False
    assert torch.equal(tracker.best_state["w"], torch.tensor([1.0]))


def test_best_adapter_state_clones_are_mutation_safe():
    tracker = _BestAdapterState()
    live_state = {"w": torch.tensor([1.0])}
    tracker.maybe_update(0.5, live_state)

    live_state["w"] += 100.0  # mutating the source after the fact must not corrupt the stored best

    assert torch.equal(tracker.best_state["w"], torch.tensor([1.0]))


_DATA_DIR = "data/raw/Daleel2026"


@pytest.mark.skipif(
    not Path(_DATA_DIR, "evaluation", "task1_scoring.py").exists(),
    reason="requires the cloned Daleel2026 data repo (scripts/fetch_data.sh) -- skipped when not present",
)
def test_ensemble_and_score_accepts_generative_run_results(tmp_path, capsys):
    """Proves the zero-change-reuse claim mechanically: ensemble_and_score (built for
    the encoder-classification pipeline) accepts RunResult objects built by the
    generative rank-classification path with no modification."""
    from data.loading import build_shared_split, load_dev_in, load_task1, load_task2
    from utils.config import Config

    task1_rows = load_task1(_DATA_DIR)
    task2_rows = load_task2(_DATA_DIR)
    dev_in = load_dev_in(_DATA_DIR)[:5]  # keep this fast
    split = build_shared_split(task1_rows, task2_rows)
    val_n = len(split.task1_val)

    rng = np.random.default_rng(0)
    run_results = [
        RunResult(val_probs=rng.random((val_n, len(LABELS))), dev_probs=rng.random((len(dev_in), len(LABELS))))
    ]

    config = Config(
        task="task1", variant="qlora_allam", backbones=["fake"], seeds=[42], output_dir=str(tmp_path)
    )

    ensemble_and_score(run_results, split, dev_in, config, _DATA_DIR)

    assert (tmp_path / "dev_pred.jsonl").exists()
    assert (tmp_path / "val_pred.jsonl").exists()
    out = capsys.readouterr().out
    assert "Overall" in out
