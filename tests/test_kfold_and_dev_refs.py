"""Tests for round-2 additions: dev-refs training pool (leakage guards) and k-fold
CV splits (reproducibility, OOF coverage, task1/task2 alignment per fold)."""

import pytest

from data.loading import _collapse_keys_for_kfold, build_kfold_splits, build_shared_split


def _make_rows(n: int) -> tuple[list[dict], list[dict]]:
    label_cycles = [["AS"], ["AS", "ST"], ["OT"]]
    domains = ["editorial", "debate"]
    task1 = [
        {
            "paragraph_id": f"p{i}",
            "text": f"paragraph number {i}",
            "labels": label_cycles[i % len(label_cycles)],
            "type": domains[i % len(domains)],
        }
        for i in range(n)
    ]
    task2 = [
        {
            "paragraph_id": r["paragraph_id"],
            "text": r["text"],
            "type": r["type"],
            "labels": [{"label": l, "start_offset": 0, "end_offset": 5} for l in r["labels"]],
        }
        for r in task1
    ]
    return task1, task2


def _make_dev_refs(n: int) -> tuple[list[dict], list[dict]]:
    t1 = [
        {"paragraph_id": f"d{i}", "text": f"dev ref paragraph {i}", "labels": ["AS"], "type": "editorial"}
        for i in range(n)
    ]
    t2 = [
        {
            "paragraph_id": r["paragraph_id"],
            "text": r["text"],
            "type": r["type"],
            "labels": [{"label": "AS", "start_offset": 0, "end_offset": 5}],
        }
        for r in t1
    ]
    return t1, t2


def test_dev_refs_go_to_train_side_only():
    task1, task2 = _make_rows(60)
    refs = _make_dev_refs(10)

    split = build_shared_split(task1, task2, dev_refs=refs)

    ref_texts = {r["text"] for r in refs[0]}
    assert {r["text"] for r in split.task1_val}.isdisjoint(ref_texts)
    assert {r["text"] for r in split.task2_val}.isdisjoint(ref_texts)
    assert sum(1 for r in split.task1_train if r["text"] in ref_texts) == 10
    assert sum(1 for r in split.task2_train if r["text"] in ref_texts) == 10


def test_kfold_oof_covers_every_row_exactly_once():
    task1, task2 = _make_rows(60)
    folds = build_kfold_splits(task1, task2, n_folds=5)

    oof_texts = [r["text"] for f in folds for r in f.task1_val]
    assert len(oof_texts) == 60
    assert len(set(oof_texts)) == 60


def test_kfold_reproducible_and_task_aligned():
    task1, task2 = _make_rows(60)
    folds_a = build_kfold_splits(task1, task2, n_folds=5, random_state=42)
    folds_b = build_kfold_splits(task1, task2, n_folds=5, random_state=42)

    for fa, fb in zip(folds_a, folds_b):
        assert [r["paragraph_id"] for r in fa.task1_val] == [r["paragraph_id"] for r in fb.task1_val]
        # task1/task2 alignment invariant, per fold
        assert {r["paragraph_id"] for r in fa.task2_val} == {r["paragraph_id"] for r in fa.task1_val}
        assert {r["paragraph_id"] for r in fa.task2_train} == {r["paragraph_id"] for r in fa.task1_train}


def test_kfold_dev_refs_in_every_train_never_in_val():
    task1, task2 = _make_rows(60)
    refs = _make_dev_refs(7)
    folds = build_kfold_splits(task1, task2, n_folds=5, dev_refs=refs)

    ref_texts = {r["text"] for r in refs[0]}
    for f in folds:
        assert {r["text"] for r in f.task1_val}.isdisjoint(ref_texts)
        assert sum(1 for r in f.task1_train if r["text"] in ref_texts) == 7
        assert sum(1 for r in f.task2_train if r["text"] in ref_texts) == 7


def test_collapse_keys_for_kfold_handles_singletons():
    # 20 common rows + 2 distinct singletons in different domains: everything must end
    # up in classes with >= n_folds members, whatever collapsing that takes.
    keys = ["editorial|AS"] * 10 + ["debate|AS"] * 10 + ["editorial|CO,ST,TE", "debate|AN,CO"]
    collapsed = _collapse_keys_for_kfold(keys, n_folds=5)
    import collections

    counts = collections.Counter(collapsed)
    assert all(v >= 5 for v in counts.values())
    assert len(collapsed) == len(keys)
