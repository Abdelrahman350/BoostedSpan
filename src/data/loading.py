"""Shared data loading and train/val split logic for Task 1 and Task 2.

Task 1 and Task 2 describe the same 612 paragraphs (identical paragraph_id, identical
text). The split is built ONCE on Task 1 and the same paragraph_id partition is reused
for Task 2 -- splitting independently would make the two tasks' validation sets
non-comparable. See CLAUDE.md section 3.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import StratifiedKFold, train_test_split

LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

BIO_TAGS = ["O"] + [f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I")]
bio2id = {tag: i for i, tag in enumerate(BIO_TAGS)}
id2bio = {i: tag for tag, i in bio2id.items()}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_task1(data_dir: str | Path) -> list[dict]:
    return read_jsonl(Path(data_dir) / "data" / "train" / "train_task_1.jsonl")


def load_task2(data_dir: str | Path) -> list[dict]:
    return read_jsonl(Path(data_dir) / "data" / "train" / "train_task_2.jsonl")


def load_dev_in(data_dir: str | Path) -> list[dict]:
    return read_jsonl(Path(data_dir) / "data" / "dev" / "dev_in.jsonl")


def load_dev_refs(data_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Gold-labeled dev set (published by the organizers after the dev phase closed;
    same schema and label-set-union invariant as the train files -- verified against
    the live data). Only legal as EXTRA TRAINING DATA for Evaluation-phase runs, where
    the blind target is test_in.jsonl -- never for dev-phase submissions (a model
    trained on dev labels predicting on dev is leakage) and never as validation.
    Returns (task1_ref_rows, task2_ref_rows)."""
    dev_dir = Path(data_dir) / "data" / "dev"
    return read_jsonl(dev_dir / "dev_task_1_ref.jsonl"), read_jsonl(dev_dir / "dev_task_2_ref.jsonl")


def strat_key(row: dict) -> str:
    """Stratification key: "{domain}|{sorted label signature}", e.g. "editorial|AS,ST"."""
    sig = ",".join(sorted(row["labels"])) or "NONE"
    return f"{row['type']}|{sig}"


def _collapse_rare_keys(keys: list[str], min_count: int = 2) -> list[str]:
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return [k if counts[k] >= min_count else f"{k.split('|')[0]}|RARE" for k in keys]


@dataclass
class SplitResult:
    task1_train: list[dict]
    task1_val: list[dict]
    task2_train: list[dict]
    task2_val: list[dict]


def build_shared_split(
    task1_rows: list[dict],
    task2_rows: list[dict],
    test_size: float = 0.15,
    random_state: int = 42,
    dev_refs: tuple[list[dict], list[dict]] | None = None,
) -> SplitResult:
    """dev_refs (from load_dev_refs) are appended to the TRAIN side only, after the
    split is carved from the original rows -- validation always stays within the
    original 612 paragraphs so scores remain comparable across all configs, with or
    without dev-ref training."""
    keys = [strat_key(row) for row in task1_rows]
    keys_collapsed = _collapse_rare_keys(keys)

    task1_train, task1_val = train_test_split(
        task1_rows, test_size=test_size, random_state=random_state, stratify=keys_collapsed
    )

    train_ids = {row["paragraph_id"] for row in task1_train}
    val_ids = {row["paragraph_id"] for row in task1_val}

    task2_train = [row for row in task2_rows if row["paragraph_id"] in train_ids]
    task2_val = [row for row in task2_rows if row["paragraph_id"] in val_ids]

    if dev_refs is not None:
        task1_train = task1_train + dev_refs[0]
        task2_train = task2_train + dev_refs[1]

    return SplitResult(
        task1_train=task1_train, task1_val=task1_val, task2_train=task2_train, task2_val=task2_val
    )


def _collapse_keys_for_kfold(keys: list[str], n_folds: int) -> list[str]:
    """StratifiedKFold needs every class to have >= n_folds members (stricter than
    train_test_split's >= 2). Collapse progressively: rare key -> "{domain}|RARE" ->
    global "RARE" -> absorbed into the largest class if even global RARE is too small."""
    counts = collections.Counter(keys)
    keys = [k if counts[k] >= n_folds else f"{k.split('|')[0]}|RARE" for k in keys]
    counts = collections.Counter(keys)
    keys = [k if counts[k] >= n_folds else "RARE" for k in keys]
    counts = collections.Counter(keys)
    if 0 < counts.get("RARE", 0) < n_folds:
        biggest = counts.most_common(1)[0][0]
        keys = [biggest if k == "RARE" else k for k in keys]
    return keys


def build_kfold_splits(
    task1_rows: list[dict],
    task2_rows: list[dict],
    n_folds: int = 5,
    random_state: int = 42,
    dev_refs: tuple[list[dict], list[dict]] | None = None,
) -> list[SplitResult]:
    """K stratified folds over the SAME (domain, label-signature) keys as
    build_shared_split, with the task1/task2 paragraph_id-alignment invariant
    preserved per fold. The union of all folds' task1_val is exactly the original
    task1_rows (each row is out-of-fold exactly once) -- which is what makes
    out-of-fold prediction matrices possible. dev_refs, when given, are appended to
    every fold's TRAIN side only (they are never out-of-fold: they're not part of the
    612-row universe the folds partition)."""
    keys = _collapse_keys_for_kfold([strat_key(r) for r in task1_rows], n_folds)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    splits = []
    for train_idx, val_idx in skf.split(task1_rows, keys):
        task1_train = [task1_rows[i] for i in train_idx]
        task1_val = [task1_rows[i] for i in val_idx]
        train_ids = {r["paragraph_id"] for r in task1_train}
        val_ids = {r["paragraph_id"] for r in task1_val}
        task2_train = [r for r in task2_rows if r["paragraph_id"] in train_ids]
        task2_val = [r for r in task2_rows if r["paragraph_id"] in val_ids]
        if dev_refs is not None:
            task1_train = task1_train + dev_refs[0]
            task2_train = task2_train + dev_refs[1]
        splits.append(
            SplitResult(task1_train=task1_train, task1_val=task1_val, task2_train=task2_train, task2_val=task2_val)
        )
    return splits
