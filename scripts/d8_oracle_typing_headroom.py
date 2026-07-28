"""D8: size the headroom for D9's span re-labeler hybrid before building it.

Two measurements, both against the official task2_scoring.py (never the internal
corpus_partial_overlap_f1 approximation, per CLAUDE.md section 8 bug 5):

1. Actual score of the current best Task 2 variant (enhanced_track_a) on its val split.
2. "Oracle typing" score: keep enhanced_track_a's predicted span BOUNDARIES exactly as
   decoded, but replace each predicted span's TYPE with the type of whichever gold span
   on the same paragraph overlaps it the most (falling back to the predicted type if no
   gold span overlaps at all). This isolates "how much of the F1 gap is a typing error
   vs a boundary error" -- if oracle typing barely beats the real score, D9's LLM
   re-labeler (which only fixes types, never moves boundaries) has little to gain and
   should be skipped.

Uses the exact same 85/15 split (default random_state=42) that produced
outputs/task2_enhanced_track_a/val_pred.jsonl, so gold/pred paragraph_ids line up.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

from data.loading import build_shared_split, load_task1, load_task2, read_jsonl, write_jsonl
from evaluation.scoring import score_task2

DATA_DIR = "data/raw/Daleel2026"
VAL_PRED_PATH = "outputs/task2_enhanced_track_a/val_pred.jsonl"
OUT_DIR = Path("/tmp/d8_oracle_typing")


def best_overlap_label(pred_span: dict, gold_spans: list[dict]) -> str:
    ps, pe = pred_span["start_offset"], pred_span["end_offset"]
    best_label, best_overlap = pred_span["label"], 0
    for g in gold_spans:
        overlap = max(0, min(pe, g["end_offset"]) - max(ps, g["start_offset"]))
        if overlap > best_overlap:
            best_overlap, best_label = overlap, g["label"]
    return best_label


def main() -> None:
    task1_rows = load_task1(DATA_DIR)
    task2_rows = load_task2(DATA_DIR)
    split = build_shared_split(task1_rows, task2_rows)

    gold_by_id = {r["paragraph_id"]: r["labels"] for r in split.task2_val}
    type_by_id = {r["paragraph_id"]: r["type"] for r in split.task2_val}

    pred_rows = read_jsonl(VAL_PRED_PATH)
    pred_by_id = {r["paragraph_id"]: r["labels"] for r in pred_rows}

    missing = set(gold_by_id) - set(pred_by_id)
    if missing:
        raise ValueError(f"val_pred.jsonl missing {len(missing)} paragraph_ids present in this split -- split mismatch, aborting")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_path = OUT_DIR / "gold.jsonl"
    pred_path = OUT_DIR / "pred_actual.jsonl"
    oracle_path = OUT_DIR / "pred_oracle_typing.jsonl"

    write_jsonl(gold_path, [{"paragraph_id": pid, "labels": gold_by_id[pid], "type": type_by_id[pid]} for pid in gold_by_id])
    write_jsonl(pred_path, [{"paragraph_id": pid, "labels": pred_by_id[pid], "type": type_by_id[pid]} for pid in gold_by_id])

    oracle_rows = []
    for pid in gold_by_id:
        gold_spans = gold_by_id[pid]
        oracle_spans = [{**s, "label": best_overlap_label(s, gold_spans)} for s in pred_by_id[pid]]
        oracle_rows.append({"paragraph_id": pid, "labels": oracle_spans, "type": type_by_id[pid]})
    write_jsonl(oracle_path, oracle_rows)

    n_pred_spans = sum(len(v) for v in pred_by_id.values())
    n_retyped = sum(
        1
        for pid in gold_by_id
        for s in pred_by_id[pid]
        if best_overlap_label(s, gold_by_id[pid]) != s["label"]
    )
    print(f"Predicted spans: {n_pred_spans}, of which oracle-typing would relabel: {n_retyped} ({100 * n_retyped / n_pred_spans:.1f}%)")

    print("\n=== ACTUAL (enhanced_track_a val predictions, as scored originally) ===")
    score_task2(DATA_DIR, str(gold_path), str(pred_path))

    print("\n=== ORACLE TYPING (same boundaries, gold-informed types) ===")
    score_task2(DATA_DIR, str(gold_path), str(oracle_path))


if __name__ == "__main__":
    main()
