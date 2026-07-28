"""D9: train the span_relabeler QLoRA adapter on the TRAIN split's gold spans only
(never val), then apply it to enhanced_track_a's val predictions and measure the
real gain via the official task2_scoring.py, sweeping the confidence threshold on
that same val set (acceptable here since this is a threshold sweep, not model
selection across many candidates, and mirrors Task 1's per-label threshold sweep --
CLAUDE.md draws the leakage line at test_in.jsonl / dev_in.jsonl, not at val).

Only "adopt" (wire into predict_eval.py/train_task2.py) if some threshold beats the
enhanced_track_a baseline (F1=0.7171, see scripts/d8_oracle_typing_headroom.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

from data.loading import build_shared_split, load_task1, load_task2, read_jsonl, write_jsonl
from evaluation.scoring import score_task2
from models.span_relabeler import cleanup_model, relabel_spans, train_span_relabeler
from utils.config import load_config

DATA_DIR = "data/raw/Daleel2026"
VAL_PRED_PATH = "outputs/task2_enhanced_track_a/val_pred.jsonl"
OUT_DIR = Path("outputs/task2_span_relabeler")


def main() -> None:
    config = load_config("configs/task2/span_relabeler.yaml")

    task1_rows = load_task1(DATA_DIR)
    task2_rows = load_task2(DATA_DIR)
    split = build_shared_split(task1_rows, task2_rows)

    backbone_id = config.backbones[0]
    seed = config.seeds[0]
    # split.task2_val is reused for THREE purposes below: per-epoch best-checkpoint
    # selection (here), the confidence-threshold sweep, and the final reported
    # baseline-vs-relabeled comparison. Accepted simplification -- CLAUDE.md's leakage
    # line is at test_in.jsonl/dev_in.jsonl, not this internal val split, and there's
    # no k-fold OOF for the QLoRA paths (5-fold CV was only built for the encoder
    # paths; QLoRA training is far too slow to repeat 5x per config).
    model, tokenizer = train_span_relabeler(
        backbone_id, seed, config, split.task2_train, str(OUT_DIR / "run"), eval_rows=split.task2_val
    )

    gold_by_id = {r["paragraph_id"]: r["labels"] for r in split.task2_val}
    type_by_id = {r["paragraph_id"]: r["type"] for r in split.task2_val}
    pred_rows = read_jsonl(VAL_PRED_PATH)
    pred_by_id = {r["paragraph_id"]: r["labels"] for r in pred_rows if r["paragraph_id"] in gold_by_id}

    gold_path = OUT_DIR / "gold.jsonl"
    write_jsonl(gold_path, [{"paragraph_id": pid, "labels": gold_by_id[pid], "type": type_by_id[pid]} for pid in gold_by_id])

    model.eval()
    print("\n=== BASELINE (enhanced_track_a, no relabeling) ===")
    base_path = OUT_DIR / "pred_baseline.jsonl"
    write_jsonl(base_path, [{"paragraph_id": pid, "labels": pred_by_id[pid], "type": type_by_id[pid]} for pid in gold_by_id])
    score_task2(DATA_DIR, str(gold_path), str(base_path))

    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        relabeled = relabel_spans(split.task2_val, pred_by_id, model, tokenizer, config.model.max_seq_len, threshold)
        pred_path = OUT_DIR / f"pred_relabeled_t{threshold}.jsonl"
        write_jsonl(pred_path, [{"paragraph_id": pid, "labels": relabeled[pid], "type": type_by_id[pid]} for pid in gold_by_id])
        print(f"\n=== RELABELED threshold={threshold} ===")
        score_task2(DATA_DIR, str(gold_path), str(pred_path))

    cleanup_model(model)


if __name__ == "__main__":
    main()
