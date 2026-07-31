"""Task 2 boundary+retype hybrid: reload an already-trained CRF ensemble's decoded
spans (boundaries -- already decent) and re-score each span's TYPE with a
dedicated, much cheaper single-backbone/seed SpanTypeClassifier (teacher-forced on
GOLD offsets, so it doesn't share the CRF's joint boundary+type training pressure).

Motivation (see docs/research and this session's diagnostic): on
enhanced_track_a_weighted's val predictions, 119 of 470 predicted spans (25%)
overlap a real gold span but with the WRONG type -- and the official scorer
(task2_scoring.py) gives these ZERO credit, identical to a total miss. Confusion
concentrates in AS/AN/OT/TE (semantically fuzzy categories), not the rare CO/ST
labels the CRF's aux-CE weighting (enhanced_track_a_weighted) already targets --
a genuinely different, unaddressed error source.

Reuses, unmodified: predict_eval.reload_task2_track_a_run (reload each source-
ensemble run's checkpoint_best and decode), postprocessing.spans'
ensemble_decode_spans/strip_non_content_spans/snap_to_word_boundary/
merge_adjacent_same_label (the same pipeline train_task2.ensemble_track_a uses,
just split apart so retyping can happen between boundary cleanup and the final
label-dependent merge step), train_task2.train_span_type_stage (Track B's stage B,
unchanged), and train_task2._write_and_score (scoring/writing/submission,
unchanged). No CRF ensemble retraining -- only the new type classifier is trained.
"""

from __future__ import annotations

import argparse
import gc

import torch

from data.loading import LABELS, build_shared_split, load_dev_in, load_task1, load_task2
from models.span_type_classifier import retype_spans_with_confidence
from postprocessing.spans import ensemble_decode_spans, merge_adjacent_same_label, snap_to_word_boundary, strip_non_content_spans
from predict_eval import reload_task2_track_a_run
from train_task2 import Task2RunResult, _write_and_score, train_span_type_stage
from utils.config import load_config


def _boundary_cleanup(text: str, spans: list[dict]) -> list[dict]:
    """postprocess_spans's first two, label-independent steps only -- the final
    merge_adjacent_same_label step is deferred until after retyping, since
    retyping can change which adjacent spans now share a label."""
    spans = strip_non_content_spans(text, spans)
    cleaned = []
    for s in spans:
        ns, ne = snap_to_word_boundary(text, s["start_offset"], s["end_offset"])
        cleaned.append({"label": s["label"], "start_offset": ns, "end_offset": ne})
    return cleaned


def reload_source_ensemble(source_config, split, dev_in: list[dict]) -> list[Task2RunResult]:
    return [
        reload_task2_track_a_run(backbone_id, seed, source_config, split, dev_in)
        for backbone_id in source_config.backbones
        for seed in source_config.seeds
    ]


def ensemble_and_clean(run_results: list[Task2RunResult], source_config, rows: list[dict], spans_attr: str) -> dict[str, list[dict]]:
    weights = (
        [r.internal_f1 for r in run_results] if source_config.ensembling.weighting == "internal_f1" else [1.0] * len(run_results)
    )
    min_weight = sum(weights) / 2
    spans_runs = [getattr(r, spans_attr) for r in run_results]
    out = {}
    for r in rows:
        raw = ensemble_decode_spans([runs[r["paragraph_id"]] for runs in spans_runs], len(r["text"]), min_weight, weights)
        out[r["paragraph_id"]] = _boundary_cleanup(r["text"], raw)
    return out


def retype_and_merge(rows: list[dict], spans_by_id: dict[str, list[dict]], model, tokenizer, max_len: int, confidence_threshold: float) -> dict[str, list[dict]]:
    out = {}
    for r in rows:
        retyped = retype_spans_with_confidence(
            r["text"], r["type"], spans_by_id[r["paragraph_id"]], model, tokenizer, LABELS, max_len, confidence_threshold
        )
        out[r["paragraph_id"]] = merge_adjacent_same_label(retyped, r["text"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/raw/Daleel2026")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if not config.source_config:
        raise ValueError("train_task2_retype.py requires config.source_config (path to the already-trained boundary ensemble's config).")
    source_config = load_config(config.source_config)

    task1_rows = load_task1(args.data_dir)
    task2_rows = load_task2(args.data_dir)
    dev_in = load_dev_in(args.data_dir)
    split = build_shared_split(task1_rows, task2_rows)

    run_results = reload_source_ensemble(source_config, split, dev_in)
    val_spans = ensemble_and_clean(run_results, source_config, split.task2_val, "val_spans")
    dev_spans = ensemble_and_clean(run_results, source_config, dev_in, "dev_spans")

    tapt_cache_dir = "outputs/tapt_checkpoints"
    backbone_id, seed = config.backbones[0], config.seeds[0]
    model, tokenizer = train_span_type_stage(
        backbone_id, seed, config, tapt_cache_dir, split, f"{config.output_dir}/runs/type_classifier"
    )

    val_spans_retyped = retype_and_merge(
        split.task2_val, val_spans, model, tokenizer, config.model.max_seq_len, config.model.retype_confidence_threshold
    )
    dev_spans_retyped = retype_and_merge(
        dev_in, dev_spans, model, tokenizer, config.model.max_seq_len, config.model.retype_confidence_threshold
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    _write_and_score(val_spans_retyped, dev_spans_retyped, split, dev_in, config, args.data_dir)


if __name__ == "__main__":
    main()
