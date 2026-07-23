"""Scoring: thin wrappers around the organizers' official scorers, plus an internal-only
partial-overlap F1 approximation.

CLAUDE.md section 8 bug 5: `corpus_partial_overlap_f1` exists purely for cheap internal
decisions (ensemble run weighting, quick track comparison) -- it must NEVER be used for
a reported/leaderboard number. Every number that goes in a report or comparison table
must come from importing and calling the organizers' task1_scoring.py/task2_scoring.py
directly, unmodified.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_scorer(data_dir: str | Path, module_name: str):
    eval_dir = Path(data_dir) / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Could not find {module_name} under {eval_dir}")
    return importlib.import_module(module_name)


def score_task1(data_dir: str | Path, gold_path: str, pred_path: str, data_type: str | None = None):
    task1_scoring = _import_scorer(data_dir, "task1_scoring")
    if data_type is None:
        return task1_scoring.main(gold_path, pred_path)
    return task1_scoring.main(gold_path, pred_path, data_type=data_type)


def score_task2(data_dir: str | Path, gold_path: str, pred_path: str, data_type: str | None = None):
    task2_scoring = _import_scorer(data_dir, "task2_scoring")
    if data_type is None:
        return task2_scoring.main(gold_path, pred_path)
    return task2_scoring.main(gold_path, pred_path, data_type=data_type)


def corpus_partial_overlap_f1(
    gold_by_id: dict[str, list[dict]], pred_by_id: dict[str, list[dict]], label_filter: str | None = None
) -> float:
    """Internal-only approximation. NEVER use this for a reported number -- see module
    docstring and CLAUDE.md section 8 bug 5. Only for ensemble-run weighting and quick
    internal track comparisons."""
    prec_num = prec_den = rec_num = rec_den = 0.0
    for pid, gold_spans in gold_by_id.items():
        pred_spans = pred_by_id.get(pid, [])
        g = [s for s in gold_spans if label_filter is None or s["label"] == label_filter]
        p = [s for s in pred_spans if label_filter is None or s["label"] == label_filter]
        prec_den += len(p)
        rec_den += len(g)
        for ps in p:
            best = 0.0
            for gs in g:
                if gs["label"] == ps["label"]:
                    inter = max(0, min(ps["end_offset"], gs["end_offset"]) - max(ps["start_offset"], gs["start_offset"]))
                    if inter > 0:
                        best = max(best, inter / (ps["end_offset"] - ps["start_offset"]))
            prec_num += best
        for gs in g:
            best = 0.0
            for ps in p:
                if gs["label"] == ps["label"]:
                    inter = max(0, min(ps["end_offset"], gs["end_offset"]) - max(ps["start_offset"], gs["start_offset"]))
                    if inter > 0:
                        best = max(best, inter / (gs["end_offset"] - gs["start_offset"]))
            rec_num += best
    precision = prec_num / prec_den if prec_den else 0.0
    recall = rec_num / rec_den if rec_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1
