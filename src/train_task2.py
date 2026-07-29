"""Task 2: span detection.

variant dispatches on config flags, not string branches inside shared functions:
baseline/boosted_crf/enhanced_track_a all flow through the Track A pipeline
(tokenize_and_align_task2 -> train_task2_model -> ensemble_track_a), differing only by
config.model.use_crf, config.data.jitter_augment, config.ensembling.*, and whether
postprocessing is applied. enhanced_track_b is a separate two-stage pipeline
(run_track_b) per CLAUDE.md section 5's docstring listing it as train_task2.py's 4th
dispatch target.
"""

from __future__ import annotations

import argparse
import collections
import gc
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from data.augmentation import jitter_spans
from data.loading import (
    BIO_TAGS,
    LABELS,
    bio2id,
    build_shared_split,
    id2bio,
    label2id,
    load_dev_in,
    load_task1,
    load_task2,
    write_jsonl,
)
from evaluation.scoring import corpus_partial_overlap_f1, score_task2
from evaluation.submission import write_submission_zip
from models.crf_tagger import TokenClassifierWithCRF
from models.span_scorer import (
    SpanScorerModel,
    SpanScorerTrainer,
    build_span_scorer_examples,
    make_span_scorer_collate_fn,
    predict_span_scorer_paragraph,
    span_class_weights,
)
from models.span_type_classifier import (
    ClassWeightedSpanTrainer,
    ContrastiveSpanTypeTrainer,
    SpanTypeClassifier,
    build_span_type_examples,
    make_span_collate_fn,
    predict_span_types,
)
from postprocessing.spans import (
    decode_bio_greedy,
    ensemble_decode_spans,
    offsets_kept_by_mask,
    postprocess_spans,
    spans_from_char_to_tag,
)
from pretraining.tapt import run_tapt
from text.cues import build_input_text
from utils.checkpointing import save_best_and_last_checkpoints
from utils.config import Config, SpanScorerConfig, load_config
from utils.logging import install_rounded_logging
from utils.tracking import RunTracker, make_trainer_callback

ARG_BIO_TAGS = ["O", "B-ARG", "I-ARG"]
arg_bio2id = {t: i for i, t in enumerate(ARG_BIO_TAGS)}
arg_id2bio = {i: t for t, i in arg_bio2id.items()}


# --- per-epoch proxy metrics for best-checkpoint selection (never used for reported
# numbers -- see evaluation/scoring.py) ---


def compute_token_classification_metrics(eval_pred) -> dict:
    """Token-level macro F1 over BIO tag ids (ignoring -100 padding), via simple
    argmax -- not the CRF's Viterbi decode and not the official partial-overlap span
    metric. Works for both the 13-tag (Track A) and 3-tag (Track B Stage A) schemes."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    return {"f1_macro": f1_score(labels[mask], preds[mask], average="macro", zero_division=0)}


def compute_span_scorer_metrics(eval_pred) -> dict:
    """Candidate-level macro F1 over the 7-way (null + 6 labels) classification,
    ignoring -100 padded candidate slots -- same shape as compute_token_classification_metrics,
    just over enumerated candidates instead of BIO tag positions. A per-epoch proxy
    for best-checkpoint selection only, not the official partial-overlap span metric."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    return {"f1_macro": f1_score(labels[mask], preds[mask], average="macro", zero_division=0)}


def compute_span_type_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}


# --- shared char-labeling / chunked tokenization (used by 13-tag and 3-tag schemes) ---


def char_labels_for_paragraph(text: str, spans: list[dict]) -> list[str | None]:
    """First-writer-wins on the rare (~1%) overlapping-gold-span paragraphs."""
    char_labels: list[str | None] = [None] * len(text)
    for s in spans:
        start, end = s["start_offset"], min(s["end_offset"], len(text))
        for i in range(start, end):
            if char_labels[i] is None:
                char_labels[i] = s["label"]
    return char_labels


def char_labels_for_paragraph_binary(text: str, spans: list[dict]) -> list[str | None]:
    char_labels: list[str | None] = [None] * len(text)
    for s in spans:
        start, end = s["start_offset"], min(s["end_offset"], len(text))
        for i in range(start, end):
            if char_labels[i] is None:
                char_labels[i] = "ARG"
    return char_labels


def _bio_tags_for_chunks(
    tokenizer, text: str, char_labels: list[str | None], tag_map: dict[str, int], max_len: int, stride: int
) -> list[dict]:
    enc = tokenizer(
        text, truncation=True, max_length=max_len, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding=False,
    )
    chunk_examples = []
    for chunk_i in range(len(enc["input_ids"])):
        offsets = enc["offset_mapping"][chunk_i]
        token_tags, prev_label = [], None
        for s, e in offsets:
            if s == e:
                token_tags.append(-100)
                prev_label = None
                continue
            chunk_chars = [c for c in char_labels[s:e] if c is not None]
            raw = collections.Counter(chunk_chars).most_common(1)[0][0] if chunk_chars else "O"
            if raw == "O":
                token_tags.append(tag_map["O"])
                prev_label = None
            elif raw == prev_label:
                token_tags.append(tag_map[f"I-{raw}"])
            else:
                token_tags.append(tag_map[f"B-{raw}"])
                prev_label = raw
        chunk_examples.append(
            {"input_ids": enc["input_ids"][chunk_i], "attention_mask": enc["attention_mask"][chunk_i], "labels": token_tags}
        )
    return chunk_examples


def tokenize_and_align_task2(
    tokenizer, rows: list[dict], max_len: int, stride: int, discourse_cues: bool, augment_jitter: int = 0, jitter_seed: int = 0
) -> list[dict]:
    examples = []
    rng = random.Random(jitter_seed)
    for r in rows:
        variants = [r["labels"]] + [jitter_spans(r["labels"], len(r["text"]), rng=rng) for _ in range(augment_jitter)]
        for spans in variants:
            text = build_input_text(r["text"], r["type"], discourse_cues)
            prefix_len = len(text) - len(r["text"])
            char_labels = [None] * prefix_len + char_labels_for_paragraph(r["text"], spans)
            examples.extend(_bio_tags_for_chunks(tokenizer, text, char_labels, bio2id, max_len, stride))
    return examples


def tokenize_and_align_boundary(tokenizer, rows: list[dict], max_len: int, stride: int, discourse_cues: bool) -> list[dict]:
    examples = []
    for r in rows:
        text = build_input_text(r["text"], r["type"], discourse_cues)
        prefix_len = len(text) - len(r["text"])
        char_labels = [None] * prefix_len + char_labels_for_paragraph_binary(r["text"], r["labels"])
        examples.extend(_bio_tags_for_chunks(tokenizer, text, char_labels, arg_bio2id, max_len, stride))
    return examples


# --- unified decode: CRF (Viterbi) or plain softmax argmax, same span reconstruction ---


@torch.no_grad()
def predict_task2_paragraph(
    text: str, domain: str, model, tokenizer, use_crf: bool, id2bio_map: dict[int, str],
    max_len: int, stride: int, discourse_cues: bool,
) -> list[dict]:
    full_text = build_input_text(text, domain, discourse_cues)
    prefix_len = len(full_text) - len(text)
    enc = tokenizer(
        full_text, truncation=True, max_length=max_len, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt",
    )
    offset_mapping_batches = enc.pop("offset_mapping")
    device = next(model.parameters()).device
    model_inputs = {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}
    mask_np = model_inputs["attention_mask"].cpu().numpy()

    char_to_tag: dict[tuple[int, int], str] = {}
    if use_crf:
        decoded_paths, _ = model.decode(**model_inputs)
        for chunk_i, path in enumerate(decoded_paths):
            offsets = offset_mapping_batches[chunk_i].tolist()
            mask_row = mask_np[chunk_i]
            kept_offsets = offsets_kept_by_mask(offsets, mask_row)
            assert len(kept_offsets) == len(path), "mask/path length mismatch"
            for (s, e), tag_id in zip(kept_offsets, path):
                if s == e:
                    continue
                char_to_tag[(s, e)] = id2bio_map[tag_id]
    else:
        logits = model(**model_inputs)["logits"]
        tag_ids_batch = logits.argmax(dim=-1).cpu().tolist()
        for chunk_i, tag_ids in enumerate(tag_ids_batch):
            offsets = offset_mapping_batches[chunk_i].tolist()
            mask_row = mask_np[chunk_i]
            char_to_tag.update(decode_bio_greedy(tag_ids, id2bio_map, offsets, mask_row))

    result = []
    for span in spans_from_char_to_tag(char_to_tag):
        s2, e2 = max(0, span["start_offset"] - prefix_len), max(0, span["end_offset"] - prefix_len)
        if e2 > s2:
            result.append({"label": span["label"], "start_offset": s2, "end_offset": e2})
    return result


# --- Track A (baseline / boosted_crf / enhanced_track_a share this pipeline) ---


@dataclass
class Task2RunResult:
    val_spans: dict[str, list[dict]]
    dev_spans: dict[str, list[dict]]
    internal_f1: float


def train_task2_model(
    backbone_id: str, seed: int, config: Config, tapt_cache_dir: str, split, dev_in: list[dict], output_dir: str
) -> Task2RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if config.tapt.enabled:
        safe_name = backbone_id.replace("/", "__")
        unlabeled_texts = [build_input_text(r["text"], r["type"], config.data.discourse_cues) for r in split.task1_train]
        unlabeled_texts += [build_input_text(r["text"], r["type"], config.data.discourse_cues) for r in dev_in]
        checkpoint = run_tapt(backbone_id, unlabeled_texts, f"{tapt_cache_dir}/{safe_name}", config.tapt, base_seed=seed)
    else:
        checkpoint = backbone_id

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if config.model.use_crf:
        model = TokenClassifierWithCRF(checkpoint, num_labels=len(BIO_TAGS))
        if config.model.gradient_checkpointing:
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(BIO_TAGS))
        if config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = tokenize_and_align_task2(
        tokenizer, split.task2_train, config.model.max_seq_len, config.model.stride, config.data.discourse_cues,
        augment_jitter=config.data.jitter_augment, jitter_seed=seed,
    )
    train_ds = Dataset.from_list(train_examples)
    val_examples = tokenize_and_align_task2(
        tokenizer, split.task2_val, config.model.max_seq_len, config.model.stride, config.data.discourse_cues,
    )
    val_ds = Dataset.from_list(val_examples)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    run_name = f"{config.task}_{config.variant}_{backbone_id.replace('/', '__')}_{seed}"
    tracker = RunTracker(config.wandb, run_name, run_config={"backbone": backbone_id, "seed": seed, "variant": config.variant})

    save_best = config.training.save_best_checkpoint
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        eval_strategy="epoch" if save_best else "no",
        save_strategy="epoch" if save_best else "no",
        save_total_limit=2 if save_best else None,
        load_best_model_at_end=save_best,
        metric_for_best_model="f1_macro" if save_best else None,
        greater_is_better=True if save_best else None,
        logging_steps=20,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds if save_best else None, data_collator=collator,
        compute_metrics=compute_token_classification_metrics if save_best else None,
        callbacks=[make_trainer_callback(tracker)] if config.wandb.enabled else None,
    )
    install_rounded_logging(trainer)
    trainer.train()

    if save_best:
        # TokenClassifierWithCRF is a bare nn.Module, not a HF PreTrainedModel, so
        # trainer.save_model() falls back to a raw state_dict (pytorch_model.bin, no
        # config.json/save_pretrained support) -- reloading it means reconstructing
        # TokenClassifierWithCRF(checkpoint, num_labels=...) and calling
        # load_state_dict() manually, unlike Task 1's full HF checkpoint.
        save_best_and_last_checkpoints(trainer, tokenizer, output_dir)

    model.eval()
    val_spans = {
        r["paragraph_id"]: predict_task2_paragraph(
            r["text"], r["type"], model, tokenizer, config.model.use_crf, id2bio,
            config.model.max_seq_len, config.model.stride, config.data.discourse_cues,
        )
        for r in split.task2_val
    }
    dev_spans = {
        r["paragraph_id"]: predict_task2_paragraph(
            r["text"], r["type"], model, tokenizer, config.model.use_crf, id2bio,
            config.model.max_seq_len, config.model.stride, config.data.discourse_cues,
        )
        for r in dev_in
    }

    gold_val_by_id = {r["paragraph_id"]: r["labels"] for r in split.task2_val}
    internal_f1 = corpus_partial_overlap_f1(gold_val_by_id, val_spans)
    tracker.log({"internal_partial_overlap_f1": internal_f1})
    tracker.finish()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return Task2RunResult(val_spans=val_spans, dev_spans=dev_spans, internal_f1=internal_f1)


def ensemble_track_a(
    run_results: list[Task2RunResult], split, dev_in: list[dict], config: Config, data_dir: str, postprocess: bool
) -> None:
    weights = (
        [r.internal_f1 for r in run_results] if config.ensembling.weighting == "internal_f1" else [1.0] * len(run_results)
    )
    min_weight = sum(weights) / 2

    def decode_for(rows: list[dict], spans_runs: list[dict[str, list[dict]]]) -> dict[str, list[dict]]:
        out = {}
        for r in rows:
            raw = ensemble_decode_spans(
                [runs[r["paragraph_id"]] for runs in spans_runs], len(r["text"]), min_weight, weights
            )
            out[r["paragraph_id"]] = postprocess_spans(r["text"], raw) if postprocess else raw
        return out

    val_spans_ensemble = decode_for(split.task2_val, [r.val_spans for r in run_results])
    dev_spans_ensemble = decode_for(dev_in, [r.dev_spans for r in run_results])

    _write_and_score(val_spans_ensemble, dev_spans_ensemble, split, dev_in, config, data_dir)


def _write_and_score(val_spans, dev_spans, split, dev_in, config: Config, data_dir: str) -> None:
    output_dir = Path(config.output_dir)
    gold_val_path = output_dir / "val_gold.jsonl"
    pred_val_path = output_dir / "val_pred.jsonl"
    dev_pred_path = output_dir / "dev_pred.jsonl"

    write_jsonl(gold_val_path, [{"paragraph_id": r["paragraph_id"], "labels": r["labels"], "type": r["type"]} for r in split.task2_val])
    write_jsonl(pred_val_path, [{"paragraph_id": r["paragraph_id"], "labels": val_spans[r["paragraph_id"]], "type": r["type"]} for r in split.task2_val])
    dev_pred_rows = [{"paragraph_id": r["paragraph_id"], "labels": dev_spans[r["paragraph_id"]], "type": r["type"]} for r in dev_in]
    write_jsonl(dev_pred_path, dev_pred_rows)

    if config.submission.enabled:
        write_submission_zip(
            dev_pred_rows,
            task_file_stem="task_2",
            team_name=config.submission.team_name,
            training_setting=config.submission.training_setting,
            output_dir=f"submissions/{config.submission.phase}",
            # See train_task1.ensemble_and_score's identical comment: output_dir's
            # basename, not variant, avoids collisions between sibling configs that
            # deliberately share the same variant string (e.g. enhanced_track_b vs.
            # enhanced_track_b_contrastive both use variant: enhanced_track_b so
            # train_task2.py's dispatch is unchanged).
            run_label=Path(config.output_dir).name,
        )

    tracker = RunTracker(config.wandb, f"{config.task}_{config.variant}_ensemble", run_config={"variant": config.variant})
    print("=== Overall ===")
    score_task2(data_dir, str(gold_val_path), str(pred_val_path))
    print("\n=== Editorials only ===")
    score_task2(data_dir, str(gold_val_path), str(pred_val_path), data_type="E")
    print("\n=== Debates only ===")
    score_task2(data_dir, str(gold_val_path), str(pred_val_path), data_type="D")
    tracker.finish()


def run_track_a_pipeline(config: Config, split, dev_in: list[dict], data_dir: str) -> None:
    postprocess = config.variant == "enhanced_track_a"
    tapt_cache_dir = "outputs/tapt_checkpoints"
    run_results = []
    for backbone_id in config.backbones:
        for seed in config.seeds:
            safe_name = backbone_id.replace("/", "__")
            run_output_dir = f"{config.output_dir}/runs/{safe_name}_seed{seed}"
            run_results.append(train_task2_model(backbone_id, seed, config, tapt_cache_dir, split, dev_in, run_output_dir))
    ensemble_track_a(run_results, split, dev_in, config, data_dir, postprocess=postprocess)


# --- span_scorer: enumerate-and-classify alternative to BIO+CRF, feeds into the SAME
# Track A ensembling/scoring glue (Task2RunResult / ensemble_track_a / _write_and_score)
# since its output is the same generic span-dict shape ---


def train_span_scorer_model(
    backbone_id: str, seed: int, config: Config, tapt_cache_dir: str, split, dev_in: list[dict], output_dir: str
) -> Task2RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    sc_cfg = config.span_scorer or SpanScorerConfig()

    if config.tapt.enabled:
        safe_name = backbone_id.replace("/", "__")
        unlabeled_texts = [build_input_text(r["text"], r["type"], config.data.discourse_cues) for r in split.task1_train]
        unlabeled_texts += [build_input_text(r["text"], r["type"], config.data.discourse_cues) for r in dev_in]
        checkpoint = run_tapt(backbone_id, unlabeled_texts, f"{tapt_cache_dir}/{safe_name}", config.tapt, base_seed=seed)
    else:
        checkpoint = backbone_id

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = SpanScorerModel(checkpoint)
    if config.model.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    rng = random.Random(seed)
    train_examples = build_span_scorer_examples(
        tokenizer, split.task2_train, config.model.max_seq_len, config.model.stride, sc_cfg.max_span_width,
        sc_cfg.negative_sampling_ratio, sc_cfg.hard_negative_fraction, config.data.discourse_cues, rng=rng,
    )
    train_ds = Dataset.from_list(train_examples)
    val_examples = build_span_scorer_examples(
        tokenizer, split.task2_val, config.model.max_seq_len, config.model.stride, sc_cfg.max_span_width,
        sc_cfg.negative_sampling_ratio, sc_cfg.hard_negative_fraction, config.data.discourse_cues, rng=rng,
    )
    val_ds = Dataset.from_list(val_examples)
    collator = make_span_scorer_collate_fn(tokenizer)
    class_weights = span_class_weights(split.task2_train, sc_cfg.class_weight_clip)

    run_name = f"{config.task}_{config.variant}_{backbone_id.replace('/', '__')}_{seed}"
    tracker = RunTracker(config.wandb, run_name, run_config={"backbone": backbone_id, "seed": seed, "variant": config.variant})

    save_best = config.training.save_best_checkpoint
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        eval_strategy="epoch" if save_best else "no",
        save_strategy="epoch" if save_best else "no",
        save_total_limit=2 if save_best else None,
        load_best_model_at_end=save_best,
        metric_for_best_model="f1_macro" if save_best else None,
        greater_is_better=True if save_best else None,
        logging_steps=20,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to=[],
    )
    trainer = SpanScorerTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds if save_best else None, data_collator=collator,
        class_weights=class_weights, compute_metrics=compute_span_scorer_metrics if save_best else None,
        callbacks=[make_trainer_callback(tracker)] if config.wandb.enabled else None,
    )
    install_rounded_logging(trainer)
    trainer.train()

    if save_best:
        # SpanScorerModel is a bare nn.Module, not a HF PreTrainedModel -- same
        # state_dict-only caveat as TokenClassifierWithCRF (see train_task2_model).
        save_best_and_last_checkpoints(trainer, tokenizer, output_dir)

    model.eval()
    val_spans = {
        r["paragraph_id"]: predict_span_scorer_paragraph(
            r["text"], r["type"], model, tokenizer, sc_cfg.max_span_width, config.model.max_seq_len, config.model.stride,
            config.data.discourse_cues, sc_cfg.decode_score_threshold, sc_cfg.decode_overlap_iou_threshold,
        )
        for r in split.task2_val
    }
    dev_spans = {
        r["paragraph_id"]: predict_span_scorer_paragraph(
            r["text"], r["type"], model, tokenizer, sc_cfg.max_span_width, config.model.max_seq_len, config.model.stride,
            config.data.discourse_cues, sc_cfg.decode_score_threshold, sc_cfg.decode_overlap_iou_threshold,
        )
        for r in dev_in
    }

    gold_val_by_id = {r["paragraph_id"]: r["labels"] for r in split.task2_val}
    internal_f1 = corpus_partial_overlap_f1(gold_val_by_id, val_spans)
    tracker.log({"internal_partial_overlap_f1": internal_f1})
    tracker.finish()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return Task2RunResult(val_spans=val_spans, dev_spans=dev_spans, internal_f1=internal_f1)


def run_span_scorer_pipeline(config: Config, split, dev_in: list[dict], data_dir: str) -> None:
    tapt_cache_dir = "outputs/tapt_checkpoints"
    run_results = []
    for backbone_id in config.backbones:
        for seed in config.seeds:
            safe_name = backbone_id.replace("/", "__")
            run_output_dir = f"{config.output_dir}/runs/{safe_name}_seed{seed}"
            run_results.append(train_span_scorer_model(backbone_id, seed, config, tapt_cache_dir, split, dev_in, run_output_dir))
    ensemble_track_a(run_results, split, dev_in, config, data_dir, postprocess=True)


# --- Track B: separate two-stage pipeline, own functions, own GPU cleanup per stage ---


def train_boundary_stage(backbone_id: str, seed: int, config: Config, tapt_cache_dir: str, split, output_dir: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if config.tapt.enabled:
        safe_name = backbone_id.replace("/", "__")
        unlabeled_texts = [build_input_text(r["text"], r["type"], config.data.discourse_cues) for r in split.task1_train]
        checkpoint = run_tapt(backbone_id, unlabeled_texts, f"{tapt_cache_dir}/{safe_name}", config.tapt, base_seed=seed)
    else:
        checkpoint = backbone_id

    stage_a_cfg = config.training.stage_a or config.training

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TokenClassifierWithCRF(checkpoint, num_labels=len(ARG_BIO_TAGS))
    if config.model.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = tokenize_and_align_boundary(
        tokenizer, split.task2_train, config.model.max_seq_len, config.model.stride, config.data.discourse_cues
    )
    train_ds = Dataset.from_list(train_examples)
    val_examples = tokenize_and_align_boundary(
        tokenizer, split.task2_val, config.model.max_seq_len, config.model.stride, config.data.discourse_cues
    )
    val_ds = Dataset.from_list(val_examples)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    save_best = stage_a_cfg.save_best_checkpoint
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=stage_a_cfg.epochs,
        per_device_train_batch_size=stage_a_cfg.per_device_batch_size,
        gradient_accumulation_steps=stage_a_cfg.gradient_accumulation_steps,
        learning_rate=stage_a_cfg.learning_rate,
        weight_decay=stage_a_cfg.weight_decay,
        warmup_ratio=stage_a_cfg.warmup_ratio,
        eval_strategy="epoch" if save_best else "no",
        save_strategy="epoch" if save_best else "no",
        save_total_limit=2 if save_best else None,
        load_best_model_at_end=save_best,
        metric_for_best_model="f1_macro" if save_best else None,
        greater_is_better=True if save_best else None,
        logging_steps=20, remove_unused_columns=False,
        fp16=torch.cuda.is_available(), seed=seed, report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds if save_best else None, data_collator=collator,
        compute_metrics=compute_token_classification_metrics if save_best else None,
    )
    install_rounded_logging(trainer)
    trainer.train()

    if save_best:
        save_best_and_last_checkpoints(trainer, tokenizer, output_dir)

    model.eval()

    # Deliberate fix over the source notebook: Track B's stage cells there are
    # un-wrapped top-level code with cleanup deferred to the very end of both stages.
    # CLAUDE.md section 8 bug 3's fix is unconditional ("every function that trains a
    # model"), so each stage gets its own cleanup here.
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer


def train_span_type_stage(backbone_id: str, seed: int, config: Config, tapt_cache_dir: str, split, output_dir: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    checkpoint = backbone_id
    if config.tapt.enabled:
        safe_name = backbone_id.replace("/", "__")
        checkpoint = f"{tapt_cache_dir}/{safe_name}"  # already TAPT'd by the boundary stage's call

    stage_b_cfg = config.training.stage_b or config.training

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = SpanTypeClassifier(checkpoint, num_types=len(LABELS))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    span_label_counts = collections.Counter(s["label"] for r in split.task2_train for s in r["labels"])
    total_spans = sum(span_label_counts.values())
    class_weights = torch.tensor(
        [total_spans / (len(LABELS) * span_label_counts[l]) for l in LABELS], dtype=torch.float32
    )

    train_examples = build_span_type_examples(tokenizer, split.task2_train, label2id, config.model.max_seq_len)
    train_ds = Dataset.from_list(train_examples)
    val_examples = build_span_type_examples(tokenizer, split.task2_val, label2id, config.model.max_seq_len)
    val_ds = Dataset.from_list(val_examples)

    save_best = stage_b_cfg.save_best_checkpoint
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=stage_b_cfg.epochs,
        per_device_train_batch_size=stage_b_cfg.per_device_batch_size,
        gradient_accumulation_steps=stage_b_cfg.gradient_accumulation_steps,
        learning_rate=stage_b_cfg.learning_rate,
        weight_decay=stage_b_cfg.weight_decay,
        warmup_ratio=stage_b_cfg.warmup_ratio,
        eval_strategy="epoch" if save_best else "no",
        save_strategy="epoch" if save_best else "no",
        save_total_limit=2 if save_best else None,
        load_best_model_at_end=save_best,
        metric_for_best_model="f1_macro" if save_best else None,
        greater_is_better=True if save_best else None,
        label_names=["type_labels"],  # SpanTypeClassifier has no HF .config to auto-infer this from
        logging_steps=20, remove_unused_columns=False,
        fp16=torch.cuda.is_available(), seed=seed, report_to=[],
    )
    trainer_cls = ContrastiveSpanTypeTrainer if config.model.contrastive_enabled else ClassWeightedSpanTrainer
    extra_kwargs = (
        {"contrastive_weight": config.model.contrastive_weight, "contrastive_temperature": config.model.contrastive_temperature}
        if config.model.contrastive_enabled
        else {}
    )
    trainer = trainer_cls(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds if save_best else None,
        data_collator=make_span_collate_fn(tokenizer), class_weights=class_weights,
        compute_metrics=compute_span_type_metrics if save_best else None,
        **extra_kwargs,
    )
    install_rounded_logging(trainer)
    trainer.train()

    if save_best:
        save_best_and_last_checkpoints(trainer, tokenizer, output_dir)

    model.eval()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer


def run_track_b(config: Config, split, dev_in: list[dict], data_dir: str) -> None:
    backbone_id, seed = config.backbones[0], config.seeds[0]
    tapt_cache_dir = "outputs/tapt_checkpoints"

    boundary_model, boundary_tokenizer = train_boundary_stage(
        backbone_id, seed, config, tapt_cache_dir, split, f"{config.output_dir}/runs/boundary"
    )
    span_type_model, span_type_tokenizer = train_span_type_stage(
        backbone_id, seed, config, tapt_cache_dir, split, f"{config.output_dir}/runs/span_type"
    )

    def decode(rows: list[dict]) -> dict[str, list[dict]]:
        out = {}
        for r in rows:
            boundary_spans = predict_task2_paragraph(
                r["text"], r["type"], boundary_model, boundary_tokenizer, True, arg_id2bio,
                config.model.max_seq_len, config.model.stride, config.data.discourse_cues,
            )
            typed_spans = predict_span_types(
                r["text"], r["type"], boundary_spans, span_type_model, span_type_tokenizer, LABELS, config.model.max_seq_len
            )
            out[r["paragraph_id"]] = postprocess_spans(r["text"], typed_spans)
        return out

    val_spans = decode(split.task2_val)
    dev_spans = decode(dev_in)

    del boundary_model, span_type_model
    gc.collect()
    torch.cuda.empty_cache()

    _write_and_score(val_spans, dev_spans, split, dev_in, config, data_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/raw/Daleel2026")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    task1_rows = load_task1(args.data_dir)
    task2_rows = load_task2(args.data_dir)
    dev_in = load_dev_in(args.data_dir)
    split = build_shared_split(task1_rows, task2_rows)

    if config.variant == "enhanced_track_b":
        run_track_b(config, split, dev_in, args.data_dir)
    elif config.variant in ("baseline", "boosted_crf", "enhanced_track_a"):
        run_track_a_pipeline(config, split, dev_in, args.data_dir)
    elif config.variant == "span_scorer":
        run_span_scorer_pipeline(config, split, dev_in, args.data_dir)
    else:
        raise ValueError(f"Unknown variant: {config.variant!r}")


if __name__ == "__main__":
    main()
