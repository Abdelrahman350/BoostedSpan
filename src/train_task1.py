"""Task 1: multi-label ADU-type classification.

variant is entirely a config-flag effect, not a code branch: baseline.yaml sets
tapt.enabled=false, model.loss="weighted_bce", data.discourse_cues=false, and single-
element backbones/seeds lists; boosted.yaml turns all of that on. train_one_run's body
is identical either way.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data.loading import LABELS, id2label, label2id, load_dev_in, load_task1, load_task2, build_shared_split, write_jsonl
from evaluation.scoring import score_task1
from evaluation.submission import write_submission_zip
from models.losses import get_task1_loss_fn, weighted_bce_pos_weight
from pretraining.tapt import run_tapt
from text.cues import build_input_text
from utils.config import Config, load_config
from utils.logging import install_rounded_logging
from utils.tracking import RunTracker, make_trainer_callback


def compute_task1_metrics(eval_pred) -> dict:
    """Per-epoch proxy for best-checkpoint selection: macro F1 at a fixed 0.5
    threshold. Distinct from ensemble_and_score's per-label threshold sweep used for
    the actual reported numbers -- this only decides which epoch's weights to keep."""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {"f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}


def tokenize_task1(tokenizer, rows: list[dict], max_len: int, discourse_cues: bool, has_labels: bool = True) -> dict:
    texts = [build_input_text(r["text"], r["type"], discourse_cues=discourse_cues) for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=False)
    data = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if has_labels:
        data["labels"] = [[1.0 if l in r["labels"] else 0.0 for l in LABELS] for r in rows]
    return data


class Task1Trainer(Trainer):
    """Renamed from the notebook's ASLTrainer: the injected loss_fn (constructor
    param, not a module-level global) now handles both weighted_bce and asymmetric,
    so a name tied to just "ASL" would be misleading."""

    def __init__(self, *args, loss_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


@dataclass
class RunResult:
    val_probs: np.ndarray
    dev_probs: np.ndarray


def train_one_run(
    backbone_id: str,
    seed: int,
    config: Config,
    tapt_cache_dir: str,
    split,
    dev_in: list[dict],
    output_dir: str,
) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if config.tapt.enabled:
        safe_name = backbone_id.replace("/", "__")
        unlabeled_texts = [build_input_text(r["text"], r["type"], config.data.discourse_cues) for r in split.task1_train]
        unlabeled_texts += [build_input_text(r["text"], r["type"], config.data.discourse_cues) for r in dev_in]
        checkpoint = run_tapt(backbone_id, unlabeled_texts, f"{tapt_cache_dir}/{safe_name}", config.tapt, base_seed=seed)
    else:
        checkpoint = backbone_id

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(LABELS), problem_type="multi_label_classification", id2label=id2label, label2id=label2id
    )
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_ds = Dataset.from_dict(
        tokenize_task1(tokenizer, split.task1_train, config.model.max_seq_len, config.data.discourse_cues)
    )
    val_ds = Dataset.from_dict(
        tokenize_task1(tokenizer, split.task1_val, config.model.max_seq_len, config.data.discourse_cues)
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    pos_weight = None
    if config.model.loss == "weighted_bce":
        pos_weight = weighted_bce_pos_weight(split.task1_train, LABELS, clip=8.0)
    loss_fn = get_task1_loss_fn(
        config.model.loss,
        pos_weight=pos_weight,
        gamma_pos=config.model.asl_gamma_pos,
        gamma_neg=config.model.asl_gamma_neg,
        clip=config.model.asl_clip,
    )

    run_name = f"{config.task}_{config.variant}_{backbone_id.replace('/', '__')}_{seed}"
    tracker = RunTracker(config.wandb, run_name, run_config={"backbone": backbone_id, "seed": seed, **_flatten_config(config)})

    save_best = config.training.save_best_checkpoint
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size or config.training.per_device_batch_size,
        gradient_checkpointing=config.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.model.gradient_checkpointing else None,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch" if save_best else "no",
        save_total_limit=1 if save_best else None,
        load_best_model_at_end=save_best,
        metric_for_best_model="f1_macro" if save_best else None,
        greater_is_better=True if save_best else None,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to=[],
    )
    trainer = Task1Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        loss_fn=loss_fn,
        compute_metrics=compute_task1_metrics if save_best else None,
        callbacks=[make_trainer_callback(tracker)] if config.wandb.enabled else None,
    )
    install_rounded_logging(trainer)
    trainer.train()

    if save_best:
        # trainer.model now holds the best epoch's weights (load_best_model_at_end);
        # save a clean, stably-named copy rather than relying on HF's numbered
        # checkpoint-<step> directory.
        checkpoint_dir = f"{output_dir}/checkpoint"
        trainer.save_model(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    val_logits = trainer.predict(val_ds).predictions
    val_probs = 1 / (1 + np.exp(-val_logits))

    dev_ds = Dataset.from_dict(
        tokenize_task1(tokenizer, dev_in, config.model.max_seq_len, config.data.discourse_cues, has_labels=False)
    )
    dev_logits = trainer.predict(dev_ds).predictions
    dev_probs = 1 / (1 + np.exp(-dev_logits))

    tracker.finish()

    # Trainer holds enough internal references that plain refcounting on return isn't
    # reliable for freeing GPU memory -- explicit cleanup after every training run.
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return RunResult(val_probs=val_probs, dev_probs=dev_probs)


def _flatten_config(config: Config) -> dict:
    return {
        "variant": config.variant,
        "tapt_enabled": config.tapt.enabled,
        "loss": config.model.loss,
        "discourse_cues": config.data.discourse_cues,
    }


def ensemble_and_score(
    run_results: list[RunResult], split, dev_in: list[dict], config: Config, data_dir: str
) -> None:
    val_probs_ensemble = np.mean([r.val_probs for r in run_results], axis=0)
    dev_probs_ensemble = np.mean([r.dev_probs for r in run_results], axis=0)

    val_gold = np.array([[1.0 if l in r["labels"] else 0.0 for l in LABELS] for r in split.task1_val])

    best_thresholds: dict[str, float] = {}
    for i, label in enumerate(LABELS):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.05, 0.96, 0.05):
            f1 = f1_score(val_gold[:, i], (val_probs_ensemble[:, i] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[label] = round(float(best_t), 2)

    def probs_to_labels(probs_row) -> list[str]:
        return [l for i, l in enumerate(LABELS) if probs_row[i] > best_thresholds[l]]

    val_pred_labels = [probs_to_labels(row) for row in val_probs_ensemble]
    dev_pred_labels = [probs_to_labels(row) for row in dev_probs_ensemble]

    output_dir = Path(config.output_dir)
    gold_val_path = output_dir / "val_gold.jsonl"
    pred_val_path = output_dir / "val_pred.jsonl"
    dev_pred_path = output_dir / "dev_pred.jsonl"

    write_jsonl(gold_val_path, [{"paragraph_id": r["paragraph_id"], "labels": r["labels"], "type": r["type"]} for r in split.task1_val])
    write_jsonl(pred_val_path, [{"paragraph_id": r["paragraph_id"], "labels": labels, "type": r["type"]} for r, labels in zip(split.task1_val, val_pred_labels)])
    dev_pred_rows = [{"paragraph_id": r["paragraph_id"], "labels": labels, "type": r["type"]} for r, labels in zip(dev_in, dev_pred_labels)]
    write_jsonl(dev_pred_path, dev_pred_rows)

    if config.submission.enabled:
        write_submission_zip(
            dev_pred_rows,
            task_file_stem="task_1",
            team_name=config.submission.team_name,
            training_setting=config.submission.training_setting,
            output_dir=f"submissions/{config.submission.phase}",
            run_label=f"{config.task}_{config.variant}",
        )

    tracker = RunTracker(config.wandb, f"{config.task}_{config.variant}_ensemble", run_config=_flatten_config(config))
    print("=== Overall ===")
    score_task1(data_dir, str(gold_val_path), str(pred_val_path))
    print("\n=== Editorials only ===")
    score_task1(data_dir, str(gold_val_path), str(pred_val_path), data_type="E")
    print("\n=== Debates only ===")
    score_task1(data_dir, str(gold_val_path), str(pred_val_path), data_type="D")
    tracker.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/raw/Daleel2026")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    seeds = [args.seed_override] if args.seed_override is not None else config.seeds

    task1_rows = load_task1(args.data_dir)
    task2_rows = load_task2(args.data_dir)
    dev_in = load_dev_in(args.data_dir)
    split = build_shared_split(task1_rows, task2_rows)

    tapt_cache_dir = "outputs/tapt_checkpoints"
    run_results: list[RunResult] = []
    for backbone_id in config.backbones:
        for seed in seeds:
            safe_name = backbone_id.replace("/", "__")
            run_output_dir = f"{config.output_dir}/runs/{safe_name}_seed{seed}"
            run_results.append(train_one_run(backbone_id, seed, config, tapt_cache_dir, split, dev_in, run_output_dir))

    ensemble_and_score(run_results, split, dev_in, config, args.data_dir)


if __name__ == "__main__":
    main()
