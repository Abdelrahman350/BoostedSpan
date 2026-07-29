"""Produce a submission for an input file OTHER than dev_in.jsonl (e.g. the
Evaluation phase's test_in.jsonl) from an ALREADY-TRAINED checkpoint, without
retraining.

Every train_*.py entrypoint only ever predicts on dev_in.jsonl inside the same
process that just trained the model -- there is no standalone "reload a saved
checkpoint and predict on an arbitrary file" path. This script adds that, reusing
the existing ensemble/scoring/submission glue (train_task1.ensemble_and_score,
train_task2.ensemble_track_a/_write_and_score) UNCHANGED -- they only need
RunResult/Task2RunResult-shaped objects, not knowledge of whether the model was
just trained or reloaded from disk.

Custom nn.Module checkpoints (TokenClassifierWithCRF, SpanTypeClassifier,
SpanScorerModel) were saved as a raw state_dict with no config.json (see each
train_task2_model-family function's checkpoint-saving comment) -- reloading them
means reconstructing the model from the SAME base checkpoint training started from
(resolve_base_checkpoint) and then load_state_dict-ing the saved weights on top, not
pointing the model constructor at the run's own output checkpoint directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import os

import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data.loading import BIO_TAGS, LABELS, build_shared_split, id2bio, load_task1, load_task2, read_jsonl
from evaluation.scoring import corpus_partial_overlap_f1
from models.crf_tagger import TokenClassifierWithCRF
from models.span_scorer import SpanScorerModel, predict_span_scorer_paragraph
from models.span_type_classifier import SpanTypeClassifier, predict_span_types
from postprocessing.spans import postprocess_spans
from train_task1 import RunResult, ensemble_and_score, tokenize_task1
from train_task1_generative import score_labels_via_logits
from train_task2 import (
    ARG_BIO_TAGS,
    Task2RunResult,
    arg_id2bio,
    ensemble_track_a,
    predict_task2_paragraph,
    _write_and_score,
)
from utils.config import Config, SpanScorerConfig, load_config


def resolve_base_checkpoint(backbone_id: str, config: Config, tapt_cache_dir: str = "outputs/tapt_checkpoints") -> str:
    """Same rule every train_task2_model-family function used at train time: if TAPT
    was enabled, the model was built on top of the cached TAPT checkpoint, not the
    raw backbone id."""
    if config.tapt.enabled:
        return f"{tapt_cache_dir}/{backbone_id.replace('/', '__')}"
    return backbone_id


def load_custom_state_dict(model: torch.nn.Module, checkpoint_dir: str) -> torch.nn.Module:
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    state_dict = load_file(st_path) if os.path.exists(st_path) else torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- Task 1: baseline / boosted (AutoModelForSequenceClassification -- a real
# PreTrainedModel, so its saved checkpoint has a full config.json and reloads
# directly via .from_pretrained) ---


def reload_task1_encoder_run(backbone_id: str, seed: int, config: Config, split, test_rows: list[dict]) -> RunResult:
    safe_name = backbone_id.replace("/", "__")
    checkpoint_dir = f"{config.output_dir}/runs/{safe_name}_seed{seed}/checkpoint_best"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(_device())
    model.eval()

    val_ds = Dataset.from_dict(tokenize_task1(tokenizer, split.task1_val, config.model.max_seq_len, config.data.discourse_cues))
    test_ds = Dataset.from_dict(
        tokenize_task1(tokenizer, test_rows, config.model.max_seq_len, config.data.discourse_cues, has_labels=False)
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(output_dir="outputs/_predict_scratch", per_device_eval_batch_size=8, report_to=[])
    trainer = Trainer(model=model, args=args, data_collator=collator)

    val_probs = 1 / (1 + np.exp(-trainer.predict(val_ds).predictions))
    test_probs = 1 / (1 + np.exp(-trainer.predict(test_ds).predictions))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return RunResult(val_probs=val_probs, dev_probs=test_probs)


# --- Task 1: qlora_allam (PEFT adapter on a 4-bit base) ---


def reload_qlora_run(backbone_id: str, seed: int, config: Config, split, test_rows: list[dict]) -> RunResult:
    from peft import PeftModel

    safe_name = backbone_id.replace("/", "__")
    checkpoint_dir = f"{config.output_dir}/runs/{safe_name}_seed{seed}/checkpoint_best"
    quant_cfg = config.quantization

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )
    base_model = AutoModelForCausalLM.from_pretrained(backbone_id, quantization_config=bnb_config, device_map={"": 0} if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()

    val_probs = np.stack(
        [score_labels_via_logits(r["text"], r["type"], model, tokenizer, config.model.max_seq_len, config.data.discourse_cues) for r in split.task1_val]
    )
    test_probs = np.stack(
        [score_labels_via_logits(r["text"], r["type"], model, tokenizer, config.model.max_seq_len, config.data.discourse_cues) for r in test_rows]
    )

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return RunResult(val_probs=val_probs, dev_probs=test_probs)


def run_task1_eval(config: Config, split, test_rows: list[dict], eval_config: Config, data_dir: str) -> None:
    run_results = []
    for backbone_id in config.backbones:
        for seed in config.seeds:
            if config.variant == "qlora_allam":
                run_results.append(reload_qlora_run(backbone_id, seed, config, split, test_rows))
            else:
                run_results.append(reload_task1_encoder_run(backbone_id, seed, config, split, test_rows))
    ensemble_and_score(run_results, split, test_rows, eval_config, data_dir)


# --- Task 2: baseline / boosted_crf / enhanced_track_a / span_scorer (Track A shape) ---


def reload_task2_track_a_run(backbone_id: str, seed: int, config: Config, split, test_rows: list[dict]) -> Task2RunResult:
    safe_name = backbone_id.replace("/", "__")
    checkpoint_dir = f"{config.output_dir}/runs/{safe_name}_seed{seed}/checkpoint_best"
    base = resolve_base_checkpoint(backbone_id, config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    device = _device()

    if config.variant == "span_scorer":
        sc_cfg = config.span_scorer or SpanScorerConfig()
        model = SpanScorerModel(base)
        load_custom_state_dict(model, checkpoint_dir)
        model.to(device)
        model.eval()

        def predict(rows):
            return {
                r["paragraph_id"]: predict_span_scorer_paragraph(
                    r["text"], r["type"], model, tokenizer, sc_cfg.max_span_width, config.model.max_seq_len,
                    config.model.stride, config.data.discourse_cues, sc_cfg.decode_score_threshold, sc_cfg.decode_overlap_iou_threshold,
                )
                for r in rows
            }
    elif config.model.use_crf:
        model = TokenClassifierWithCRF(base, num_labels=len(BIO_TAGS))
        load_custom_state_dict(model, checkpoint_dir)
        model.to(device)
        model.eval()

        def predict(rows):
            return {
                r["paragraph_id"]: predict_task2_paragraph(
                    r["text"], r["type"], model, tokenizer, True, id2bio, config.model.max_seq_len, config.model.stride, config.data.discourse_cues
                )
                for r in rows
            }
    else:
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
        model.to(device)
        model.eval()

        def predict(rows):
            return {
                r["paragraph_id"]: predict_task2_paragraph(
                    r["text"], r["type"], model, tokenizer, False, id2bio, config.model.max_seq_len, config.model.stride, config.data.discourse_cues
                )
                for r in rows
            }

    val_spans = predict(split.task2_val)
    test_spans = predict(test_rows)

    gold_val_by_id = {r["paragraph_id"]: r["labels"] for r in split.task2_val}
    internal_f1 = corpus_partial_overlap_f1(gold_val_by_id, val_spans)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return Task2RunResult(val_spans=val_spans, dev_spans=test_spans, internal_f1=internal_f1)


def run_task2_track_a_eval(config: Config, split, test_rows: list[dict], eval_config: Config, data_dir: str) -> None:
    postprocess = config.variant == "enhanced_track_a"
    run_results = []
    for backbone_id in config.backbones:
        for seed in config.seeds:
            run_results.append(reload_task2_track_a_run(backbone_id, seed, config, split, test_rows))
    ensemble_track_a(run_results, split, test_rows, eval_config, data_dir, postprocess=postprocess)


# --- Task 2: enhanced_track_b (separate two-stage pipeline, single run, no ensembling) ---


def run_task2_track_b_eval(config: Config, split, test_rows: list[dict], eval_config: Config, data_dir: str) -> None:
    backbone_id = config.backbones[0]
    base = resolve_base_checkpoint(backbone_id, config)
    device = _device()

    boundary_dir = f"{config.output_dir}/runs/boundary/checkpoint_best"
    boundary_tokenizer = AutoTokenizer.from_pretrained(boundary_dir)
    boundary_model = TokenClassifierWithCRF(base, num_labels=len(ARG_BIO_TAGS))
    load_custom_state_dict(boundary_model, boundary_dir)
    boundary_model.to(device)
    boundary_model.eval()

    span_type_dir = f"{config.output_dir}/runs/span_type/checkpoint_best"
    span_type_tokenizer = AutoTokenizer.from_pretrained(span_type_dir)
    span_type_model = SpanTypeClassifier(base, num_types=len(LABELS))
    load_custom_state_dict(span_type_model, span_type_dir)
    span_type_model.to(device)
    span_type_model.eval()

    def decode(rows):
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
    test_spans = decode(test_rows)

    del boundary_model, span_type_model
    gc.collect()
    torch.cuda.empty_cache()

    _write_and_score(val_spans, test_spans, split, test_rows, eval_config, data_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-file", default="data/raw/Daleel2026/data/test/test_in.jsonl")
    parser.add_argument("--data-dir", default="data/raw/Daleel2026")
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--training-setting", default="both")
    args = parser.parse_args()

    config = load_config(args.config)
    test_rows = read_jsonl(args.test_file)

    task1_rows = load_task1(args.data_dir)
    task2_rows = load_task2(args.data_dir)
    split = build_shared_split(task1_rows, task2_rows)

    # Never overwrite the dev-phase outputs/submission already sitting in
    # config.output_dir -- write eval-phase results to a sibling directory.
    eval_config = dataclasses.replace(
        config,
        output_dir=f"{config.output_dir}_eval",
        submission=dataclasses.replace(
            config.submission, enabled=True, team_name=args.team_name, training_setting=args.training_setting, phase="eval"
        ),
    )

    if config.task == "task1":
        run_task1_eval(config, split, test_rows, eval_config, args.data_dir)
    elif config.variant == "enhanced_track_b":
        run_task2_track_b_eval(config, split, test_rows, eval_config, args.data_dir)
    else:
        run_task2_track_a_eval(config, split, test_rows, eval_config, args.data_dir)


if __name__ == "__main__":
    main()
