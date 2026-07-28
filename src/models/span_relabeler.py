"""D9: Task 2 span re-labeler hybrid.

D8 (scripts/d8_oracle_typing_headroom.py) measured a large gap between
enhanced_track_a's actual val score (F1=0.7171) and its "oracle typing" score
(F1=0.8868, same boundaries + gold types) -- 23.7% of predicted spans are mistyped.
This module fixes types without touching boundaries: it reuses the exact
rank-classification recipe that won Task 1 (P(yes) via next-token logits at a fixed
yes/no prompt position, train_task1_generative.py), but scoped to "is THIS marked
span of type X" instead of "does this paragraph contain type X anywhere". Boundaries
are never moved -- the literature (and D8) says encoders keep the boundary edge and
partial-overlap scoring forgives residual drift; only span TYPING gets a second
opinion from the LLM.

A dedicated small QLoRA adapter is trained here (NOT the Task 1 adapter reused
as-is) -- Task 1's adapter was SFT-trained on a materially different task (paragraph-
level "does this paragraph contain any span of type X") and was never shown a
span-marked-in-context prompt, so reusing it as-is would be a task mismatch, not a
transfer. Teacher-forced on gold (span, type) pairs, same as Track B's Stage B
classifier but with the LLM instead of a small linear head.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from data.loading import LABELS
from train_task1_generative import LABEL_DESCRIPTIONS, build_sft_example, make_generative_collate_fn
from utils.config import Config
from utils.logging import install_rounded_logging
from utils.tracking import RunTracker, make_trainer_callback

SPAN_OPEN, SPAN_CLOSE = "«", "»"


def mark_span(text: str, start_offset: int, end_offset: int) -> str:
    return text[:start_offset] + SPAN_OPEN + text[start_offset:end_offset] + SPAN_CLOSE + text[end_offset:]


def build_relabel_prompt(text: str, domain: str, span: dict, label: str) -> str:
    domain_desc = "افتتاحية صحفية" if domain == "editorial" else "نقاش أو مناظرة"
    marked = mark_span(text, span["start_offset"], span["end_offset"])
    label_desc = LABEL_DESCRIPTIONS[label]
    return (
        f"فيما يلي فقرة من {domain_desc}، وقد تم تحديد جزء منها بين علامتي {SPAN_OPEN} و {SPAN_CLOSE}:\n\n{marked}\n\n"
        f"هل الجزء المحدد بين {SPAN_OPEN} و {SPAN_CLOSE} هو عنصر حجاجي من النوع التالي: {label_desc}؟\n"
        f"أجب بكلمة واحدة فقط: نعم أو لا.\n"
        f"الإجابة:"
    )


def build_relabel_sft_dataset(tokenizer, task2_rows: list[dict], max_len: int) -> list[dict]:
    """One example per (gold span, candidate label) pair -- the gold label answers
    نعم, the other 5 answer لا, matching how score_span_types_via_logits later probes
    each label independently at inference."""
    examples = []
    for r in task2_rows:
        for span in r["labels"]:
            for label in LABELS:
                prompt = build_relabel_prompt(r["text"], r["type"], span, label)
                answer = " نعم" if label == span["label"] else " لا"
                examples.append(build_sft_example(tokenizer, prompt, answer, max_len))
    return examples


def _yes_no_token_ids(tokenizer) -> tuple[int, int]:
    yes_ids = tokenizer(" نعم", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer(" لا", add_special_tokens=False)["input_ids"]
    return yes_ids[-1], no_ids[-1]


@torch.no_grad()
def score_span_types_via_logits(text: str, domain: str, span: dict, model, tokenizer, max_len: int) -> np.ndarray:
    """(6,) P(yes)-per-label vector for one span, LABELS order."""
    yes_id, no_id = _yes_no_token_ids(tokenizer)
    device = next(model.parameters()).device
    probs = []
    for label in LABELS:
        prompt = build_relabel_prompt(text, domain, span, label)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**enc)
        last_logits = out.logits[0, -1]
        p_yes = torch.softmax(torch.stack([last_logits[yes_id], last_logits[no_id]]), dim=0)[0].item()
        probs.append(p_yes)
    return np.array(probs)


def relabel_spans(
    rows: list[dict], spans_by_id: dict, model, tokenizer, max_len: int, confidence_threshold: float
) -> dict:
    """Returns a NEW spans_by_id with each span's label possibly replaced by the
    LLM's argmax type, only when the LLM's confidence in that type exceeds
    confidence_threshold (tuned on val/OOF against the official scorer -- see
    scripts/d9_train_and_eval_relabeler.py). Never touches start_offset/end_offset."""
    type_by_id = {r["paragraph_id"]: r["type"] for r in rows}
    text_by_id = {r["paragraph_id"]: r["text"] for r in rows}
    out = {}
    for pid, spans in spans_by_id.items():
        domain = type_by_id[pid]
        text = text_by_id[pid]
        new_spans = []
        for span in spans:
            probs = score_span_types_via_logits(text, domain, span, model, tokenizer, max_len)
            best_idx = int(np.argmax(probs))
            new_label = LABELS[best_idx]
            if new_label != span["label"] and probs[best_idx] >= confidence_threshold:
                new_spans.append({**span, "label": new_label})
            else:
                new_spans.append(span)
        out[pid] = new_spans
    return out


def train_span_relabeler(backbone_id: str, seed: int, config: Config, train_rows: list[dict], output_dir: str):
    """Trains and returns (model, tokenizer), adapter also saved to output_dir/checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    quant_cfg = config.quantization
    lora_cfg = config.lora
    if quant_cfg is None or lora_cfg is None:
        raise ValueError("span_relabeler requires both `quantization:` and `lora:` blocks in the config.")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        backbone_id, quantization_config=bnb_config, device_map={"": 0} if torch.cuda.is_available() else "cpu",
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg.r, lora_alpha=lora_cfg.alpha, lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules, task_type="CAUSAL_LM",
        ),
    )
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_examples = build_relabel_sft_dataset(tokenizer, train_rows, config.model.max_seq_len)
    train_ds = Dataset.from_list(train_examples)
    collator = make_generative_collate_fn(tokenizer)

    run_name = f"{config.task}_{config.variant}_{backbone_id.replace('/', '__')}_{seed}"
    tracker = RunTracker(config.wandb, run_name, run_config={"backbone": backbone_id, "seed": seed, "variant": config.variant})

    use_bf16 = quant_cfg.bnb_4bit_compute_dtype == "bfloat16"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        save_strategy="no",
        logging_steps=20,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=torch.cuda.is_available() and use_bf16,
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, data_collator=collator,
        callbacks=[make_trainer_callback(tracker)] if config.wandb.enabled else None,
    )
    install_rounded_logging(trainer)
    trainer.train()

    checkpoint_dir = f"{output_dir}/checkpoint"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    tracker.finish()

    return model, tokenizer


def cleanup_model(model, trainer=None) -> None:
    del model
    if trainer is not None:
        del trainer
    gc.collect()
    torch.cuda.empty_cache()
