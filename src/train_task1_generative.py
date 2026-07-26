"""Task 1 qlora_allam variant: QLoRA fine-tuning of ALLaM-7B-Instruct-preview as a
generative alternative to the encoder-classification pipeline (train_task1.py).

Design choice: RANK-CLASSIFICATION, not free-form generation. For each of the 6
labels, score P(yes) via the model's next-token logits at a fixed yes/no prompt
position, rather than generating free-form JSON and parsing it. This means
train_task1.ensemble_and_score's existing probability-averaging + per-label threshold
sweep + submission writing is reusable UNCHANGED -- it only needs a
RunResult(val_probs, dev_probs) array, regardless of how the probabilities were
produced. Free-form generation would need new JSON-parsing/malformed-output-fallback/
hard-voting machinery instead; this design avoids all of that.

TAPT is deliberately NOT supported for this variant (see train_one_qlora_run) --
MLM continuation doesn't apply to a causal instruction-tuned decoder the way it does
to the encoder backbones, and there's no notebook precedent for it here (CLAUDE.md
section 9). A qlora_allam.yaml-style config must set tapt.enabled: false; setting it
true raises rather than being silently ignored.
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from data.loading import LABELS, build_shared_split, load_dev_in, load_task1, load_task2
from text.cues import CUE_PATTERNS
from train_task1 import RunResult, ensemble_and_score
from utils.config import Config, load_config
from utils.logging import install_rounded_logging
from utils.tracking import RunTracker, make_trainer_callback

LABEL_DESCRIPTIONS: dict[str, str] = {
    "AS": "افتراض أو رأي شخصي غير مدعوم بدليل مباشر",
    "AN": "قصة شخصية أو حكاية أو تجربة فردية",
    "ST": "إحصائية أو بيانات رقمية أو نسبة مئوية",
    "TE": "شهادة أو رأي منقول عن شخص أو مصدر آخر",
    "CO": "أرضية مشتركة أو حقيقة متفق عليها بين الطرفين",
    "OT": "نوع آخر من الحجج لا يندرج تحت الأنواع السابقة",
}
CUE_DESCRIPTIONS: dict[str, str] = {
    "TE": "نقل كلام عن مصدر آخر",
    "ST": "أرقام أو إحصاءات",
    "AN": "سرد قصة أو تجربة شخصية",
}


def build_generative_prompt(text: str, domain: str, label: str, discourse_cues: bool = True) -> str:
    """Natural-language instruction framing, not build_input_text's bracket-tag
    prefix -- an instruct-tuned chat model was never trained to interpret
    "[EDITORIAL]"/"[CUES:...]" tags the way an encoder's raw input can absorb them.
    Reuses CUE_PATTERNS as the same regex source of truth as text/cues.py, just
    rendered as a natural-language hint instead of a bracket tag.
    """
    domain_desc = "افتتاحية صحفية" if domain == "editorial" else "نقاش أو مناظرة"
    cue_note = ""
    if discourse_cues:
        hits = [name for name, pattern in CUE_PATTERNS.items() if pattern.search(text) and name in CUE_DESCRIPTIONS]
        if hits:
            hint = "، ".join(CUE_DESCRIPTIONS[h] for h in hits)
            cue_note = f" (تحتوي الفقرة على مؤشرات لغوية قد تدل على: {hint})"
    label_desc = LABEL_DESCRIPTIONS[label]
    return (
        f"فيما يلي فقرة من {domain_desc}:\n\n{text}\n\n"
        f"هل تحتوي هذه الفقرة على عنصر حجاجي من النوع التالي: {label_desc}؟{cue_note}\n"
        f"أجب بكلمة واحدة فقط: نعم أو لا.\n"
        f"الإجابة:"
    )


def build_sft_example(tokenizer, prompt: str, completion: str, max_len: int) -> dict:
    """Loss only on completion tokens (-100 on the prompt, this repo's existing
    ignore-index convention -- see BIO's -100 usage elsewhere). Tokenizes the prompt
    alone to get its exact token count, then the full prompt+completion string,
    slicing at that boundary -- an approximation since tokenization near a boundary
    can shift by a token or two versus tokenizing the pieces jointly; acceptable here
    since we only need an approximate split, not exact alignment.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def build_generative_sft_dataset(tokenizer, rows: list[dict], max_len: int, discourse_cues: bool) -> list[dict]:
    """One training example per (paragraph, label) pair -- completion is the gold
    yes/no answer token for that label, matching how score_labels_via_logits later
    probes each label independently at inference."""
    examples = []
    for r in rows:
        for label in LABELS:
            prompt = build_generative_prompt(r["text"], r["type"], label, discourse_cues)
            answer = " نعم" if label in r["labels"] else " لا"
            examples.append(build_sft_example(tokenizer, prompt, answer, max_len))
    return examples


def make_generative_collate_fn(tokenizer):
    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = torch.tensor(b["attention_mask"], dtype=torch.long)
            labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate_fn


def _yes_no_token_ids(tokenizer) -> tuple[int, int]:
    """The single riskiest assumption in this file: that " نعم"/" لا" tokenize to a
    stable single trailing token id in this exact prompt-continuation position.
    NOT verifiable without the real ALLaM tokenizer -- if this assumption is wrong,
    score_labels_via_logits will silently score the wrong token. Flagged, not
    resolved, here; verify against the real tokenizer before trusting results.
    """
    yes_ids = tokenizer(" نعم", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer(" لا", add_special_tokens=False)["input_ids"]
    return yes_ids[-1], no_ids[-1]


@torch.no_grad()
def score_labels_via_logits(text: str, domain: str, model, tokenizer, max_len: int, discourse_cues: bool) -> np.ndarray:
    """Produces the (6,) P(yes)-per-label probability vector that
    train_task1.ensemble_and_score expects in place of a classification head's
    sigmoid output."""
    yes_id, no_id = _yes_no_token_ids(tokenizer)
    device = next(model.parameters()).device
    probs = []
    for label in LABELS:
        prompt = build_generative_prompt(text, domain, label, discourse_cues)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**enc)
        last_logits = out.logits[0, -1]
        p_yes = torch.softmax(torch.stack([last_logits[yes_id], last_logits[no_id]]), dim=0)[0].item()
        probs.append(p_yes)
    return np.array(probs)


def train_one_qlora_run(backbone_id: str, seed: int, config: Config, split, dev_in: list[dict], output_dir: str) -> RunResult:
    if config.tapt.enabled:
        raise ValueError(
            "qlora_allam does not support tapt.enabled=true -- MLM continuation doesn't apply to a causal "
            "instruction-tuned decoder the way it does to the encoder backbones. Set tapt.enabled: false."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    quant_cfg = config.quantization
    lora_cfg = config.lora
    if quant_cfg is None or lora_cfg is None:
        raise ValueError("qlora_allam requires both `quantization:` and `lora:` blocks in the config.")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # common gotcha: causal LM tokenizers often lack a dedicated pad token

    model = AutoModelForCausalLM.from_pretrained(backbone_id, quantization_config=bnb_config, device_map={"": 0} if torch.cuda.is_available() else "cpu")
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

    train_examples = build_generative_sft_dataset(tokenizer, split.task1_train, config.model.max_seq_len, config.data.discourse_cues)
    train_ds = Dataset.from_list(train_examples)
    collator = make_generative_collate_fn(tokenizer)

    run_name = f"{config.task}_{config.variant}_{backbone_id.replace('/', '__')}_{seed}"
    tracker = RunTracker(config.wandb, run_name, run_config={"backbone": backbone_id, "seed": seed, "variant": config.variant})

    # No per-epoch eval_strategy/load_best_model_at_end here, unlike this repo's other
    # variants: "best checkpoint" would need score_labels_via_logits (many per-paragraph
    # forward passes) run inside HF Trainer's per-epoch eval hook, which doesn't fit the
    # standard batch-forward compute_metrics interface used elsewhere in this repo. A
    # disclosed simplification: this variant always saves the FINAL epoch's adapter,
    # not a per-epoch-selected best one.
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
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, data_collator=collator,
        callbacks=[make_trainer_callback(tracker)] if config.wandb.enabled else None,
    )
    install_rounded_logging(trainer)
    trainer.train()

    # PEFT adapter-only save (tens of MB), not a merged model (~14GB fp16) -- reload
    # via PeftModel.from_pretrained(base_model, adapter_dir).
    checkpoint_dir = f"{output_dir}/checkpoint"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    model.eval()
    val_probs = np.stack(
        [score_labels_via_logits(r["text"], r["type"], model, tokenizer, config.model.max_seq_len, config.data.discourse_cues) for r in split.task1_val]
    )
    dev_probs = np.stack(
        [score_labels_via_logits(r["text"], r["type"], model, tokenizer, config.model.max_seq_len, config.data.discourse_cues) for r in dev_in]
    )

    tracker.finish()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return RunResult(val_probs=val_probs, dev_probs=dev_probs)


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

    run_results = []
    for backbone_id in config.backbones:
        for seed in config.seeds:
            safe_name = backbone_id.replace("/", "__")
            run_output_dir = f"{config.output_dir}/runs/{safe_name}_seed{seed}"
            run_results.append(train_one_qlora_run(backbone_id, seed, config, split, dev_in, run_output_dir))

    ensemble_and_score(run_results, split, dev_in, config, args.data_dir)


if __name__ == "__main__":
    main()
