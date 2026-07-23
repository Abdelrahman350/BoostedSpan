"""Task-adaptive pretraining: continue MLM on unlabeled text before attaching a head.

Shared, byte-identical between Task 1 and Task 2 in the source notebooks -- this is
the single biggest de-duplication point in the port. Labels are never touched (MLM is
unsupervised); the unlabeled corpus is the union of train + dev_in text, closed-track
compliant per CLAUDE.md section 2.
"""

from __future__ import annotations

import gc
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.config import TaptConfig


def run_tapt(backbone_id: str, unlabeled_texts: list[str], out_dir: str, config: TaptConfig, base_seed: int = 42) -> str:
    """Continue MLM on unlabeled_texts. Idempotent: returns out_dir immediately if it
    already holds a checkpoint (matches the notebooks' own skip-if-exists guard)."""
    if os.path.isdir(out_dir):
        return out_dir

    tokenizer = AutoTokenizer.from_pretrained(backbone_id)
    enc = tokenizer(unlabeled_texts, truncation=True, max_length=512, padding=False)
    dataset = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    model = AutoModelForMaskedLM.from_pretrained(backbone_id)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability)

    args = TrainingArguments(
        output_dir=f"{out_dir}_run",
        num_train_epochs=config.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch size 16, at 1/4 the peak activation memory (T4 sizing)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=config.learning_rate,
        warmup_ratio=0.1,
        logging_steps=20,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        seed=base_seed,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Trainer holds enough internal references (optimizer state, accelerate wrapper, callbacks)
    # that plain refcounting on function return isn't reliable for freeing GPU memory.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir
