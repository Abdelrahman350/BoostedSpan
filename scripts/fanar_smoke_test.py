"""B6 stretch-candidate smoke test: 50 QLoRA SFT steps on QCRI/Fanar-1-9B-Instruct
(Gemma2 architecture), watching for NaN/Inf loss.

Gemma2's attention/final logit soft-capping is documented as fp16-unstable on
bf16-less (Turing) GPUs -- the concern that gated this model behind a smoke test in
the research plan. This machine's actual GPU (compute capability 8.9, Ada Lovelace)
supports bf16 natively, unlike the T4-class hardware CLAUDE.md's other pinned
decisions (fp16-only QLoRA configs) were sized for -- so this smoke test uses bf16
compute dtype specifically to sidestep that instability rather than reproduce it.
FlashAttention-2 is incompatible with Gemma2's soft-capping either way, so
attn_implementation="eager" regardless of dtype.

Not wired into the config system: this is a one-off go/no-go gate, not a repeated
training path. If this passes, a real configs/task1/qlora_fanar.yaml can be written
following qlora_allam.yaml's pattern.
"""

from __future__ import annotations

import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

sys.path.insert(0, "src")
from data.loading import load_task1
from train_task1_generative import build_generative_sft_dataset

BACKBONE = "QCRI/Fanar-1-9B-Instruct"
MAX_STEPS = 50


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    rows = load_task1("data/raw/Daleel2026")
    examples = build_generative_sft_dataset(tokenizer, rows[:60], max_len=1024, discourse_cues=True)
    train_ds = Dataset.from_list(examples)

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

    losses = []

    class WatchTrainer(Trainer):
        def training_step(self, model, inputs, *args, **kwargs):
            loss = super().training_step(model, inputs, *args, **kwargs)
            losses.append(loss.item())
            if not torch.isfinite(loss):
                print(f"NON-FINITE LOSS at step {len(losses)}: {loss.item()}")
                raise SystemExit(1)
            return loss

    args = TrainingArguments(
        output_dir="/tmp/fanar_smoke_test",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2.0e-4,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=5,
        remove_unused_columns=False,
        bf16=True,
        report_to=[],
    )
    trainer = WatchTrainer(model=model, args=args, train_dataset=train_ds, data_collator=collate_fn)
    trainer.train()

    print(f"\nCompleted {len(losses)} steps. First 5 losses: {losses[:5]}. Last 5 losses: {losses[-5:]}")
    print("PASS: no NaN/Inf loss observed." if all(torch.isfinite(torch.tensor(losses))) else "FAIL")


if __name__ == "__main__":
    main()
