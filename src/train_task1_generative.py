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
import os

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from data.loading import LABELS, build_shared_split, load_dev_in, load_task1, load_task2
from models.losses import weighted_bce_pos_weight
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
# Richer, explicitly contrastive descriptions -- used only when
# config.model.fewshot_prompt is true (see build_generative_prompt). Motivation:
# diagnosed that the plain (non-fewshot) qlora_allam baseline's error is
# concentrated in precision, not recall (macro recall 0.927 vs. macro precision
# 0.763), worst on TE/OT/AN specifically -- a discrimination problem (can't tell
# "yes this type" from "sounds similar but isn't"), not an imbalance problem (the
# class-balanced-loss experiments already ruled that out, both underperformed this
# same baseline). These descriptions add the contrastive boundary each fuzzy label
# needs against its most-confusable neighbor.
LABEL_DESCRIPTIONS_RICH: dict[str, str] = {
    "AS": (
        "افتراض أو رأي شخصي يطرحه الكاتب من عنده دون الاستناد إلى مصدر خارجي منقول أو حادثة واقعية "
        "محددة بتفاصيلها (يختلف عن 'شهادة منقولة' التي تُنسب صراحة لشخص آخر، وعن 'قصة شخصية' التي "
        "ترد فيها تفاصيل حدث بعينه)"
    ),
    "AN": (
        "قصة أو واقعة شخصية محددة وقعت فعليًا، تتضمن تفاصيل حدث أو زمان أو مكان "
        "(وليست رأيًا عامًا أو افتراضًا مجردًا)"
    ),
    "ST": "إحصائية أو بيانات رقمية أو نسبة مئوية محددة",
    "TE": (
        "رأي أو معلومة منقولة صراحة عن شخص أو مصدر آخر بذكر القائل أو المصدر (نقل كلام الغير)، "
        "وليست رأي الكاتب نفسه المطروح كافتراض"
    ),
    "CO": "أرضية مشتركة أو حقيقة متفق عليها بين طرفي النقاش",
    "OT": (
        "حجة واضحة لا تندرج تحت الأنواع الأخرى إطلاقًا؛ لا تخْتر هذا النوع إذا كانت الفقرة تحتمل "
        "تصنيفها كافتراض شخصي (AS) أو أحد الأنواع الأخرى"
    ),
}
CUE_DESCRIPTIONS: dict[str, str] = {
    "TE": "نقل كلام عن مصدر آخر",
    "ST": "أرقام أو إحصاءات",
    "AN": "سرد قصة أو تجربة شخصية",
}


def select_fewshot_exemplars(rows: list[dict], labels: list[str]) -> dict[str, dict[str, dict | None]]:
    """One positive + one hard-negative training-row exemplar per label, for
    build_generative_prompt's fewshot_prompt mode. Deterministic (shortest
    paragraph first, tie-broken by paragraph_id) -- no randomness, and drawn only
    from `rows` (the training split), so this stays closed-track compliant the
    same way TAPT's unlabeled-text use already is: no external data, no dev_in
    labels touched.

    Positive = shortest training paragraph that HAS the label (a clean, concise
    demonstration). Hard negative = shortest training paragraph that does NOT
    have the label, restricted to paragraphs that DO have "AS" when the target
    label isn't AS (AS is the dominant/default-looking category and the one most
    often confused with TE/AN/OT per the diagnosed precision gap -- see
    LABEL_DESCRIPTIONS_RICH's docstring), or restricted to paragraphs WITHOUT any
    label-set overlap with AS when the target label IS "AS" itself.
    """
    exemplars: dict[str, dict[str, dict | None]] = {}
    for label in labels:
        positives = sorted((r for r in rows if label in r["labels"]), key=lambda r: (len(r["text"]), r["paragraph_id"]))
        if label == "AS":
            neg_candidates = [r for r in rows if label not in r["labels"] and r["labels"]]
        else:
            neg_candidates = [r for r in rows if label not in r["labels"] and "AS" in r["labels"]]
        negatives = sorted(neg_candidates, key=lambda r: (len(r["text"]), r["paragraph_id"]))
        exemplars[label] = {
            "positive": positives[0] if positives else None,
            "negative": negatives[0] if negatives else None,
        }
    return exemplars


def build_generative_prompt(
    text: str, domain: str, label: str, discourse_cues: bool = True, exemplars: dict[str, dict] | None = None
) -> str:
    """Natural-language instruction framing, not build_input_text's bracket-tag
    prefix -- an instruct-tuned chat model was never trained to interpret
    "[EDITORIAL]"/"[CUES:...]" tags the way an encoder's raw input can absorb them.
    Reuses CUE_PATTERNS as the same regex source of truth as text/cues.py, just
    rendered as a natural-language hint instead of a bracket tag.

    `exemplars` (from select_fewshot_exemplars, only passed when
    config.model.fewshot_prompt is true) switches to the more discriminative
    LABEL_DESCRIPTIONS_RICH and prepends a positive/hard-negative demonstration
    pair for this label before the actual query -- both fixed per label, not
    recomputed per call.
    """
    domain_desc = "افتتاحية صحفية" if domain == "editorial" else "نقاش أو مناظرة"
    cue_note = ""
    if discourse_cues:
        hits = [name for name, pattern in CUE_PATTERNS.items() if pattern.search(text) and name in CUE_DESCRIPTIONS]
        if hits:
            hint = "، ".join(CUE_DESCRIPTIONS[h] for h in hits)
            cue_note = f" (تحتوي الفقرة على مؤشرات لغوية قد تدل على: {hint})"

    fewshot_block = ""
    if exemplars is not None:
        label_desc = LABEL_DESCRIPTIONS_RICH[label]
        pos, neg = exemplars.get(label, {}).get("positive"), exemplars.get(label, {}).get("negative")
        demo_parts = []
        if pos is not None:
            demo_parts.append(f"مثال إيجابي:\n{pos['text']}\nهل يحتوي على {label_desc}؟\nالإجابة: نعم")
        if neg is not None:
            demo_parts.append(f"مثال سلبي:\n{neg['text']}\nهل يحتوي على {label_desc}؟\nالإجابة: لا")
        if demo_parts:
            fewshot_block = "أمثلة توضيحية:\n\n" + "\n\n".join(demo_parts) + "\n\n---\n\n"
    else:
        label_desc = LABEL_DESCRIPTIONS[label]

    return (
        f"{fewshot_block}"
        f"فيما يلي فقرة من {domain_desc}:\n\n{text}\n\n"
        f"هل تحتوي هذه الفقرة على عنصر حجاجي من النوع التالي: {label_desc}؟{cue_note}\n"
        f"أجب بكلمة واحدة فقط: نعم أو لا.\n"
        f"الإجابة:"
    )


def build_sft_example(tokenizer, prompt: str, completion: str, max_len: int, weight: float = 1.0) -> dict:
    """Loss only on completion tokens (-100 on the prompt, this repo's existing
    ignore-index convention -- see BIO's -100 usage elsewhere). Tokenizes the prompt
    alone to get its exact token count, then the full prompt+completion string,
    slicing at that boundary -- an approximation since tokenization near a boundary
    can shift by a token or two versus tokenizing the pieces jointly; acceptable here
    since we only need an approximate split, not exact alignment.

    `weight` is carried through unused by plain Trainer (default 1.0, harmless) --
    only WeightedQLoRATrainer (class_balanced_sft) reads it.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels, "weight": weight}


def build_generative_sft_dataset(
    tokenizer, rows: list[dict], max_len: int, discourse_cues: bool, class_balanced: bool = False, clip: float = 8.0,
    exemplars: dict[str, dict] | None = None,
) -> list[dict]:
    """One training example per (paragraph, label) pair -- completion is the gold
    yes/no answer token for that label, matching how score_labels_via_logits later
    probes each label independently at inference.

    class_balanced=True attaches a per-example loss weight (models/losses.py's
    weighted_bce_pos_weight, same clipped-inverse-frequency formula the encoder
    path already uses) -- >1x for a label's minority-positive answer (e.g. CO/ST's
    "yes"), <1x for a label's majority-positive answer (e.g. AS's "yes", where "no"
    is actually the minority -- pos_weight naturally comes out <1 there, correctly
    downweighting AS's dominant class instead of only ever upweighting rarity).
    `clip` (config.model.class_balanced_sft_clip) defaults to the encoder path's
    8.0 but is expected to need a gentler value here -- see that config field's
    docstring for why the encoder path's clip over-corrected on CO/ST.

    `exemplars` (from select_fewshot_exemplars, only when config.model.fewshot_prompt
    is true) is a FIXED dict reused identically for every row -- a training row that
    happens to be its own label's exemplar will see itself as the demo; a disclosed,
    accepted simplification (affects at most ~12 of ~520 rows), not engineered around.
    """
    pos_weight = weighted_bce_pos_weight(rows, LABELS, clip=clip) if class_balanced else None
    examples = []
    for r in rows:
        for i, label in enumerate(LABELS):
            prompt = build_generative_prompt(r["text"], r["type"], label, discourse_cues, exemplars=exemplars)
            is_yes = label in r["labels"]
            answer = " نعم" if is_yes else " لا"
            weight = float(pos_weight[i]) if (class_balanced and is_yes) else 1.0
            examples.append(build_sft_example(tokenizer, prompt, answer, max_len, weight=weight))
    return examples


def make_generative_collate_fn(tokenizer, include_weight: bool = False):
    """include_weight=False (default) keeps the exact original output shape --
    important because plain Trainer's default compute_loss calls model(**inputs)
    directly with no key-filtering, so an unexpected "weight" kwarg would raise
    TypeError against AutoModelForCausalLM.forward (same failure mode as CLAUDE.md
    section 8 bug 2). Only WeightedQLoRATrainer's path needs it, and only that path
    passes include_weight=True."""

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
        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if include_weight:
            out["weight"] = torch.tensor([b.get("weight", 1.0) for b in batch], dtype=torch.float32)
        return out

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
def score_labels_via_logits(
    text: str, domain: str, model, tokenizer, max_len: int, discourse_cues: bool, exemplars: dict[str, dict] | None = None
) -> np.ndarray:
    """Produces the (6,) P(yes)-per-label probability vector that
    train_task1.ensemble_and_score expects in place of a classification head's
    sigmoid output."""
    yes_id, no_id = _yes_no_token_ids(tokenizer)
    device = next(model.parameters()).device
    probs = []
    for label in LABELS:
        prompt = build_generative_prompt(text, domain, label, discourse_cues, exemplars=exemplars)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**enc)
        last_logits = out.logits[0, -1]
        p_yes = torch.softmax(torch.stack([last_logits[yes_id], last_logits[no_id]]), dim=0)[0].item()
        probs.append(p_yes)
    return np.array(probs)


def score_rows(
    rows: list[dict], model, tokenizer, max_len: int, discourse_cues: bool, desc: str = "scoring",
    exemplars: dict[str, dict] | None = None,
) -> np.ndarray:
    """score_labels_via_logits over many rows, with a progress bar. Not cosmetic:
    each row needs len(LABELS)=6 separate forward passes with no natural batching
    (score_labels_via_logits probes one label at a time), so a few hundred rows can
    take tens of minutes with zero visible output otherwise -- this was silent
    before and looked indistinguishable from a hang."""
    return np.stack(
        [
            score_labels_via_logits(r["text"], r["type"], model, tokenizer, max_len, discourse_cues, exemplars=exemplars)
            for r in tqdm(rows, desc=desc)
        ]
    )


class _BestAdapterState:
    """In-memory best-epoch tracker for the QLoRA path, where score_labels_via_logits'
    many-forward-passes-per-row shape doesn't fit Trainer's batched compute_metrics
    interface (see train_one_qlora_run). PEFT adapters are small enough to keep a
    full clone in memory rather than round-tripping through disk every epoch."""

    def __init__(self):
        self.best_f1 = float("-inf")
        self.best_state: dict[str, torch.Tensor] | None = None

    def maybe_update(self, f1: float, state_dict: dict[str, torch.Tensor]) -> bool:
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
            return True
        return False


def _qlora_val_f1(
    model, tokenizer, val_rows: list[dict], max_len: int, discourse_cues: bool, exemplars: dict[str, dict] | None = None
) -> float:
    """Per-epoch proxy for best-adapter selection: macro F1 at a fixed 0.5 threshold
    over score_labels_via_logits' P(yes) outputs -- same shape as
    train_task1.compute_task1_metrics, just computed outside Trainer's eval loop."""
    val_probs = score_rows(val_rows, model, tokenizer, max_len, discourse_cues, desc="scoring val (per-epoch)", exemplars=exemplars)
    val_gold = np.array([[1.0 if l in r["labels"] else 0.0 for l in LABELS] for r in val_rows])
    return f1_score(val_gold, (val_probs > 0.5).astype(int), average="macro", zero_division=0)


def make_qlora_best_adapter_callback(
    tokenizer, val_rows: list[dict], max_len: int, discourse_cues: bool, tracker: _BestAdapterState,
    exemplars: dict[str, dict] | None = None,
):
    from peft import get_peft_model_state_dict
    from transformers import TrainerCallback

    class _BestAdapterCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            model.eval()
            f1 = _qlora_val_f1(model, tokenizer, val_rows, max_len, discourse_cues, exemplars=exemplars)
            tracker.maybe_update(f1, get_peft_model_state_dict(model))
            model.train()

    return _BestAdapterCallback()


class WeightedQLoRATrainer(Trainer):
    """Plain Trainer's default compute_loss passes labels straight into the causal
    LM, which computes its own internal unweighted mean cross-entropy -- no way to
    inject a per-example weight there. Instead: forward with labels=None (skip the
    model's internal loss), replicate the standard causal-LM shift manually
    (predict token t from the logits at t-1), per-token CrossEntropyLoss(reduction=
    "none") (naturally 0 at -100-masked prompt positions), sum per example, divide
    by that example's non-masked token count (so answer-length doesn't bias the
    loss scale), multiply by the example's weight (build_generative_sft_dataset's
    class-balanced pos_weight), mean over the batch."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weight = inputs.pop("weight")
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        per_token_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="none"
        ).view(shift_labels.shape)

        n_supervised = (shift_labels != -100).sum(dim=1).clamp(min=1)
        per_example_loss = per_token_loss.sum(dim=1) / n_supervised
        loss = (per_example_loss * weight).mean()

        return (loss, outputs) if return_outputs else loss


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

    # Fixed per-label positive/hard-negative demonstration pair, reused identically
    # across every row/epoch/reload -- see select_fewshot_exemplars's docstring.
    exemplars = select_fewshot_exemplars(split.task1_train, LABELS) if config.model.fewshot_prompt else None

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )

    checkpoint_best_dir = f"{output_dir}/checkpoint_best"
    checkpoint_last_dir = f"{output_dir}/checkpoint_last"
    # Resumability: training this 7B model can run for 1.5-2+ hours, and the
    # adapter is already fully saved to disk before the (separately slow) val/dev
    # scoring pass runs below -- if that scoring pass was interrupted (e.g. the
    # process was killed), re-running from scratch would needlessly redo the
    # expensive part. Same idempotent "skip if the output already exists" pattern
    # as pretraining/tapt.py's run_tapt.
    resume_only = os.path.isdir(checkpoint_best_dir)
    trainer = None  # only bound in the training branch; deleted unconditionally below

    if resume_only:
        print(f"{checkpoint_best_dir} already exists -- skipping training, reloading the saved adapter to score it.")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_best_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            backbone_id, quantization_config=bnb_config, device_map={"": 0} if torch.cuda.is_available() else "cpu"
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_best_dir)
    else:
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

        class_balanced = config.model.class_balanced_sft
        train_examples = build_generative_sft_dataset(
            tokenizer, split.task1_train, config.model.max_seq_len, config.data.discourse_cues,
            class_balanced=class_balanced, clip=config.model.class_balanced_sft_clip, exemplars=exemplars,
        )
        train_ds = Dataset.from_list(train_examples)
        collator = make_generative_collate_fn(tokenizer, include_weight=class_balanced)

        run_name = f"{config.task}_{config.variant}_{backbone_id.replace('/', '__')}_{seed}"
        tracker = RunTracker(config.wandb, run_name, run_config={"backbone": backbone_id, "seed": seed, "variant": config.variant})

        # No per-epoch eval_strategy/load_best_model_at_end here, unlike this repo's other
        # variants: score_labels_via_logits (many per-paragraph forward passes) doesn't fit
        # the standard batch-forward compute_metrics interface. Instead, when save_best is
        # true, an on_epoch_end callback runs that eval manually and tracks the best
        # in-memory adapter state (see make_qlora_best_adapter_callback / _BestAdapterState)
        # -- PEFT adapters are small enough to keep a full clone in memory across epochs.
        save_best = config.training.save_best_checkpoint
        best_state = _BestAdapterState()
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
        callbacks = [make_trainer_callback(tracker)] if config.wandb.enabled else []
        if save_best:
            callbacks.append(
                make_qlora_best_adapter_callback(
                    tokenizer, split.task1_val, config.model.max_seq_len, config.data.discourse_cues, best_state,
                    exemplars=exemplars,
                )
            )
        trainer_cls = WeightedQLoRATrainer if class_balanced else Trainer
        trainer = trainer_cls(
            model=model, args=args, train_dataset=train_ds, data_collator=collator,
            callbacks=callbacks or None,
        )
        install_rounded_logging(trainer)
        trainer.train()

        # PEFT adapter-only save (tens of MB), not a merged model (~14GB fp16) -- reload
        # via PeftModel.from_pretrained(base_model, adapter_dir).
        # model currently holds the final epoch's weights ("last").
        model.save_pretrained(checkpoint_last_dir)
        tokenizer.save_pretrained(checkpoint_last_dir)

        if save_best:
            from peft import set_peft_model_state_dict

            # Fall back to the final epoch's state if no epoch ever beat float("-inf")
            # (e.g. num_train_epochs=0 in a smoke test) -- best_state.best_state stays None.
            if best_state.best_state is not None:
                set_peft_model_state_dict(model, best_state.best_state)
            model.save_pretrained(checkpoint_best_dir)
            tokenizer.save_pretrained(checkpoint_best_dir)

        tracker.finish()

    # Predictions (and therefore ensembling/threshold-sweep/submission) always come
    # from the best adapter when save_best is enabled, matching the encoder paths.
    model.eval()
    val_probs = score_rows(split.task1_val, model, tokenizer, config.model.max_seq_len, config.data.discourse_cues, desc="scoring val", exemplars=exemplars)
    dev_probs = score_rows(dev_in, model, tokenizer, config.model.max_seq_len, config.data.discourse_cues, desc="scoring dev", exemplars=exemplars)

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
