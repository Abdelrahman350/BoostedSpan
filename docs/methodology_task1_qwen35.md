# Task 1 Qwen35 Methodology

This document describes the methodology implemented by `src/task1_qwen_35.py`.
It covers the data pipeline, prompt design, model setup, fine-tuning strategy,
evaluation protocol, and output expectations for Task 1.

## 1. Purpose and task definition

`src/task1_qwen_35.py` is built for **Task 1** of the Daleel 2026 shared task:
Arabic paragraph-level multi-label classification into six argumentative discourse
unit types: `AS`, `AN`, `ST`, `TE`, `CO`, `OT`.

The script uses the Qwen3.5-9B pretrained multimodal language model and fine-tunes
it with **LoRA** adapters, producing a small trainable adapter on top of the frozen
base model.

## 2. Data inputs

The script expects a local `data/` directory containing:

- `train_task_1.jsonl` — labeled training paragraphs
- `dev_task_1_ref.jsonl` — labeled validation paragraphs

Each example is validated for the required fields:
`paragraph_id`, `type`, `text`, and `labels`.

The dataset loader also rejects duplicates, invalid domain types, and labels
outside the permitted set.

## 3. Prompt and target formatting

The model is trained with a chat-style prompt template.

### User prompt

The prompt includes:

- the paragraph domain (`editorial` or `debate`)
- the raw text paragraph
- a request to classify the paragraph and return only JSON

Example prompt prefix:

```
نوع النص: debate
الفقرة: ...
صنّف الفقرة وأعد JSON فقط.
```

### Target output

The target is a JSON object with a single key `labels` and a sorted list of
selected labels, e.g.

```json
{"labels":["AS","TE"]}
```

Labels are ordered consistently according to the fixed label order:
`AS`, `AN`, `ST`, `TE`, `CO`, `OT`.

## 4. Model and tokenizer setup

The base model is loaded with:

- `AutoProcessor.from_pretrained(args.model_name)`
- `AutoModelForMultimodalLM.from_pretrained(args.model_name)`

The processor provides the tokenizer and chat-template support. If the tokenizer
lacks a `pad_token_id`, it is assigned from the EOS token and padded on the
right.

The model is loaded in mixed precision using either `bfloat16` (if supported)
or `float16`.

## 5. LoRA adapter configuration

The script dynamically discovers the language attention projection modules to
apply LoRA to, using the model's named modules and excluding visual/image-related
layers.

The LoRA configuration uses:

- `r=16`
- `alpha=32`
- `dropout=0.05`
- `bias='none'`
- `task_type='CAUSAL_LM'`

These adapters are attached to the frozen base model, making the trainable
parameters small and efficient to fine-tune.

## 6. Dataset and batching

A custom `QwenSFTDataset` builds training examples as follows:

- encodes the prompt messages with `processor.apply_chat_template`
- encodes the full prompt plus answer target
- trims examples longer than `max_length`
- sets label tokens to `-100` for the prompt prefix so only the generated labels
  contribute to loss

The `QwenSFTCollator` pads all batch examples to the same length using the
tokenizer pad token and `-100` for label padding.

## 7. Training loop

The main training loop runs for `args.epochs` epochs and includes:

- gradient accumulation over `args.gradient_accumulation_steps`
- AdamW optimizer with `args.learning_rate` and `args.weight_decay`
- linear warmup scheduler with `args.warmup_ratio` of total steps
- gradient clipping at norm 1.0
- checkpoint saving after each epoch

The script keeps a best-model directory at `<experiment_dir>/best_adapter` and
copies the best checkpoint after each epoch if the validation macro F1 improves.

## 8. Validation and evaluation

Validation uses generation and post-processing:

- the model generates new tokens after the prompt
- the generated text is cleaned and parsed for JSON
- only the first JSON object in the model output is used
- invalid or malformed outputs are recorded as parse errors

Predictions are converted to multi-hot vectors and scored with:

- micro F1
- macro F1
- weighted F1
- sample-based F1
- valid-format rate

Validation diagnostics are written to `results/final_validation_outputs.jsonl`.

## 9. Inference and final evaluation

After training completes, the best adapter is loaded with `PeftModel.from_pretrained`
and evaluated again on the validation set.

The final validation metrics and per-domain breakdowns for `editorial`
and `debate` are printed and saved.

## 10. Outputs

The script writes:

- checkpoints under `<experiment_dir>/checkpoints/epoch_{n}`
- the best adapter under `<experiment_dir>/best_adapter`
- validation metrics and history under `<experiment_dir>/results/`
- the final validation outputs file:
  `<experiment_dir>/results/final_validation_outputs.jsonl`

## 11. How to run

From the repo root on branch `task1_qwen35`:

```bash
cd /tmp/BoostedSpan_clone
python3 -c "from src.task1_qwen_35 import main; main()"
```

Or, if `src/task1_qwen_35.py` contains the normal module entrypoint:

```bash
python3 src/task1_qwen_35.py --data-dir data --experiment-dir outputs/task1_qwen35
```

The file `docs/methodology_task1_qwen35.md` is a companion document to
`src/task1_qwen_35.py` and explains the design decisions behind that branch's
Task 1 workflow.
