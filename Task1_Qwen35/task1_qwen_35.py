"""Task 1 Qwen3.5-9B LoRA fine-tuning and evaluation.

This script is a professional Python equivalent of the notebook workflow
and is compatible with the existing BoostedSpan repository style.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import peft
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)

SEED = 42
LABEL_ORDER = ['AS', 'AN', 'ST', 'TE', 'CO', 'OT']
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
NUM_LABELS = len(LABEL_ORDER)
MINORITY_LABELS = {'ST', 'CO'}
MINORITY_REPEAT_FACTOR = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train and evaluate a Qwen3.5-9B LoRA model for Task 1.'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Path to the dataset directory containing train_task_1.jsonl and dev_task_1_ref.jsonl.',
    )
    parser.add_argument(
        '--experiment-dir',
        type=Path,
        default=Path('outputs/task1_qwen35'),
        help='Directory where checkpoints, adapters, and results are written.',
    )
    parser.add_argument(
        '--model-name',
        default='Qwen/Qwen3.5-9B',
        help='Hugging Face model identifier for the base Qwen checkpoint.',
    )
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=1, help='Training batch size.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='AdamW learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='AdamW weight decay.')
    parser.add_argument('--warmup-ratio', type=float, default=0.10, help='Warmup proportion of total steps.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Gradient accumulation steps.')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum token length for training examples.')
    parser.add_argument('--max-new-tokens', type=int, default=48, help='Max new tokens to generate during evaluation.')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank.')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha.')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout.')
    parser.add_argument('--team-name', default='CHANGE_ME', help='Team name for submission metadata.')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed.')
    parser.add_argument('--evaluate-only', action='store_true', help='Skip training and only run evaluation on a saved adapter.')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with file_path.open('r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSON at line {line_number} in {file_path}') from error
    return examples


def validate_labeled_dataset(dataset: list[dict[str, Any]], name: str) -> None:
    required = {'paragraph_id', 'type', 'text', 'labels'}
    valid_domains = {'editorial', 'debate'}
    seen_ids: set[str] = set()
    for index, example in enumerate(dataset):
        missing = required - set(example)
        if missing:
            raise ValueError(f'{name}[{index}] missing {sorted(missing)}')
        pid = example['paragraph_id']
        if pid in seen_ids:
            raise ValueError(f'Duplicate ID {pid} in {name}')
        seen_ids.add(pid)
        if example['type'] not in valid_domains:
            raise ValueError(f'Invalid type in {pid}')
        unknown = set(example['labels']) - set(LABEL_ORDER)
        if unknown:
            raise ValueError(f'Unknown labels in {pid}: {unknown}')


def build_oversampled_training_set(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for example in examples:
        expanded.append(example)
        if set(example['labels']) & MINORITY_LABELS:
            for _ in range(MINORITY_REPEAT_FACTOR - 1):
                expanded.append(dict(example))
    random.Random(SEED).shuffle(expanded)
    return expanded


def ordered_labels(labels: list[str]) -> list[str]:
    label_set = set(labels)
    return [label for label in LABEL_ORDER if label in label_set]


def build_user_prompt(example: dict[str, Any]) -> str:
    return (
        f"نوع النص: {example['type']}"
        f"الفقرة:{example['text'].strip()}"
        "صنّف الفقرة وأعد JSON فقط."
    )


def build_target_text(example: dict[str, Any]) -> str:
    return json.dumps(
        {'labels': ordered_labels(example['labels'])},
        ensure_ascii=False,
        separators=(',', ':'),
    )


def show_label_distribution(train_examples: list[dict[str, Any]], validation_examples: list[dict[str, Any]]) -> None:
    def distribution(examples: list[dict[str, Any]], name: str) -> pd.DataFrame:
        counts = Counter()
        for example in examples:
            counts.update(example['labels'])
        return pd.DataFrame({'dataset': name, 'label': LABEL_ORDER, 'count': [counts[x] for x in LABEL_ORDER]})

    summary = pd.concat([
        distribution(train_examples, 'train'),
        distribution(validation_examples, 'validation'),
    ], ignore_index=True)
    print(summary.to_string(index=False))


def find_language_attention_modules(model: torch.nn.Module) -> list[str]:
    target_suffixes = (
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'out_proj',
        'in_proj_qkv',
    )
    excluded_fragments = ('visual', 'vision', 'image', 'lm_head')
    targets: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lowered_name = module_name.lower()
        if any(fragment in lowered_name for fragment in excluded_fragments):
            continue
        if module_name.endswith(target_suffixes):
            targets.append(module_name)
    return targets


def apply_chat_template_safely(
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tokenize: bool,
    return_tensors: str | None = None,
    return_dict: bool = False,
) -> Any:
    kwargs: dict[str, Any] = {
        'add_generation_prompt': add_generation_prompt,
        'tokenize': tokenize,
    }
    if return_tensors is not None:
        kwargs['return_tensors'] = return_tensors
    if return_dict:
        kwargs['return_dict'] = True
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def build_prompt_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': build_user_prompt(example)}]},
    ]


def build_training_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    messages = build_prompt_messages(example)
    messages.append({
        'role': 'assistant',
        'content': [{'type': 'text', 'text': build_target_text(example)}],
    })
    return messages


class QwenSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], max_length: int) -> None:
        self.examples = examples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        prompt_encoding = apply_chat_template_safely(
            build_prompt_messages(example),
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt',
            return_dict=True,
        )
        full_encoding = apply_chat_template_safely(
            build_training_messages(example),
            add_generation_prompt=False,
            tokenize=True,
            return_tensors='pt',
            return_dict=True,
        )

        input_ids = full_encoding['input_ids'][0]
        attention_mask = full_encoding['attention_mask'][0]
        prompt_length = min(prompt_encoding['input_ids'].shape[-1], input_ids.shape[-1])

        if input_ids.shape[-1] > self.max_length:
            removed = input_ids.shape[-1] - self.max_length
            input_ids = input_ids[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
            prompt_length = max(0, prompt_length - removed)

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


class QwenSFTCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item['input_ids'].shape[0] for item in batch)
        input_ids, attention_masks, labels = [], [], []
        for item in batch:
            pad_len = max_len - item['input_ids'].shape[0]
            input_ids.append(torch.cat([item['input_ids'], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            attention_masks.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
        return {'input_ids': torch.stack(input_ids), 'attention_mask': torch.stack(attention_masks), 'labels': torch.stack(labels)}


def remove_thinking_content(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def parse_label_output(raw_output: str) -> dict[str, Any]:
    cleaned = remove_thinking_content(raw_output)
    cleaned = cleaned.replace('```json', '').replace('```', '').strip()
    match = re.search(r'\{.*?\}', cleaned, flags=re.DOTALL)
    if match is None:
        return {'labels': [], 'valid': False, 'error': 'no_json_object'}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {'labels': [], 'valid': False, 'error': 'invalid_json'}
    if set(parsed.keys()) != {'labels'}:
        return {'labels': [], 'valid': False, 'error': 'unexpected_json_keys'}
    labels = parsed['labels']
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        return {'labels': [], 'valid': False, 'error': 'invalid_labels_type'}
    unknown = set(labels) - set(LABEL_ORDER)
    if unknown:
        return {'labels': [], 'valid': False, 'error': 'unknown_labels:' + ','.join(sorted(unknown))}
    normalized = ordered_labels(labels)
    if len(normalized) != len(set(labels)):
        return {'labels': normalized, 'valid': False, 'error': 'duplicate_labels'}
    return {'labels': normalized, 'valid': True, 'error': None}


def labels_to_multihot(labels: list[str]) -> np.ndarray:
    vector = np.zeros(NUM_LABELS, dtype=np.int32)
    for label in labels:
        vector[LABEL_TO_ID[label]] = 1
    return vector


def generate_one_output(current_model: torch.nn.Module, example: dict[str, Any], max_new_tokens: int) -> str:
    inputs = apply_chat_template_safely(
        build_prompt_messages(example),
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
        return_dict=True,
    )
    inputs = {key: value.to(current_model.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    input_length = inputs['input_ids'].shape[-1]
    output_ids = current_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0, input_length:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_generation(current_model: torch.nn.Module, examples: list[dict[str, Any]], description: str, max_new_tokens: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    current_model.eval()
    current_model.config.use_cache = True
    targets, predictions, diagnostics = [], [], []

    for example in tqdm(examples, desc=description):
        raw = generate_one_output(current_model, example, max_new_tokens=max_new_tokens)
        parsed = parse_label_output(raw)
        targets.append(labels_to_multihot(example['labels']))
        predictions.append(labels_to_multihot(parsed['labels']))
        diagnostics.append({
            'paragraph_id': example['paragraph_id'],
            'type': example['type'],
            'gold_labels': ordered_labels(example['labels']),
            'predicted_labels': parsed['labels'],
            'valid_format': parsed['valid'],
            'parse_error': parsed['error'],
            'raw_output': raw,
        })

    y_true = np.stack(targets)
    y_pred = np.stack(predictions)
    metrics = {
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'samples_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
        'valid_format_rate': float(sum(x['valid_format'] for x in diagnostics) / len(diagnostics)),
    }
    print(classification_report(y_true, y_pred, target_names=LABEL_ORDER, digits=4, zero_division=0))
    return metrics, diagnostics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    global processor  # noqa: WPS420
    global tokenizer  # noqa: WPS420
    global SYSTEM_PROMPT

    experiment_dir = args.experiment_dir
    checkpoint_dir = experiment_dir / 'checkpoints'
    best_adapter_dir = experiment_dir / 'best_adapter'
    results_dir = experiment_dir / 'results'
    submission_dir = experiment_dir / 'submission'

    for directory in [checkpoint_dir, best_adapter_dir, results_dir, submission_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_file = args.data_dir / 'train_task_1.jsonl'
    dev_file = args.data_dir / 'dev_task_1_ref.jsonl'

    print('Train:', train_file, train_file.exists())
    print('Dev:', dev_file, dev_file.exists())
    print('Experiment dir:', experiment_dir)

    train_examples = load_jsonl(train_file)
    validation_examples = load_jsonl(dev_file)
    validate_labeled_dataset(train_examples, 'train')
    validate_labeled_dataset(validation_examples, 'validation')
    assert not ({x['paragraph_id'] for x in train_examples} & {x['paragraph_id'] for x in validation_examples}), 'Train/validation leakage detected.'
    print('Train examples:', len(train_examples))
    print('Validation examples:', len(validation_examples))

    show_label_distribution(train_examples, validation_examples)
    oversampled_train_examples = build_oversampled_training_set(train_examples)
    print('Original train size:', len(train_examples))
    print('Oversampled train size:', len(oversampled_train_examples))

    global processor  # noqa: WPS420
    global tokenizer  # noqa: WPS420
    global SYSTEM_PROMPT

    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    supports_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print('Training dtype:', compute_dtype)

    base_model = AutoModelForMultimodalLM.from_pretrained(
        args.model_name,
        torch_dtype=compute_dtype,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    language_targets = find_language_attention_modules(base_model)
    if not language_targets:
        raise RuntimeError('No trusted language attention projection modules found for LoRA.')
    print('LoRA target modules:', len(language_targets))

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=language_targets,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    non_lora_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad and 'lora_' not in name]
    if non_lora_trainable:
        print('Additional trainable parameters:', non_lora_trainable[:20])
    else:
        print('Only LoRA adapter parameters are trainable.')

    train_dataset = QwenSFTDataset(oversampled_train_examples, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=QwenSFTCollator(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
    )

    if args.evaluate_only:
        inference_base_model = AutoModelForMultimodalLM.from_pretrained(
            args.model_name,
            torch_dtype=compute_dtype,
            device_map='auto',
            low_cpu_mem_usage=True,
        )
        best_model = PeftModel.from_pretrained(inference_base_model, best_adapter_dir)
        best_model.eval()
        best_model.config.use_cache = True
        metrics, diagnostics = evaluate_generation(best_model, validation_examples, 'Final validation', args.max_new_tokens)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        with open(results_dir / 'final_validation_outputs.jsonl', 'w', encoding='utf-8') as writer:
            for item in diagnostics:
                writer.write(json.dumps(item, ensure_ascii=False) + '\n')
        return

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_training_steps = updates_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print('Updates per epoch:', updates_per_epoch)
    print('Total updates:', total_training_steps)

    training_history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = -1
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        print('=' * 90)
        print(f'Epoch {epoch}/{args.epochs}')
        print('=' * 90)

        model.train()
        model.config.use_cache = False
        running_loss = 0.0
        seen_batches = 0

        progress = tqdm(train_loader, desc=f'Training epoch {epoch}')
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            (loss / args.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            seen_batches += 1

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=running_loss / seen_batches, lr=scheduler.get_last_lr()[0])

        checkpoint_epoch_dir = checkpoint_dir / f'epoch_{epoch}'
        if checkpoint_epoch_dir.exists():
            shutil.rmtree(checkpoint_epoch_dir)
        model.save_pretrained(checkpoint_epoch_dir, safe_serialization=True)
        processor.save_pretrained(checkpoint_epoch_dir)

        validation_metrics, validation_diagnostics = evaluate_generation(
            model,
            validation_examples,
            description=f'Validation epoch {epoch}',
            max_new_tokens=args.max_new_tokens,
        )

        epoch_record = {
            'epoch': epoch,
            'train_loss': float(running_loss / max(seen_batches, 1)),
            **validation_metrics,
        }
        training_history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2, ensure_ascii=False))

        with open(results_dir / f'validation_epoch_{epoch}.jsonl', 'w', encoding='utf-8') as writer:
            for item in validation_diagnostics:
                writer.write(json.dumps(item, ensure_ascii=False) + '\n')

        with open(results_dir / 'training_history.json', 'w', encoding='utf-8') as file:
            json.dump(training_history, file, ensure_ascii=False, indent=2)

        if validation_metrics['macro_f1'] > best_macro_f1 + 1e-6:
            best_macro_f1 = validation_metrics['macro_f1']
            best_epoch = epoch
            if best_adapter_dir.exists():
                shutil.rmtree(best_adapter_dir)
            shutil.copytree(checkpoint_epoch_dir, best_adapter_dir)
            with open(best_adapter_dir / 'best_metrics.json', 'w', encoding='utf-8') as file:
                json.dump(epoch_record, file, ensure_ascii=False, indent=2)
            print('Saved new best adapter.')

        gc.collect()
        torch.cuda.empty_cache()

    print('Best epoch:', best_epoch)
    print('Best validation macro_f1:', best_macro_f1)

    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    inference_base_model = AutoModelForMultimodalLM.from_pretrained(
        args.model_name,
        torch_dtype=compute_dtype,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    best_model = PeftModel.from_pretrained(inference_base_model, best_adapter_dir)
    best_model.eval()
    best_model.config.use_cache = True

    final_validation_metrics, final_validation_diagnostics = evaluate_generation(
        best_model,
        validation_examples,
        description='Final validation',
        max_new_tokens=args.max_new_tokens,
    )

    print(json.dumps(final_validation_metrics, indent=2, ensure_ascii=False))
    with open(results_dir / 'final_validation_outputs.jsonl', 'w', encoding='utf-8') as writer:
        for item in final_validation_diagnostics:
            writer.write(json.dumps(item, ensure_ascii=False) + '\n')

    for domain in ['editorial', 'debate']:
        subset = [x for x in validation_examples if x['type'] == domain]
        print('=' * 80)
        print(domain.upper())
        domain_metrics, _ = evaluate_generation(best_model, subset, description=f'{domain} validation', max_new_tokens=args.max_new_tokens)
        print(json.dumps(domain_metrics, indent=2, ensure_ascii=False))


SYSTEM_PROMPT = """
أنت نظام متخصص في تصنيف وحدات الخطاب الحجاجي العربي.
المهمة متعددة التصنيفات؛ قد تنطبق تسمية واحدة أو عدة تسميات على الفقرة.

التسميات المسموح بها فقط:
AS: موقف أو ادعاء أو حجة أساسية يعرضها الكاتب أو المتحدث.
AN: دليل أو مثال أو تفسير غير إحصائي يدعم موقفًا أو حجة.
ST: دليل رقمي أو إحصائي أو نسبة أو كمية تستخدم كدليل.
TE: نقل أو تلخيص أو مناقشة أو رد على كلام طرف أو متحدث آخر.
CO: مقارنة أو تعارض مباشر بين موقفين أو خيارين أو حالتين.
OT: محتوى تنظيمي أو تمهيدي أو ختامي أو خارج الوظائف السابقة.

قواعد الإجابة:
1. أعد JSON صحيحًا فقط دون شرح.
2. استخدم المفتاح labels فقط.
3. لا تستخدم أي تسمية خارج القائمة.
4. رتب التسميات هكذا: AS ثم AN ثم ST ثم TE ثم CO ثم OT.
5. استخدم قائمة فارغة فقط إذا لم تنطبق أي تسمية.

الشكل المطلوب:
{"labels":["AS","TE"]}
""".strip()


if __name__ == '__main__':
    main()
