# sanad-argmining

A from-scratch, installable-package port of validated research notebooks for the
**Daleel 2026 shared task** (QatarDebate / ArabicNLP 2026, closed track):

- **Task 1** — multi-label paragraph classification into 6 argumentative discourse unit
  (ADU) types: `AS`, `AN`, `ST`, `TE`, `CO`, `OT`.
- **Task 2** — span detection: `(label, start_offset, end_offset)` character spans for
  each ADU instance.

See `CLAUDE.md` for the full project spec (repo structure, config schema, algorithm
details, and known bugs from notebook development that this port deliberately avoids
reintroducing).

## Project layout

```
src/
├── data/            # loading, shared train/val split, boundary-jitter augmentation
├── text/            # discourse-marker cues, domain-prefix input composition
├── models/          # losses, CRF tagger, span-type classifier, span-scorer (enumerate-and-classify)
├── pretraining/      # task-adaptive MLM pretraining (TAPT)
├── postprocessing/   # span decode/cleanup, ensemble voting
├── evaluation/        # thin wrappers around the organizers' scorers, submission zip packaging
├── utils/             # config loading, W&B tracking, log rounding
├── train_task1.py
├── train_task1_generative.py   # QLoRA ALLaM-7B rank-classification variant
└── train_task2.py
```

Modules live flat under `src/` (no top-level package namespace) and are imported
directly, e.g. `from data.loading import ...`, `from models.crf_tagger import ...`.

## Command reference

### Setup

```bash
uv sync                    # install runtime dependencies
uv sync --extra dev        # also install dev dependencies (pytest)
```

### Fetch data

```bash
scripts/fetch_data.sh          # clone Argmining/Daleel2026 into data/raw/ (no-op if already present)
scripts/fetch_data.sh --force  # remove data/raw/Daleel2026 and re-clone
```

### Training — Task 1

```bash
uv run python -m train_task1 --config configs/task1/baseline.yaml
uv run python -m train_task1 --config configs/task1/boosted.yaml

# EXPERIMENTAL -- QLoRA ALLaM-7B, rank-classification (see "New architectural
# variants" below). Not yet validated end-to-end against the real 7B model.
uv run python -m train_task1_generative --config configs/task1/qlora_allam.yaml
```

### Training — Task 2

```bash
uv run python -m train_task2 --config configs/task2/baseline.yaml
uv run python -m train_task2 --config configs/task2/boosted_crf.yaml
uv run python -m train_task2 --config configs/task2/enhanced_track_a.yaml
uv run python -m train_task2 --config configs/task2/enhanced_track_b.yaml

# EXPERIMENTAL -- enumerate-and-classify span scoring (see "New architectural
# variants" below). Verified end-to-end on a real (short) training run; not yet
# compared against boosted_crf/enhanced_track_a on a full run.
uv run python -m train_task2 --config configs/task2/span_scorer.yaml
```

### Training — optional flags

Available on both `train_task1` and `train_task2` (in addition to the required `--config`):

```bash
--seed-override SEED     # train_task1 only: override config.seeds with a single seed
--output-dir DIR         # override config.output_dir
--data-dir DIR           # path to the cloned data repo (default: data/raw/Daleel2026)
```

Example:

```bash
uv run python -m train_task1 --config configs/task1/boosted.yaml --seed-override 123 --output-dir outputs/task1_boosted_seed123
```

### Training — installed console scripts

Equivalent to the `-m` invocations above, once the package is installed (`uv sync`):

```bash
uv run sanad-train-task1 --config configs/task1/baseline.yaml
uv run sanad-train-task2 --config configs/task2/boosted_crf.yaml
```

### Tests

```bash
uv run pytest              # run the full test suite
uv run pytest -q           # quiet output
uv run pytest tests/test_postprocessing.py   # run a single test file
```

No GPU, network access, or the real dataset is required — all fixtures are small,
hand-built inline in the test files. The four test modules regression-test the known
bugs documented in `CLAUDE.md` section 8: CRF decode/offset-mapping alignment
(`test_crf_decode.py`), Arabic word-character detection
(`test_postprocessing.py`), the shared train/val split (`test_data_loading.py`), and
the internal partial-overlap F1 approximation (`test_scoring.py`).

### Sanity checks

Validate every config loads without running training:

```bash
uv run python -c "
from utils.config import load_config
import glob
for path in sorted(glob.glob('configs/*/*.yaml')):
    c = load_config(path)
    print(f'{path}: OK (task={c.task}, variant={c.variant})')
"
```

Verify the package imports cleanly:

```bash
uv run python -c "import data.loading, train_task1, train_task2; print('OK')"
```

### Cleanup

```bash
find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
rm -rf .pytest_cache
```

## Notes

`variant` in each config determines behavior purely through config flags
(`tapt.enabled`, `model.loss`/`model.use_crf`, `data.discourse_cues`,
`data.jitter_augment`, `ensembling.*`) — baseline is "everything off", not a
separately-coded path.

TAPT checkpoints are cached once per backbone at `outputs/tapt_checkpoints/{backbone}/`
and shared across configs, so re-running a second config against an already-adapted
backbone skips the expensive retrain.

Set `wandb.enabled: true` in a config to log runs to Weights & Biases (project
`sanad-argmining` by default); it's a true no-op when disabled.

Every reported/leaderboard number comes from the organizers' own
`task1_scoring.py`/`task2_scoring.py` (imported unmodified from the cloned data repo),
never from this package's own internal approximation metrics.

### Checkpointing

`training.save_best_checkpoint: true` (the default in all 6 configs) evaluates every
epoch and keeps only the checkpoint with the best macro F1 on the validation set,
saved to `<run_output_dir>/checkpoint` — a full HF-format checkpoint for Task 1's
`AutoModelForSequenceClassification` (reloadable via `from_pretrained`), or a raw
`state_dict` for Task 2's custom CRF/span-type models (reload by reconstructing the
model class and calling `load_state_dict`). Set it to `false` to skip evaluation and
checkpoint saving entirely, matching the original save_strategy="no" behavior.

### Submission packaging

Set `submission.enabled: true` and `submission.team_name` in a config to package the
run's `dev_in` predictions into a CodaBench-style submission zip at the end of
training: a single `task_1.jsonl`/`task_2.jsonl` at the zip root, with contents and
integrity verified immediately after writing. Off by default — flip it on per-config
when you're ready to submit.

Every `train_*.py` entrypoint writes its zip to **`submissions/dev/`** (Dev phase,
predictions on the 217-row `dev_in.jsonl`), named
`{task}_{variant}_{team_name}_{training_setting}.zip` — e.g.
`submissions/dev/task2_boosted_crf_Nu_Analytics_both.zip`. The `{task}_{variant}`
prefix is what keeps every variant's zip uniquely named inside the same shared
directory; without it, every config would produce an identically-named zip and it'd
be easy to upload the wrong one.

`predict_eval.py` (below) writes to **`submissions/eval/`** instead, with the same
naming convention — the two directories exist specifically so a 217-row Dev
submission and a 213-row Evaluation submission are never one accidental drag-and-drop
away from each other.

### Predicting on a different input file (`predict_eval.py`)

Every `train_*.py` entrypoint only ever predicts on `dev_in.jsonl`, inside the same
process that just trained the model. `predict_eval.py` reloads an **already-trained**
checkpoint (no retraining) and predicts on an arbitrary input file with the same
`{paragraph_id, text, type}` schema — built for the shared task's separate
Evaluation-phase test set (`data/test/test_in.jsonl`, 213 rows, distinct from the
217-row `dev_in.jsonl`; re-run `scripts/fetch_data.sh --force` if your clone predates
this file being added upstream).

```bash
uv run python -m predict_eval --config configs/task1/qlora_allam.yaml --team-name Nu_Analytics --training-setting both
uv run sanad-predict-eval --config configs/task2/boosted_crf.yaml --team-name Nu_Analytics --training-setting both
```

Works for every variant across both tasks — encoder classifiers, the QLoRA/PEFT
generative model, CRF taggers, the two-stage `enhanced_track_b` pipeline, and
`span_scorer` — by reconstructing each model from the same base checkpoint training
started from (`outputs/tapt_checkpoints/{backbone}/` if TAPT was used, otherwise the
raw backbone id) and loading the trained weights on top; see the module docstring for
why custom (non-`PreTrainedModel`) checkpoints need this two-step reload instead of a
plain `.from_pretrained()`. Optional `--test-file`/`--data-dir` override the input
file and data directory; `--config`, `--team-name` are required.

### New architectural variants (experimental)

Two additional variants beyond CLAUDE.md's original scope, aimed at higher-ceiling
performance gains rather than incremental tuning:

**Task 2 `span_scorer`** (`configs/task2/span_scorer.yaml`, `src/models/span_scorer.py`) —
an enumerate-and-classify span scorer replacing `boosted_crf`'s BIO+CRF decode. Scores
every candidate `(start, end)` span up to `span_scorer.max_span_width` tokens directly
(48 by default, chosen empirically against `train_task_2.jsonl`'s real span-length
distribution — excludes ~2% of gold spans), rather than decoding one BIO tag sequence.
Unlike BIO, it can represent the ~1% of paragraphs with overlapping gold spans, and
optimizes more directly for the official partial-overlap F1 metric. Reuses the same
`postprocess_spans`/`ensemble_decode_spans`/`write_submission_zip`/`score_task2`
pipeline as every other Task 2 variant — its output is plain `{label, start_offset,
end_offset}` span dicts. Verified end-to-end on a real (short) training run; not yet
compared against `boosted_crf`/`enhanced_track_a` on a full run — check
`corpus_partial_overlap_f1` (and then the official scorer) before trusting it over the
proven CRF path.

**Task 1 `qlora_allam`** (`configs/task1/qlora_allam.yaml`, `src/train_task1_generative.py`) —
QLoRA (4-bit NF4) fine-tuning of `ALLaM-7B-Instruct-preview`, a generative alternative
to the encoder-classification pipeline. Uses **rank-classification** (scores `P(yes)`
per label via next-token logits) rather than free-form JSON generation, specifically
so `train_task1.py`'s existing per-label threshold sweep and probability-averaging
ensemble work unmodified. TAPT is unsupported for this variant (the entrypoint raises
if `tapt.enabled: true`). The QLoRA mechanism (4-bit load + LoRA + forward/backward)
was verified working on this hardware using a small stand-in causal LM; **VRAM fit
against the real 7B model and the yes/no token-id assumption in
`score_labels_via_logits` both still need verification against the real
`ALLaM-7B-Instruct-preview` tokenizer/weights** before trusting results — see the
docstrings in `train_task1_generative.py` for the exact open risks.

### Running the branch-local Task 1 script

This repo now includes `src/task1_qwen_35.py` on the `task1_qwen35` branch.
Run it from the repo root using the current branch and the same Python environment:

```bash
cd /tmp/BoostedSpan_clone
python3 -c "from src.task1_qwen_35 import main; main()"
```

If you prefer a direct script invocation after adding the missing module entrypoint,
append `if __name__ == '__main__': main()` to the bottom of `src/task1_qwen_35.py`,
then run:

```bash
cd /tmp/BoostedSpan_clone
python3 src/task1_qwen_35.py --data-dir data --experiment-dir outputs/task1_qwen35
```

The script expects the `data/` directory to contain `train_task_1.jsonl` and
`dev_task_1_ref.jsonl`.
