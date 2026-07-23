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
├── models/          # losses, CRF tagger, span-type classifier
├── pretraining/      # task-adaptive MLM pretraining (TAPT)
├── postprocessing/   # span decode/cleanup, ensemble voting
├── evaluation/        # thin wrappers around the organizers' scorers, submission zip packaging
├── utils/             # config loading, W&B tracking, log rounding
├── train_task1.py
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
```

### Training — Task 2

```bash
uv run python -m train_task2 --config configs/task2/baseline.yaml
uv run python -m train_task2 --config configs/task2/boosted_crf.yaml
uv run python -m train_task2 --config configs/task2/enhanced_track_a.yaml
uv run python -m train_task2 --config configs/task2/enhanced_track_b.yaml
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
training: a single `task_1.jsonl`/`task_2.jsonl` at the zip root, named
`{team_name}_{training_setting}.zip`, with contents and integrity verified
immediately after writing. Off by default — flip it on per-config when you're ready
to submit.
