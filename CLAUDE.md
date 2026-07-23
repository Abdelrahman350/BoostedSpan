# CLAUDE.md — sanad-argmining

Project context and build instructions for Claude Code. Read this whole file before writing any code.
If you rename the repo from the `sanad-argmining` working title, update the `pyproject.toml` package name
and this file's title accordingly — nothing else here depends on the specific name.

## 1. What this repo is

A from-scratch reimplementation, as a proper installable Python package with CLI entrypoints, of work that was
previously developed and validated interactively in Colab notebooks for the **Daleel 2026 shared task**
(QatarDebate / ArabicNLP 2026, closed track):

- **Task 1** — multi-label paragraph classification. Given an Arabic paragraph, predict which of 6 argumentative
  discourse unit (ADU) types it contains: `AS` (Assumption), `AN` (Anecdote), `ST` (Statistics), `TE` (Testimony),
  `CO` (Common Ground), `OT` (Other). Official metric: micro/macro/weighted F1.
- **Task 2** — span detection. Given the same paragraph, predict `(label, start_offset, end_offset)` character
  spans for each ADU instance. Official metric: partial-overlap F1 (not exact-match).

Both tasks share the same 612 labeled training paragraphs and the same unlabeled 217-paragraph dev set
(`dev_in.jsonl`, used only for final CodaBench submission — no gold labels exist for it).

**This is a port, not a from-scratch research effort.** Every technique below was already designed, implemented,
debugged, and (for several) run successfully in a set of Colab notebooks. Those notebooks are the source of truth
for *what to build*; this repo's job is to turn them into clean, reusable, config-driven, testable Python — not to
redesign the approach. Where this document and notebook behavior conflict, treat it as a porting bug and fix the
port, not the spec — but ask if the conflict looks intentional rather than assuming.

## 2. Non-negotiable: closed-track compliance

Every model, script, and config in this repo must stay within the Daleel 2026 closed-track rules:

- **Data**: only the organizers' own `train_task_1.jsonl`, `train_task_2.jsonl`, and `dev_in.jsonl` (from
  `https://github.com/Argmining/Daleel2026`). No external labeled data, no manually created examples.
- **Models**: open-weight only, ≤70B parameters. Everything in scope here is ~135-163M (CAMeLBERT-Mix,
  MARBERTv2) or 7B (ALLaM-7B-Instruct-preview, for a possible future generative-approach port — see §9).
  **Never** call or route through a closed-weight/proprietary model (no OpenAI/Anthropic/Google API calls) for
  *anything*, including data augmentation.
  Anything that needs internet access to a model API is out of scope for this repo by construction.
- **No test-label leakage**: `dev_in.jsonl` text may be used unlabeled (e.g. for TAPT continued pretraining) and
  for final prediction — never for computing a metric, threshold, or loss that assumes knowledge of its labels.
- **No self-training/pseudo-labeling on `dev_in.jsonl`.** This was a deliberate decision made during the
  notebook-development phase to keep the closed-track submission unambiguously conservative. Don't add it without
  an explicit, separate ask.

If a technique you're porting or asked to add would violate any of this, stop and flag it rather than
implementing a workaround.

## 3. Data

Source: `https://github.com/Argmining/Daleel2026` (public, no auth needed). Clone it into `data/raw/` (gitignored)
via `scripts/fetch_data.sh` rather than committing the data to the repo.

```
data/raw/Daleel2026/
├── data/train/train_task_1.jsonl   # 612 rows: {paragraph_id, text, labels: [str], type: "editorial"|"debate"}
├── data/train/train_task_2.jsonl   # 612 rows: {paragraph_id, text, labels: [{label, span_text, start_offset, end_offset}], type}
├── data/dev/dev_in.jsonl           # 217 rows: {paragraph_id, text, type} -- NO labels, this is the submission input
└── evaluation/
    ├── task1_scoring.py            # official Task 1 scorer -- import and use unmodified, never reimplement for final numbers
    └── task2_scoring.py            # official Task 2 scorer -- same
```

**Key dataset facts** (from EDA during notebook development — re-verify against the live repo, don't assume these
are still exactly current, but they shouldn't have changed since the data is frozen for the competition):
- `train_task_1.jsonl` and `train_task_2.jsonl` describe **the same 612 paragraphs** — identical `paragraph_id`s,
  identical `text`, and Task 1's `labels` is exactly the label-set-union of Task 2's per-span labels for every row.
  Build train/val splits **once**, on Task 1, and reuse the same `paragraph_id` partition for Task 2 — don't split
  independently, or the two tasks' validation sets stop being comparable.
- Label distribution is heavily imbalanced: `AS` in ~75% of paragraphs, `CO`/`ST` in <6% each (paragraph-level);
  span-level counts are similarly skewed (`AS`:1739, `OT`:492, `AN`:343, `TE`:313, `CO`:50, `ST`:38 spans).
- Domains: `debate` (357 rows) vs `editorial` (255 rows) — mildly imbalanced, stratify splits on `(domain,
  label-signature)`.
- Paragraph length: mean ~415 chars / ~98 subword tokens (CAMeLBERT tokenizer), 98.9% under 512 tokens, max 1071
  tokens. This is why chunked sliding-window tokenization exists in the Task 2 code path — it matters for ~1% of
  paragraphs, not most of them, so don't let it complicate the common path.
- Task 2: mean ~4.86 spans/paragraph (max 60, one outlier paragraph), mean span length ~78 chars (min 1, max 566).
  ~1% of paragraphs have overlapping gold spans (BIO tagging can't represent this — see §7's known limitations).

## 4. Environment & tooling

- **Package manager: `uv`.** `pyproject.toml` + `uv.lock`, no `requirements.txt`, no conda, no poetry.
  `uv sync` to set up, `uv run python -m train_task1 --config ...` to run things. Modules live flat under
  `src/` (no top-level package namespace) and are imported directly, e.g. `from data.loading import ...`.
- **Config: plain YAML + a small loader, no Hydra/OmegaConf.** One YAML file per experiment configuration (see §6).
  Write a single `load_config(path: str) -> dict` (or a thin dataclass wrapper if you want type-checking, your
  call) in `src/utils/config.py`. Keep it simple — no config composition/inheritance framework;
  if two configs share most of their fields, that's fine, some duplication across YAML files is an acceptable
  trade for "every config is fully readable standalone," which was an explicit request.
- **Pinned dependency versions** (matched to what was actually validated in the notebooks — don't casually bump
  these without re-verifying, transformers/peft/trl move fast and break things):
  `transformers==4.46.3`, `datasets==3.1.0`, `accelerate==1.1.1`, `scikit-learn==1.5.2`, `seqeval==1.2.2`,
  `evaluate==0.4.3`, `pytorch-crf==0.7.2`. Torch: whatever CUDA-matched build the target machine needs (not pinned
  in the notebooks since Colab provides it — pin appropriately for wherever this repo actually runs).
- **Target compute**: the notebooks were built for a free-tier Colab T4 (~15GB VRAM) and everything was sized
  accordingly (small batches + gradient accumulation, gradient checkpointing on Task 2's CRF model, 4-bit QLoRA for
  the 7B generative approach). If this repo is meant to run on different hardware (e.g. multi-GPU, more VRAM),
  ask — batch sizes and `device_map` usage should probably change, and it changes how much you can parallelize the
  multi-backbone/multi-seed grid in §6.

## 5. Target repository structure

```
sanad-argmining/
├── CLAUDE.md                      # this file
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore                     # data/, outputs/, *.jsonl (except configs/small fixtures), wandb/mlruns/ if used
├── configs/
│   ├── task1/
│   │   ├── baseline.yaml          # v1: single CAMeLBERT-Mix, weighted BCE, no TAPT/ensembling
│   │   └── boosted.yaml           # v2: TAPT + ASL + discourse cues + multi-seed + multi-backbone ensemble
│   └── task2/
│       ├── baseline.yaml          # v1: single CAMeLBERT-Mix, plain BIO + softmax token classification
│       ├── boosted_crf.yaml       # v2: TAPT + CRF + discourse cues + multi-seed + multi-backbone ensemble
│       ├── enhanced_track_a.yaml  # v3 Track A: boosted_crf + boundary jitter + weighted voting + post-processing
│       └── enhanced_track_b.yaml  # v3 Track B: two-stage boundary-detection -> span-type-classification
├── src/
│   ├── data/
│   │   ├── loading.py             # read_jsonl, load_task1/2, build_shared_split (stratified, see §3)
│   │   └── augmentation.py        # jitter_spans (boundary-jitter augmentation, Task 2 only)
│   ├── text/
│   │   └── cues.py                # discourse-marker regex cues, build_input_text (domain prefix + cue tag)
│   ├── models/
│   │   ├── losses.py              # AsymmetricLoss, weighted-BCE pos_weight helper
│   │   ├── crf_tagger.py          # TokenClassifierWithCRF
│   │   ├── span_type_classifier.py  # SpanTypeClassifier (Track B stage 2)
│   │   └── multitask.py           # MultiTaskModel (shared trunk, two heads) -- optional, see §9
│   ├── pretraining/
│   │   └── tapt.py                # run_tapt() -- task-adaptive MLM continuation, see §8 for a leak you must not reintroduce
│   ├── postprocessing/
│   │   └── spans.py               # strip_non_content_spans, snap_to_word_boundary, merge_adjacent_same_label,
│   │                               # ensemble_decode_spans (weighted char-level voting), is_word_char (Unicode-category based -- see §8)
│   ├── evaluation/
│   │   └── scoring.py             # thin wrappers around the organizers' task1_scoring.py/task2_scoring.py (import,
│   │                               # don't reimplement) + corpus_partial_overlap_f1 (internal-use-only approximation,
│   │                               # see §8 for why it must never replace the official scorer for reported numbers)
│   ├── train_task1.py             # CLI entrypoint, dispatches on config's `variant` field (baseline | boosted)
│   └── train_task2.py             # CLI entrypoint, dispatches on config's `variant` field
│                                   # (baseline | boosted_crf | enhanced_track_a | enhanced_track_b)
├── scripts/
│   └── fetch_data.sh              # git clone Argmining/Daleel2026 into data/raw/
├── tests/
│   ├── test_data_loading.py       # split reproducibility, task1/task2 paragraph_id alignment
│   ├── test_postprocessing.py     # is_word_char on Arabic letters/diacritics/punctuation (regression test for §8's bug)
│   ├── test_crf_decode.py         # mask/offset alignment (regression test for §8's bug)
│   └── test_scoring.py            # corpus_partial_overlap_f1 against a small hand-built example with known F1
├── outputs/                       # gitignored: checkpoints, predictions, logs
└── data/                          # gitignored: data/raw/ (cloned) + data/processed/ if you add caching
```

## 6. Config schema

One YAML file per experiment. Every config should be runnable standalone (`uv run python -m
train_task1 --config configs/task1/boosted.yaml`) and should fully determine the run — no
required CLI flags beyond `--config` and maybe `--seed-override`/`--output-dir` for convenience. Suggested shape
(adapt as needed, this is a starting point not a rigid contract):

```yaml
task: task1                 # task1 | task2
variant: boosted             # matches the filename, used for logging/output naming
backbones:
  - CAMeL-Lab/bert-base-arabic-camelbert-mix
  - UBC-NLP/MARBERTv2
seeds: [42, 123]
tapt:
  enabled: true
  epochs: 10
  learning_rate: 5.0e-5
  mlm_probability: 0.15
model:
  max_seq_len: 512
  # task1-specific:
  loss: asymmetric            # asymmetric | weighted_bce
  asl_gamma_pos: 1.0
  asl_gamma_neg: 4.0
  asl_clip: 0.05
  # task2-specific (ignored for task1 configs):
  stride: 64
  use_crf: true
  gradient_checkpointing: true
training:
  epochs: 8
  per_device_batch_size: 16
  gradient_accumulation_steps: 1
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
data:
  discourse_cues: true
  jitter_augment: 0           # task2 only; boosted_crf.yaml=0, enhanced_track_a.yaml=2
ensembling:
  enabled: true
  weighting: uniform           # uniform | internal_f1 (internal_f1 only meaningful once trained runs exist)
output_dir: outputs/task1_boosted
```

Baseline configs (`baseline.yaml`) should set `backbones` to a single entry, `seeds` to a single value,
`tapt.enabled: false`, `data.discourse_cues: false`, `ensembling.enabled: false` — i.e. baseline is "everything
off," not a separately-coded path. Resist the temptation to special-case baseline in the training scripts; it
should just be what falls out of a minimal config.

## 7. What to port, and from where

The notebooks are the ground truth for algorithmic details (exact math, exact function bodies where feasible).
If you don't have direct access to the notebook files, everything you need to reimplement is summarized below;
ask if anything here is ambiguous rather than guessing at a reasonable-sounding alternative.

### Task 1 (`train_task1.py`)

- **Baseline**: `AutoModelForSequenceClassification`, `problem_type="multi_label_classification"`,
  `num_labels=6`, standard weighted BCE (`pos_weight` from inverse label frequency, clipped at 8x).
- **Boosted, on top of baseline**:
  - **TAPT**: continue MLM (`AutoModelForMaskedLM` + `DataCollatorForLanguageModeling`, `mlm_probability=0.15`)
    on the *unlabeled* union of train + `dev_in` text (labels never touched), before attaching the classification
    head. Run once per backbone, reused by every downstream config using that backbone.
  - **Asymmetric Loss** in place of weighted BCE:
    `L = -y(1-p)^γ+ log(p) - (1-y)·p_m^γ- log(1-p_m)`, `p_m = max(p-m, 0)`, defaults `γ+=1, γ-=4, m=0.05`.
  - **Discourse-marker cues**: regex-match Arabic surface cues for `TE` (reporting verbs: قال, صرّح, أكد, بحسب...),
    `ST` (numerals + `%`/`٪`, دراسة, إحصائي...), `AN` (أتذكر, ذات مرة, قصتي...); fold matches into the input as a
    `[CUES:...]` prefix tag, same mechanism as the `[EDITORIAL]`/`[DEBATE]` domain-prefix tag.
  - **Per-label threshold tuning**: after training, sweep each label's decision threshold on the validation set
    independently, keep whichever maximizes that label's F1 (don't just use 0.5).
  - **Multi-seed + multi-backbone ensembling**: average sigmoid probabilities across all `(backbone, seed)` runs
    before threshold tuning/decoding — not a post-hoc vote over hard labels.

### Task 2 (`train_task2.py`)

- **Baseline**: `AutoModelForTokenClassification`, standard 13-tag BIO scheme (`O` + `B-`/`I-` × 6 labels),
  independent per-token softmax cross-entropy, greedy argmax decode with a "dangling I-tag recovers as B-tag"
  heuristic.
- **`boosted_crf`, on top of baseline**:
  - TAPT (shared implementation with Task 1's).
  - Discourse-marker cues (shared implementation with Task 1's).
  - **CRF layer** (`torchcrf.CRF`) wrapping the token-classification head — replaces independent-per-token argmax
    with a learned transition matrix + Viterbi decoding at inference. Loss is `-crf(emissions, tags, mask,
    reduction="mean")`.
  - **Sliding-window chunking** for the ~1% of paragraphs over 512 tokens: `return_overflowing_tokens=True,
    stride=64`; merge overlapping-chunk predictions by letting the later chunk win for the overlap region.
  - Multi-seed + multi-backbone ensembling — but spans aren't probability vectors and different backbones have
    different tokenizers, so this is **character-level majority voting** over decoded spans (paint per-character
    label votes from every run's decoded output, take the majority label per character, re-decode contiguous runs
    into final spans), not logit averaging.
- **`enhanced_track_a`, on top of `boosted_crf`**:
  - **Boundary-jitter augmentation**: for each training paragraph, add a couple of extra copies with gold span
    offsets shifted by ±1-2 characters (clipped to stay in bounds) before building BIO tags — a data-level proxy
    for "boundary smoothing" (the published technique is defined for span-matrix classifiers with a natural
    per-span probability to smooth; a CRF's loss is a global sequence likelihood, so the literal technique doesn't
    transplant — document this distinction if it comes up in write-ups, don't claim the published method).
  - **Weighted ensemble voting**: weight each `(backbone, seed)` run's vote by that run's own internal validation
    partial-overlap F1 (computed with `corpus_partial_overlap_f1`, §5's `scoring.py`), not a uniform vote.
  - **Post-processing**, applied to every decoded span set (val and dev, both tracks):
    1. `strip_non_content_spans` — drop spans whose text contains no actual word character (pure
       whitespace/punctuation noise).
    2. `snap_to_word_boundary` — expand span edges outward while sitting mid-word (fixes WordPiece sub-token
       boundary artifacts).
    3. `merge_adjacent_same_label` — merge same-label spans separated by a tiny gap (≤2 chars) of only
       whitespace/punctuation (fixes BIO fragmentation from a single stray `O` token).
- **`enhanced_track_b`**: a **separate, single-backbone/single-seed** two-stage pipeline, not a variant layered on
  top of Track A:
  1. **Stage A — boundary tagger**: same `TokenClassifierWithCRF` architecture, but a 3-tag scheme (`O`, `B-ARG`,
     `I-ARG`) pooling all 6 label types into one "is this part of any argumentative span" problem — denser
     positive signal per token than any single rare label gets alone.
  2. **Stage B — span-type classifier**: separate small model — shared encoder, mean-pool token representations
     inside a span, concatenate with `[CLS]`, linear layer over 6 types, trained on **gold** span offsets (teacher
     forcing) with inverse-frequency class-weighted cross-entropy.
  3. **Combine**: run Stage A to get untyped spans, run Stage B on each to assign a type, apply the same
     post-processing as Track A.
  - Kept to one backbone/seed deliberately (like the multi-task ablation in §9) — the question this answers is
    "does decoupling boundary-finding from typing help at all," before investing in scaling it to a full grid.

## 8. Known bugs from notebook development — do not reintroduce these

These were real bugs caught during the original notebook development, each fixed after producing an actual
traceback or a wrong-looking test result. Port the *fixed* versions; each is worth a regression test (see §5's
`tests/`).

1. **CRF decode / offset-mapping misalignment.** `torchcrf`'s `.decode()` only returns tag ids for positions
   where `attention_mask == 1`, in order. If you zip decoded tags against the tokenizer's raw
   `offset_mapping` without filtering the offsets by the *same* mask first, every `(token, tag)` pair silently
   shifts by one wherever a batch has padding (batch size > 1) — because special tokens like `CLS`/`SEP` still
   have `attention_mask == 1` and *are* included in the decoded path, but a naive "skip tokens with offset
   `(0, 0)`" filter drops them from the offset side while the CRF path still includes them, desynchronizing the
   two lists. Fix: filter offsets by `attention_mask == 1` first (same filter the CRF used), *then* skip `(0,0)`
   entries inside the loop when populating the char→tag map — never filter by "looks like a special token" as a
   proxy for "the CRF actually masked this position."
2. **`token_type_ids` leaking into a custom model's forward call.** BERT-style tokenizers return
   `token_type_ids` by default alongside `input_ids`/`attention_mask`. Any custom `nn.Module` whose `forward()`
   or `decode()` doesn't explicitly accept it will raise `TypeError: unexpected keyword argument` if you ever
   call it with `**enc` (unpacking the tokenizer's full raw output) instead of naming the exact keys you want.
   Same failure mode for `return_overflowing_tokens=True`, which adds an `overflow_to_sample_mapping` key that
   also isn't a valid model input. Fix both ways: (a) at call sites, build the kwargs dict explicitly
   (`{"input_ids": ..., "attention_mask": ...}`) rather than blind-unpacking tokenizer output; (b) as
   defense-in-depth, give every custom model's `forward`/`decode` a `**kwargs` catch-all so an unexpected key
   doesn't crash a call site you forgot to audit.
3. **GPU memory not released between sequential training runs.** A `transformers.Trainer` instance holds enough
   internal references (optimizer state, the `accelerate` wrapper, callbacks) that letting its local variable go
   out of scope at the end of a function is *not* reliable for freeing GPU memory — plain Python refcounting
   doesn't cut it, and this caused a real `CUDA out of memory` failure when training a second backbone
   immediately after a first one completed successfully. Fix: explicitly `del model, trainer` then
   `gc.collect()` then `torch.cuda.empty_cache()` at the end of every function that trains a model, before
   returning — do this every time, including inside `run_tapt()` specifically (that one was missed once and
   caused the OOM). If you hit an OOM after a crash mid-development, note that Jupyter/IPython's exception
   traceback can independently pin a failed run's tensors in memory (`sys.last_traceback`) — a runtime/process
   restart may be needed on top of the code fix, this isn't purely a code issue in an interactive setting (less
   relevant for this repo's non-interactive scripts, but worth knowing if you're debugging interactively).
4. **Arabic "word character" detection via a hand-picked Unicode range was wrong.** An earlier version of
   `is_word_char` used the regex range `\u0600-\u06FF` (labeled as "the Arabic block") to mean "Arabic letters" —
   but that range is the *entire* Arabic Unicode block, which also contains Arabic punctuation (، ؛ ؟) and other
   non-letter symbols. This silently broke both `strip_non_content_spans` (failed to strip punctuation-only noise
   spans) and `snap_to_word_boundary` (treated commas as word characters, so boundaries wouldn't stop expanding
   at them). Fix: use `unicodedata.category(c)` and exclude Punctuation (`P*`), Separator/whitespace (`Z*`), and
   Control (`C*`) categories — keep everything else, including combining marks (`Mn`, e.g. Arabic tashkeel
   diacritics, which attach to a word rather than separating one). This is script-agnostic and correct by
   construction, unlike a hand-picked codepoint range. Verified: Arabic letters → `True`, Arabic diacritics →
   `True`, whitespace/Arabic-and-Latin punctuation → `False`.
5. **The internal `corpus_partial_overlap_f1` is an approximation, never the reported metric.** It exists purely
   for cheap internal decisions (ensemble run weighting, quick Track A/B comparison) where re-invoking the
   organizers' `task2_scoring.py` via subprocess/file-roundtrip repeatedly would be slower and noisier to work
   with programmatically. **Every number that goes in a report, a leaderboard submission, or a comparison table
   must come from importing and calling the organizers' `task1_scoring.py`/`task2_scoring.py` directly, unmodified.**
   Don't let the internal approximation drift into being treated as authoritative.

## 9. Explicitly out of scope for now

- **The generative-LLM approach** (fine-tuning `ALLaM-7B-Instruct-preview` via QLoRA, instruction-tuned to emit a
  JSON label set for Task 1) exists as a separate, earlier notebook and was *not* included in the "full port"
  scope for this repo's first pass. It's a legitimate closed-track-compliant alternative worth adding later as
  e.g. `configs/task1/qlora_allam.yaml` plus a `train_task1_generative.py` entrypoint (different enough from the
  encoder pipeline — 4-bit quantization, instruction-formatted SFT with loss masking, generation-based inference —
  that it probably shouldn't be forced into the same `train_task1.py` dispatch). Ask before building this; it
  wasn't part of the current scope decision.
- **Multi-task joint encoder** (single shared trunk, two heads, trained jointly on both tasks' losses) was
  explored as a single-run ablation during notebook development, with mixed/unconfirmed results (it was
  deliberately scoped small — one backbone, one seed — specifically to check "is this worth scaling up" before
  investing further, and that follow-up scaling never happened). Worth porting as
  `configs/multitask/joint.yaml` + a dedicated entrypoint if you want the comparison, but it doesn't fit the
  per-task config structure cleanly (it inherently needs both tasks' losses in one run) — treat it as a third,
  separate thing rather than bolting it onto `train_task1.py`/`train_task2.py`.
- **Semi-supervised / self-training on `dev_in.jsonl`** — see §2, deliberately excluded, don't add without an
  explicit separate request.

## 10. Questions for you (the repo owner), not yet resolved

I'm flagging these rather than guessing, since getting them wrong would mean redoing real work:

1. **Experiment tracking**: your other projects use MLflow — want it wired in here too (run params/metrics logged
   per config), or is stdout logging + the `outputs/` directory enough for this repo's scope?
2. **CI**: do you want GitHub Actions running `tests/` (and maybe a lint/format check) on push, or is this
   staying a local-only research repo for now?
3. **License**: none specified yet — public repo needs one before it's actually shareable.
4. **Compute target**: confirm whether this repo should stay sized for a single T4-class GPU (small batches +
   grad accumulation, as the notebooks were), or whether it should assume better hardware and use larger batches /
   parallelize the backbone×seed grid across multiple GPUs.
