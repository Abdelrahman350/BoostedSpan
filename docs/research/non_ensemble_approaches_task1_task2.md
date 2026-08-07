# Non-ensemble contrastive / generative / RL approaches for Task 1 & Task 2

Research report only — no code in this repo was written or changed to produce this.
Scope: for each task, evaluate contrastive learning, generative formulations, and
reinforcement learning as **single-backbone, single-seed** (no probability/span-vote
averaging) alternatives to the current best ensembled configs
(`configs/task1/boosted.yaml`, `configs/task2/enhanced_track_a.yaml`).

Every angle below is checked against this repo's actual constraints:
- **Closed-track compliance** (CLAUDE.md §2): organizers' data only, open-weight
  ≤70B models, no leakage from `dev_in.jsonl` (unlabeled) or the local
  `dev_task_1_ref.jsonl`/`dev_task_2_ref.jsonl` files (present but explicitly
  off-limits without a separate ask).
- **Data size**: 612 labeled paragraphs total, ~520 after the train/val split.
  Small for anything that wants to learn a representation space from scratch.
- **Hardware**: confirmed via `nvidia-smi` this session — a single 12GB RTX 4080
  Laptop, not the notebooks' assumed 15GB T4. Tighter than CLAUDE.md's sizing
  assumes.

## Baseline: the single-model number to beat

Nothing needs to be built to get this — the existing config schema already supports
it (CLAUDE.md §6: baseline configs are "everything off," boosted configs are the
same code path with more turned on). Set `backbones` and `seeds` to length 1 and
`ensembling.enabled: false` in `boosted.yaml` / `enhanced_track_a.yaml`, or just
read the per-run numbers already logged during an ensembled run before averaging.
Observed this session: Task 1's single (CAMeLBERT-mix, seed 42) `boosted` run hit
`eval_f1_macro ≈ 0.758` on the per-epoch 0.5-threshold proxy metric, pre-ensemble.
That — or the equivalent single-run number from a fresh `enhanced_track_a` run for
Task 2 — is the real number any of the approaches below need to beat; the ensembled
scores reported by `evaluation/scoring.py`'s official scorer are a different,
higher bar since they benefit from 4-way averaging.

---

## Task 1 — multi-label ADU classification

### Contrastive

Two distinct contrastive ideas apply here, at different pipeline stages:

1. **Supervised contrastive loss (SupCon) on the classification head.** Add a
   contrastive term over paragraph `[CLS]` embeddings, pulling paragraphs with
   overlapping label sets together and pushing disjoint-label-set paragraphs
   apart, jointly with (or as a pretraining phase before) the existing
   `AsymmetricLoss`/weighted-BCE head (`src/models/losses.py`). Standard SupCon is
   defined for single-label classification (same class = positive pair); Task 1 is
   multi-label, so this needs a multi-label variant — e.g. weight positive pairs by
   Jaccard overlap of their label sets rather than exact match. This is a genuine
   design decision, not a drop-in loss swap.
2. **Unsupervised contrastive continued-pretraining (SimCSE-style)**, replacing or
   augmenting TAPT's MLM objective (`src/pretraining/tapt.py`) on the same
   unlabeled train+`dev_in` text: two dropout-noised forward passes of the same
   paragraph are the positive pair, other paragraphs in the batch are negatives.
   This is a closer drop-in than (1) — it only touches the TAPT stage, the
   downstream classification head is untouched — and stays within CLAUDE.md §2's
   compliance boundary the same way MLM TAPT already does (unlabeled text only,
   `dev_in.jsonl` labels never touched).

**Feasibility**: (2) is low-risk and directly comparable to the existing TAPT
ablation the boosted config already runs — cheap to try, plausible modest gain, and
it's an established published technique (SimCSE) rather than a from-scratch design.
(1) is higher-design-risk (no established multi-label SupCon formula in the
literature that maps cleanly onto 6 partially-overlapping labels) and 520 training
rows is thin for learning a good embedding-space structure via a genuinely
from-scratch contrastive term — likely to be finicky to tune (temperature,
positive-pair-weighting scheme) for uncertain payoff on a 6-way multi-label problem
this small.

### Generative

The QLoRA rank-classification approach (`train_task1_generative.py`) already *is*
the generative single-model candidate for Task 1 — it's a legitimate, already-built
alternative, not something this report needs to invent. Its design (score P(yes) per
label via next-token logits rather than free-form JSON generation, per the file's
own docstring) sidesteps the usual generative-classification failure mode
(malformed/unparseable output) entirely. What's genuinely unverified, per the code's
own disclosed caveats: real VRAM fit for the full 7B model on this GPU (being
tested as of this session), and whether `" نعم"`/`" لا"` tokenize to stable
single trailing tokens in this exact prompt position (`_yes_no_token_ids`'s
docstring flags this explicitly — not yet checked against the real ALLaM tokenizer).

**Feasibility**: highest of the three Task 1 angles, precisely because it's already
built and mid-validation rather than proposed. The open question isn't "should we
try generative for Task 1" — it's "does the already-built implementation actually
work as designed," which is an execution/verification task, not a research one.

### Reinforcement learning

Framing: policy-gradient fine-tuning on top of the QLoRA generative model, reward =
macro-F1 (or per-label F1) computed against the training-split labels (or a
held-out slice of them) — this keeps the reward closed-track compliant since it
needs no external reward model or preference data, just the task's own metric.

**Feasibility**: weakest of the three angles for Task 1, for two compounding
reasons. First, sample efficiency — ~520 training paragraphs is very little signal
for policy-gradient methods (PPO/GRPO), which typically need thousands of reward
evaluations to reduce gradient variance to something usable; a 6-way yes/no
rank-classification setup already gets a full gradient signal per example via
supervised cross-entropy, so RL would be trading a dense, low-variance training
signal for a sparse, high-variance one, for a problem that doesn't obviously need
sequential-decision framing (there's no multi-step generation/exploration structure
being optimized — the SFT setup already directly optimizes what RL would only
optimize *indirectly* through a scalar reward). Second, engineering cost — a full
PPO/GRPO loop is real infrastructure (value model or advantage estimation, KL
penalty against the SFT reference, rollout sampling) that doesn't exist anywhere in
this repo yet. A cheaper stepping stone exists — **best-of-n rejection-sampling
fine-tuning**: sample N completions per training prompt from the SFT'd model, keep
the reward-maximizing one per example, do one more supervised fine-tuning pass on
the filtered set. This is much simpler to implement (no value model, no rollout
loop) and is a reasonable "is RL-flavored fine-tuning worth it at all" probe before
committing to full RL — but even this is unlikely to beat plain SFT by much on a
task this small and this cleanly supervised.

---

## Task 2 — span detection

### Contrastive

This is the strongest-fit contrastive angle across both tasks, because the
groundwork already exists: both `SpanTypeClassifier` (Track B stage B,
`src/models/span_type_classifier.py`) and `SpanScorerModel`
(`src/models/span_scorer.py`) already produce a per-span embedding (mean-pooled
span tokens concatenated with `[CLS]`, via `pool_span_means`'s vectorized
cumsum trick) before the final linear classifier — a contrastive loss can act
directly on that representation with no architectural change, just an added loss
term. Concretely: supervised contrastive loss over span representations, same-label
gold spans pulled together, different-label spans and null/negative candidates
(`SpanScorerModel` already samples hard negatives at partial IoU 0.1–0.5 via
`sample_training_candidates`, `span_scorer.py:105-138`) pushed apart. The hard
negatives already being mined for the classification loss are exactly the useful
negatives a contrastive term wants too — this reuses existing data-prep work rather
than requiring new negative-mining logic.

**Feasibility**: good. Span-level counts are more numerous than paragraph-level
Task 1 counts (1739 `AS` spans, but only 38–50 for `CO`/`ST`), so the same
small-CO/ST-count risk applies, but the existing `hard_negative_fraction`/
`negative_sampling_ratio` knobs (`configs/task2/span_scorer.yaml`) already navigate
a similar tradeoff for the plain classification loss, so there's a template to
extend rather than a blank page.

### Generative

LLM-based span extraction is a much harder fit for Task 2 than the rank-
classification framing was for Task 1, and the report should be explicit about why:
LLMs do not reliably emit correct character offsets by direct generation (no model
reliably "counts characters" over an Arabic paragraph well enough to hit exact
`start_offset`/`end_offset` integers, and the official metric is partial-overlap
F1, so even small offset generation errors compound). The realistic design is
**quote-and-locate**: prompt the model to output the quoted span *text* (and its
label) rather than offsets, then recover `(start_offset, end_offset)` by string-
searching for that quoted text in the source paragraph (first occurrence, or
closest-match if the model paraphrases slightly — a real failure mode to handle).
This sidesteps offset generation but introduces two new problems needing real
design: (a) what to do when the quoted text doesn't appear verbatim (paraphrase,
whitespace/diacritic normalization mismatches — Arabic tashkeel makes this
particularly likely), and (b) multi-span, multi-label paragraphs need a structured
multi-span output format the model has to get right in one generation, which is a
harder generation target than Task 1's one-label-at-a-time yes/no scoring.

**Feasibility**: weakest of the Task 2 angles as a from-scratch design (unlike Task
1's generative path, there's no existing implementation here to build on — this
would be new code, new prompt design, and new fuzzy-matching/normalization logic
for the quote-locate step, on top of the same 7B-model VRAM/compute constraints
already being stress-tested by the Task 1 QLoRA run).

### Reinforcement learning

Framing: policy-gradient over the BIO tagging decision sequence, reward =
partial-overlap F1 on the decoded spans, as an alternative to the CRF's Viterbi
decoding (`src/models/crf_tagger.py`).

**Feasibility**: weakest of all six angles in this report. The CRF layer already
directly models exactly what RL would have to relearn from a much noisier signal —
tag-transition structure (`torchcrf.CRF`'s learned transition matrix) is a closed-
form, exactly-computed component of the CRF's loss (`-crf(emissions, tags, mask,
reduction="mean")`, per CLAUDE.md §7), trained with full per-token gradient signal.
Replacing that with policy-gradient over a scalar per-paragraph partial-overlap F1
reward throws away that dense signal in exchange for a sparse one that only fires
once per full paragraph decode, and the known bug class this repo already
regression-tests for (CLAUDE.md §8 bug 1: CRF decode/offset-mapping
mask-desynchronization) shows how easy this decode path already is to get subtly
wrong even in the current supervised setup — adding RL on top increases surface
area for exactly that class of bug without a clear mechanism for why it would beat
the CRF's already-structure-aware training signal on 520 training paragraphs.

---

## Recommendation

Ranked by expected-benefit-vs-effort-vs-risk, given this repo's actual data size,
hardware, and closed-track constraints:

1. **Task 1 generative (QLoRA), verify-not-redesign.** Already built
   (`train_task1_generative.py`), already mid-validation this session. The highest-
   value next step isn't new research — it's finishing verification of the two
   disclosed unknowns (VRAM fit, yes/no token-id stability) and reading real
   `eval_f1_macro` numbers once that run completes.
2. **Task 2 contrastive (SupCon on span representations).** Best-fit new idea in
   this report — reuses `SpanScorerModel`'s existing span-pooling and hard-negative
   mining, adds one loss term, no new data pipeline. Worth a real implementation
   plan if the user wants to pursue one new technique.
3. **Task 1 contrastive, SimCSE-style TAPT replacement.** Cheap, low-risk,
   directly comparable to the existing TAPT ablation already in the config schema
   — a reasonable second choice if more than one new angle is wanted, but expected
   gains are likely smaller than (2) since it only touches pretraining, not the
   task-specific signal.
4. **Task 1 contrastive, multi-label SupCon on the classification head** and
   **Task 1 RL (rejection-sampling fine-tuning as the RL-adjacent stepping
   stone)** — both plausible but genuinely exploratory, no strong literature
   template for the multi-label SupCon case, uncertain payoff over plain SFT for
   the RL case given how small and cleanly supervised this dataset already is.
5. **Task 2 generative (quote-and-locate span extraction)** and **Task 2 RL
   (policy-gradient over BIO decisions)** — not recommended. Both require
   substantial new infrastructure (fuzzy span-matching logic; a full RL rollout
   loop) to replace components (the CRF's structured decode; a well-defined
   generation target) that are already well-suited to their current supervised
   formulations, with no clear mechanism for why either would win on 520 training
   paragraphs.

**Bottom line**: don't chase all six. Finish verifying the Task 1 QLoRA run that's
already in flight, and if one new from-scratch technique is worth building next,
Task 2's SupCon-on-span-representations is the strongest candidate — it's the only
angle in this report that's both a genuine literature-backed technique *and* a
near-drop-in given this repo's existing `SpanScorerModel` architecture.
