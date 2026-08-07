# Task 2 Methodology: Span Detection

This document explains how this repository approaches Task 2 of the Daleel 2026
shared task — given the same Arabic paragraphs as Task 1, predict `(label,
start_offset, end_offset)` character spans for each argumentative discourse unit
(ADU) instance they contain. Unlike Task 1, this is a structured-prediction problem:
a paragraph can contain several spans of several types, at arbitrary, possibly
adjacent or nested character positions, scored by the organizers' own
**partial-overlap** F1 (a predicted span doesn't need to match a gold span exactly to
get credit — see `task2_scoring.py`), which shapes several decisions below.

As with the Task 1 document, every technique here traces to a specific file, and
where it isn't this repo's own idea, to the paper it's from. §7's numbers come from
running the organizers' unmodified `task2_scoring.py` against this repo's own saved
validation predictions.

## 1. The data, and why span detection is harder than it looks here

Same 612-paragraph corpus as Task 1, same shared train/val split (`build_shared_split`
in `src/data/loading.py`, reused rather than re-split so both tasks' validation sets
line up on the same paragraphs). Three data properties matter specifically for span
detection:

- **Span-level imbalance is even more extreme than Task 1's paragraph-level
  imbalance**: `AS` spans (1739) outnumber `CO` spans (50) or `ST` spans (38) by
  roughly 35–45×, not the ~15× seen at the paragraph level.
- **~99% of paragraphs are short enough (under 512 subword tokens) for a single
  forward pass, but ~1% are not** (max 1071 tokens) — a sliding-window chunking path
  exists specifically for that long tail, not the common case.
- **~1% of paragraphs have overlapping gold spans** — two different-labeled spans
  whose character ranges intersect. This matters because the standard tagging scheme
  used below (BIO) structurally cannot represent overlapping spans at all; it's a
  known, disclosed limitation of every BIO-based variant in this repo, not an
  oversight.

## 2. Baseline

`train_task2.py`'s baseline (`configs/task2/baseline.yaml`) frames span detection as
per-token sequence labeling: `AutoModelForTokenClassification` with a standard 13-tag
BIO scheme [1] — `O` plus `B-`/`I-` for each of the 6 labels — trained with
independent per-token softmax cross-entropy (no modeling of tag-to-tag dependencies
at all). At inference, decoding is greedy per-token argmax, with a "dangling I-tag
recovers as a B-tag" heuristic: if the model predicts `I-AS` with no open `B-AS` span
before it (or after a different label), it's treated as if it were `B-AS` rather than
discarded (`spans_from_char_to_tag` in `src/postprocessing/spans.py` implements this
uniformly for every decode path in this repo, baseline included).

This baseline was **not run and has no saved predictions** in this repository's
current `outputs/` — the earliest scored artifacts on disk start from `boosted_crf`'s
architectural family (specifically `enhanced_track_a`, since `boosted_crf` itself
also has no saved run in this repo). It's included here because everything after it
is described as what it changes.

## 3. `boosted_crf`: structured decoding and TAPT

`configs/task2/boosted_crf.yaml` (config exists; not run/scored in this repo's
current outputs — see §7's caveat) adds three things to the baseline:

### 3.1 CRF layer

**What**: `TokenClassifierWithCRF` (`src/models/crf_tagger.py`) replaces
independent per-token softmax with a linear-chain Conditional Random Field
[2] (via `torchcrf.CRF`) on top of the same per-token emission scores. Instead of
each token's tag being chosen independently, the CRF learns a tag-to-tag transition
matrix jointly with the emissions and decodes the whole sequence at once with
Viterbi decoding, replacing greedy argmax. Training loss is the negative
log-likelihood of the gold tag sequence under the CRF (`-crf(emissions, tags, mask,
reduction="mean")`).

**Why**: BIO tagging has hard structural constraints a plain per-token classifier
has no way to enforce — e.g., `I-AS` should essentially never immediately follow
`O` or `B-TE` in a well-formed tag sequence, but independent per-token softmax has
no mechanism to learn or enforce that. A CRF's transition matrix learns exactly
these adjacency patterns from the training data and uses them at decode time, which
tends to produce cleaner, more span-shaped predictions instead of fragmented,
inconsistent tag runs. This repo's own regression tests (`tests/test_crf_decode.py`)
specifically guard against a subtle bug caught during development: `torchcrf`'s
`.decode()` only returns tags for positions where `attention_mask == 1`, so
offsets must be filtered by that *same* mask before being paired back up with
decoded tags — filtering by "does this offset look like a special token" instead
silently desynchronizes the two lists whenever a batch has padding
(`offsets_kept_by_mask` in `src/postprocessing/spans.py` is the fix).

### 3.2 TAPT and discourse cues (shared with Task 1)

Both are the exact same implementations described in `docs/methodology_task1.md`
§3.1 and §3.3 (`src/pretraining/tapt.py`, `src/text/cues.py`) — continued MLM
pretraining on unlabeled train+`dev_in` text, and regex-based discourse-marker
prefix tags. They're genuinely shared code between the two tasks (not reimplemented
per-task), so the same rationale applies without repeating it here.

### 3.3 Sliding-window chunking

**What**: for the ~1% of paragraphs exceeding 512 tokens, the tokenizer is called
with `return_overflowing_tokens=True, stride=64`, producing overlapping chunks. Each
chunk is tagged independently, and where two chunks' predictions overlap in
character space, the later chunk's prediction wins (`spans_from_char_to_tag`'s
reconstruction naturally implements "later entry overwrites earlier" since it's
keyed by character offset).

**Why**: rather than truncating long paragraphs (losing labels entirely for
whatever falls past token 512) or complicating the common case with a
general-purpose long-sequence architecture, this only engages for the minority of
paragraphs that need it, keeping the ~99% common path a plain single-pass forward.
"Later chunk wins" is a simple, deterministic tie-break for the ~64-token overlap
region — not claimed to be optimal, just consistent.

### 3.4 Character-level majority-vote ensembling

**What**: multiple `(backbone, seed)` runs' *decoded spans* (not raw logits) are
combined via `ensemble_decode_spans` (`src/postprocessing/spans.py`): every run
"paints" a vote for its predicted label onto every character its spans cover
(optionally weighted, e.g. by that run's own internal validation F1), and each
character keeps whichever label crosses a minimum total-weight threshold; contiguous
same-label runs of characters are then re-merged into final spans.

**Why**: Task 1's ensembling (§3.5 of the Task 1 document) averages raw sigmoid
*probabilities* directly — but that only works because every backbone shares the
same fixed output space (6 independent per-label probabilities). Task 2's different
backbones use different tokenizers with different vocabularies and different
sub-token boundaries, so there's no shared token-index space to average logits over
directly; two backbones simply don't agree on what "position 37" even refers to.
Character offsets, unlike token indices, are backbone-agnostic — every run's output
can be expressed as spans over the *original* paragraph's characters, which is what
makes character-level voting the natural common ground for combining architecturally
different models' span predictions.

## 4. `enhanced_track_a`: augmentation, weighted voting, and post-processing

`configs/task2/enhanced_track_a.yaml` builds on `boosted_crf` with three more
techniques, plus a family of explored variants (§4.4) that push on specific,
diagnosed failure modes of the base recipe.

### 4.1 Boundary-jitter augmentation

**What**: `jitter_spans` (`src/data/augmentation.py`) creates a couple of extra
training copies of each paragraph where every gold span's start/end offsets are
randomly shifted by ±1–2 characters (clipped to stay in-bounds), before BIO tags are
built from them.

**Why, and an important honest caveat**: this is explicitly a *data-level proxy*
for the published "boundary smoothing" technique, not that technique itself. The
literature version of boundary smoothing is defined for span-matrix classifiers that
have a natural per-span probability distribution to smooth over neighboring
boundaries; a CRF's loss is a single global sequence log-likelihood, which has no
equivalent per-span probability object to smooth. Rather than force-fitting the
published method onto an architecture it wasn't designed for, this repo instead
perturbs the *training labels themselves* slightly — training the model to be a
little more tolerant of exactly-where a boundary falls, which is the same underlying
goal (don't over-penalize a prediction that's off by a couple of characters), reached
by a different, architecture-compatible mechanism. This distinction is worth
preserving in any write-up rather than claiming this implements the published
technique.

### 4.2 Internal-F1-weighted ensemble voting

**What**: the same `ensemble_decode_spans` mechanism from §3.4, but each
`(backbone, seed)` run's vote is weighted by that specific run's own internal
validation partial-overlap F1 (computed via `corpus_partial_overlap_f1`,
`src/evaluation/scoring.py`) rather than uniform 1.0 weights.

**Why, and an important scope note**: `corpus_partial_overlap_f1` exists *only* for
this kind of cheap internal decision (which run should get more say in the vote) —
it is never used to produce a reported number. Every F1 in §7 of this document comes
from the organizers' own `task2_scoring.py`, called unmodified. Weighting by a
run's own quality lets a demonstrably stronger run outvote a weaker one on
disagreements, rather than treating all runs as equally trustworthy regardless of
how well they actually did on held-out data.

### 4.3 Post-processing

Applied to every decoded span set, three fixed steps
(`postprocess_spans`, `src/postprocessing/spans.py`):

1. **`strip_non_content_spans`** — discards any predicted span whose text contains
   no actual word character (i.e., pure whitespace/punctuation noise that
   shouldn't have been predicted as a span at all).
2. **`snap_to_word_boundary`** — expands a span's edges outward while they sit
   mid-word, fixing the common artifact of a sub-word (WordPiece-style) tokenizer
   predicting a boundary that lands inside a word rather than at its edge.
3. **`merge_adjacent_same_label`** — merges two same-label spans separated by a
   tiny gap (≤2 characters) of only whitespace/punctuation, fixing BIO
   fragmentation caused by a single stray `O` prediction splitting what should be
   one continuous span.

Steps 1 and 2 both depend on correctly identifying "is this character part of a
word" — `is_word_char` — which this repo deliberately implements via
`unicodedata.category()`, excluding Punctuation/Separator/Control categories, rather
than a hand-picked Unicode code-point range. An earlier version used the range
`؀-ۿ` (informally "the Arabic block") to mean "Arabic letters," but that
range also contains Arabic punctuation (، ؛ ؟), which silently broke both
downstream functions — punctuation-only noise spans weren't stripped, and boundary
expansion didn't stop at commas. The category-based approach is correct by
construction (and script-agnostic), and is regression-tested in
`tests/test_postprocessing.py`.

### 4.4 Explored variants: chasing the precision/recall imbalance and short-span recall

Beyond the named `enhanced_track_a` config, several standalone variants explore
specific, diagnosed weaknesses of the base recipe — each is its own fully
standalone, reproducible config (not a patch applied at runtime):

- **`enhanced_track_a_weighted`** — `enhanced_track_a`'s validation precision
  (0.649) trailed its recall (0.751) even after post-processing. Diagnosis:
  `torchcrf.CRF`'s sequence log-likelihood has no per-tag class weighting at all,
  so rare tags (`CO`/`ST`) get diluted gradient signal purely from their rarity —
  unlike the Task 1 encoder path, which already handles this via Asymmetric
  Loss/weighted BCE. `WeightedCRFTrainer` (`src/models/crf_tagger.py`) adds a
  second, per-tag class-weighted token-level cross-entropy term computed from the
  *same* emissions the CRF already produces (no extra forward pass), added
  alongside — not replacing — the CRF's own loss.
- **`enhanced_track_a_clustered`** — same trained ensemble as `_weighted`, but
  swaps the ensemble *decode* strategy from per-character majority voting to
  span-level clustering (`cluster_ensemble_decode_spans`,
  `src/postprocessing/spans.py`). This exists because of a specific diagnosed
  failure: char-voting was found to drop ~26% of gold spans entirely from the
  ensembled output, concentrated in *short* spans (mean 50 characters vs. 90 for
  correctly matched spans), even though each individual run recalled 96–99% of
  those same spans on its own. The mechanism: independently boundary-jittered
  predictions across runs fragment a short span's per-character vote so no single
  character clears the majority-weight threshold, even though every run found
  something roughly in the right place. Clustering candidate spans by transitive
  character overlap *first*, then counting each run's vote once per cluster
  (rather than diluted per character), and taking the union of a cluster's member
  spans once the cluster clears the weight threshold, fixes this — a naive
  alternative (dilating spans by a couple of characters before char-voting) was
  tried and made things *worse* (F1 0.687 vs. 0.706) by bridging gaps between
  separate, correctly predicted nearby spans as a side effect.
- **`enhanced_track_a_large`** — isolates backbone *capacity* as a lever, distinct
  from backbone *diversity* (already explored via the char-vote/cluster ensembling
  above): swaps the two ~110–163M-parameter base-size backbones for a single
  ~371M-parameter large Arabic BERT (`aubmindlab/bert-large-arabertv02`, 24 layers)
  at 2 seeds.
- **`enhanced_track_a_large_ensemble`** — same large backbone, but 4 seeds instead
  of 2 (matching the run-count of the earlier char-vote ensemble for a clean
  comparison), testing whether more redundancy (not more architectural diversity)
  recovers more of the short-span false negatives that `_large`'s smaller,
  2-run ensemble was still missing.
- **`enhanced_track_a_retyped`** — a boundary/retype hybrid: reloads
  `enhanced_track_a_weighted`'s already-trained ensemble checkpoints purely for
  boundary detection (not retrained), and trains one new, single-backbone/seed
  `SpanTypeClassifier` (Track B's Stage B design, §5.2 below, with the contrastive
  term from §5.3 enabled) whose only job is to re-score each decoded span's *type*
  when its own prediction clears a confidence threshold — decoupling "where is the
  span" from "what type is it" as two separately-optimizable problems.

## 5. `enhanced_track_b`: decoupled two-stage pipeline

Unlike Track A, `enhanced_track_b` is a **separate, single-backbone/single-seed**
pipeline — not a variant layered on top of Track A's ensembling. It answers a
different question: does decoupling *where* a span is from *what type* it is help at
all, deliberately kept small-scale (one backbone, one seed) as a first check before
any decision to scale it up.

### 5.1 Stage A: boundary tagger

**What**: the exact same `TokenClassifierWithCRF` architecture as Track A (§3.1),
but with a 3-tag scheme instead of 13: `O`, `B-ARG`, `I-ARG` — every one of the 6
ADU types pooled into a single "is this token part of *any* argumentative span"
problem.

**Why**: pooling all 6 label types into one boundary-detection problem gives this
stage a much denser positive-example signal per token than any single rare label
(`CO`, `ST`) would get on its own in the 13-tag scheme — boundary-finding and
type-assignment are genuinely different sub-problems, and separating them lets the
boundary detector train on the *union* of all positive spans rather than being
diluted across 6 separate, imbalanced label channels.

### 5.2 Stage B: span-type classifier

**What**: `SpanTypeClassifier` (`src/models/span_type_classifier.py`) — a separate
model (shared encoder architecture, independently trained weights) that mean-pools
the encoder's token representations *inside* a given span, concatenates that with
the `[CLS]` representation, and applies a linear layer over the 6 ADU types. It is
trained via **teacher forcing**: during training it's always given the *gold* span
offsets (never Stage A's predicted boundaries), with inverse-frequency
class-weighted cross-entropy to counter the same span-level imbalance described in
§1.

**Why teacher forcing**: this deliberately isolates "can a model correctly *type* a
span, given that its boundaries are already right" from "can it also find the right
boundaries" — a genuinely different learning problem than an end-to-end BIO tagger
has to solve simultaneously. At inference, Stage A's predicted (untyped) spans are
fed through Stage B to assign each one a type, and the same post-processing from
§4.3 is applied to the combined output.

### 5.3 Contrastive variant

**What**: `enhanced_track_b_contrastive` adds a supervised contrastive loss term
(SupCon, Khosla et al. [3], implemented in `src/models/contrastive.py`) over Stage
B's pre-classifier representation (the same `[CLS]`+mean-pooled-span vector the
cross-entropy loss already sees), pulling same-type span representations together
and pushing different-type ones apart, jointly with the existing class-weighted
cross-entropy.

**Why this stage specifically**: per this repo's own research note
(`docs/research/non_ensemble_approaches_task1_task2.md`), Stage B's span
representation was identified as the strongest-fit target for a contrastive term
across *both* tasks, precisely because the architecture already produces a natural
per-span embedding before the final classifier — no new architecture is needed, only
an added loss term — and because Stage B is teacher-forced on gold offsets, so a
contrastive signal here is unconfounded by any boundary-decode error (unlike, e.g.,
`span_scorer` below, whose representations are affected by its own separate,
currently weak decode quality).

**Disclosed limitation**: SupCon as implemented only sees *in-batch* positives —
`CO`/`ST` have only 38–50 total span instances across the entire 612-paragraph
corpus, so many training batches will contain zero same-label pairs for those two
labels, and the contrastive term contributes essentially nothing for them on those
batches. There's no cross-batch memory bank or class-balanced sampler here; this is
a disclosed, deliberate scope cut for a first pass, not an oversight.

## 6. Experimental extension: enumerate-and-classify span scoring

`src/models/span_scorer.py`, config `configs/task2/span_scorer.yaml`, is a third,
architecturally distinct alternative to both Track A and Track B.

**What**: instead of decoding one BIO tag sequence (which structurally cannot
represent overlapping spans), `SpanScorerModel` enumerates *every* candidate
`(start, end)` span up to a maximum width (48 tokens by default, chosen against the
real training-set span-length distribution — this excludes roughly the longest 2% of
gold spans) and scores each one directly, using the same
mean-pool-span-tokens-concat-`[CLS]` representation pattern as Stage B, vectorized
via a cumulative-sum trick (`pool_span_means`) so scoring hundreds of candidate spans
per paragraph is one batched operation rather than a per-span Python loop. Because
candidate spans carry independent scores rather than sharing one tag sequence, this
representation *can* natively output overlapping spans of different labels — directly
addressing the ~1% BIO-incompatible case from §1 — and its greedy, score-ranked
decode (`decode_spans_from_scores`) only suppresses overlaps above a same-label (any
overlap) or different-label (IoU-threshold) rule, rather than forbidding all overlap
by construction.

**Honest status**: this is the least mature technique in the repository. Per the
repo's own README, it was "verified end-to-end on a real (short) training run" but
"not yet compared against `boosted_crf`/`enhanced_track_a` on a full run," and this
repo's internal research notes record an early full-run score (F1 ≈ 0.066, using the
internal approximation metric, not the official scorer) far below every other
variant reported in §7 — a result explicitly attributed to the technique's own
current weaknesses (e.g. candidate/negative-sampling balance, decode threshold
tuning) rather than to anything wrong with the underlying idea. It has no scored
validation output in this repo's current `outputs/`, so it is not included in §7's
results table; it's documented here as a real, working, but not-yet-competitive
architectural alternative, not a proven technique on par with Track A or B.

## 7. Results

All numbers are validation-set partial-overlap precision/recall/F1 (92 held-out
paragraphs), produced by calling the organizers' unmodified `task2_scoring.py` via
`src/evaluation/scoring.py`'s `score_task2()` against each run's own saved
`val_pred.jsonl` / `val_gold.jsonl`. These are internal validation numbers, not the
CodaBench leaderboard score.

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| `enhanced_track_a` (base) | 0.6494 | 0.7514 | 0.6967 |
| `enhanced_track_a_weighted` (+ class-weighted aux CE) | 0.6651 | 0.7521 | 0.7059 |
| `enhanced_track_a_retyped` (+ Stage-B retyping) | 0.6720 | 0.7567 | 0.7118 |
| `enhanced_track_a_clustered` (+ cluster decode) | 0.7317 | 0.7271 | 0.7294 |
| `enhanced_track_a_large` (large backbone, 2 seeds) | 0.7120 | 0.7601 | 0.7352 |
| `enhanced_track_a_large_ensemble` (large backbone, 4 seeds) | 0.7308 | 0.7572 | **0.7438** |
| `enhanced_track_b_contrastive` (two-stage + SupCon) | 0.6264 | 0.6151 | 0.6207 |
| `baseline` / `boosted_crf` / plain `enhanced_track_b` / `span_scorer` | not run in this repo — no saved predictions | | |
| `enhanced_track_a_mixed_ensemble` | not scored — checkpoints only, no saved val predictions | | |

The clearest pattern in this table: every Track A refinement in §4.4 moved F1
upward over the base `enhanced_track_a` recipe, largely by closing the
precision/recall gap the base recipe started with (precision rising from 0.649 to
0.731 by `_clustered`, without recall collapsing) — consistent with each variant
targeting a specific, previously diagnosed weakness rather than being applied
speculatively. `enhanced_track_b_contrastive`, by contrast, scores below every Track
A variant here; given it's deliberately a single-backbone/single-seed ablation (no
ensembling at all, unlike every Track A number above), this isn't a fair
apples-to-apples comparison of "Track A vs. Track B" so much as a reminder that most
of Track A's score in this table comes from ensembling multiple runs, not from any
single run in isolation.

## References

1. Ramshaw, L. A., & Marcus, M. P. (1995). *Text Chunking Using Transformation-Based
   Learning.* Third Workshop on Very Large Corpora (WVLC). (Origin of the BIO
   tagging scheme.)
2. Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional Random Fields:
   Probabilistic Models for Segmenting and Labeling Sequence Data.* ICML 2001.
3. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot,
   A., Liu, C., & Krishnan, D. (2020). *Supervised Contrastive Learning.* NeurIPS
   2020. (SupCon, `src/models/contrastive.py`.)

Techniques shared with Task 1 — task-adaptive pretraining and discourse-marker
cues — are cited in `docs/methodology_task1.md`'s references (Gururangan et al.
2020; the CAMeLBERT/MARBERT backbone papers) rather than repeated here.
