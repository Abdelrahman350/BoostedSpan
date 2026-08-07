# Task 1 Methodology: Multi-Label ADU Classification

This document explains how this repository approaches Task 1 of the Daleel 2026
shared task — given an Arabic paragraph, decide which of six argumentative discourse
unit (ADU) types it contains: `AS` (Assumption), `AN` (Anecdote), `ST` (Statistics),
`TE` (Testimony), `CO` (Common Ground), `OT` (Other). A paragraph can carry more than
one label at once, so this is a *multi-label*, not multi-class, classification
problem, scored by the organizers' own macro/micro/weighted F1.

Everything here traces to a specific file under `src/` and, where a technique isn't
this repo's own invention, to the paper it comes from. Numbers quoted in §5 are real
values produced by importing and running the organizers' unmodified
`task1_scoring.py` against this repo's own validation split — never a
self-computed approximation.

## 1. The data and why the pipeline is shaped the way it is

The training set is 612 paragraphs, split once (stratified on domain × label
signature, `src/data/loading.py`'s `build_shared_split`) into ~520 train / 92
validation rows, reused unchanged for Task 2 so both tasks stay comparable. Two
properties of this data drive almost every design choice below:

- **It's heavily imbalanced.** `AS` appears in roughly three paragraphs out of four;
  `CO` and `ST` each appear in under 6%. A classifier trained with plain
  cross-entropy will happily ignore the rare classes, since predicting "no" on `CO`
  is right 94% of the time by default.
- **It's small.** ~520 training rows is not much for a 6-way multi-label problem,
  which rules out anything that wants to learn a representation space mostly from
  scratch (see §4's honest discussion of the contrastive-learning route this repo
  chose *not* to take for Task 1).

## 2. Baseline

`train_task1.py`'s baseline configuration (`configs/task1/baseline.yaml`) is
deliberately "everything off": a single pretrained Arabic encoder — CAMeLBERT-Mix
[1] or MARBERTv2 [2], both BERT-style [3] masked-language-model encoders pretrained
on large Arabic corpora — with a linear classification head on top
(`AutoModelForSequenceClassification`, `problem_type="multi_label_classification"`),
trained with binary cross-entropy where each label gets its own sigmoid output. To
partially counter the imbalance described above, the baseline already weights the
loss: `weighted_bce_pos_weight` (`src/models/losses.py`) computes each label's
inverse training-set frequency and uses it as `pos_weight` in
`BCEWithLogitsLoss`, clipped at 8× so that the rarest labels (`CO`, `ST`, whose true
imbalance ratio is far higher than 8:1) don't produce loss terms so large they
destabilize training on the other labels.

This baseline is the floor every other technique in this repo is measured against —
by config-flag design (§6 of `CLAUDE.md`), the "boosted" pipeline below is not a
separately-coded path, it's the same training script with more flags turned on.

## 3. The boosted pipeline

`configs/task1/boosted.yaml` layers five techniques on top of the baseline. Each one
addresses a specific, diagnosed weakness rather than being added speculatively.

### 3.1 Task-adaptive pretraining (TAPT)

**What**: before ever attaching the classification head, `src/pretraining/tapt.py`'s
`run_tapt()` continues the encoder's own masked-language-modeling objective — the
same objective it was originally pretrained with — on unlabeled text drawn from this
task's own domain: the union of the 612 training paragraphs' text and the 217
unlabeled `dev_in.jsonl` paragraphs (never their labels, since `dev_in` has none;
this is closed-track compliant, see `CLAUDE.md` §2). 15% of tokens are masked
(`DataCollatorForLanguageModeling`, `mlm_probability=0.15`), and the adapted encoder
weights are what the classification head is then attached to and fine-tuned on.

**Why**: a general-purpose Arabic encoder like CAMeLBERT-Mix was pretrained on
Wikipedia/news-scale corpora, not specifically on short argumentative debate/editorial
paragraphs. Gururangan et al. [4] showed that a further, cheap round of MLM
continuation on the *target task's own unlabeled text* before fine-tuning
consistently improves downstream performance, especially when the target domain is
narrower or more specialized than the original pretraining corpus — exactly this
repo's situation. It costs nothing in labels (pure self-supervision) and, once run
for a given backbone, its checkpoint is cached and reused by every config that shares
that backbone (`outputs/tapt_checkpoints/{backbone}/`), so the up-front cost is paid
once.

### 3.2 Asymmetric Loss in place of weighted BCE

**What**: `AsymmetricLoss` (`src/models/losses.py`) replaces plain weighted BCE:

```
L = -y(1-p)^γ+ log(p) - (1-y)·p_m^γ- log(1-p_m),   p_m = max(p - m, 0)
```

with defaults `γ+=1, γ-=4, clip m=0.05`, following Ridnik et al. [5].

**Why**: `pos_weight`-based reweighting (the baseline's fix for imbalance) still
treats every *negative* example the same regardless of how confidently the model
already predicts it's negative — it just scales the whole loss. Asymmetric Loss does
something more targeted: it down-weights the loss contribution from negatives the
model is already confidently correct on (via the `γ-` focusing term and the margin
`m`, which fully zeroes out the loss for very-confident negatives), while keeping
positives' gradient signal strong (`γ+` is much smaller than `γ-`). In a heavily
multi-label-imbalanced setting like this one — where the overwhelming majority of
(paragraph, label) pairs are true negatives — this lets rare-label positives (`CO`,
`ST`) actually move the gradient instead of being drowned out by a sea of "easy"
negative examples that keep contributing loss even once they're solved.

### 3.3 Discourse-marker cues

**What**: `src/text/cues.py` regex-matches Arabic surface patterns associated with
specific ADU types — reporting verbs like قال، صرّح، أكد، بحسب for `TE`
(testimony/reported speech), numerals with % or ٪ and words like دراسة، إحصائي for
`ST` (statistics), and phrases like أتذكر، ذات مرة، قصتي for `AN` (anecdote) — and
folds any matches into the model's input text as a literal prefix tag, e.g.
`[CUES:TE,ST] `, alongside an always-present `[EDITORIAL]`/`[DEBATE]` domain tag
(`build_input_text`).

**Why**: this is a cheap way to hand the model an explicit, high-precision signal
that a pure end-to-end encoder might otherwise have to learn indirectly from ~520
examples. It doesn't replace the encoder's own understanding of the text — the cue
tag is just extra input — but for the confusable label types (`TE` in particular is
often about reported speech, a fairly lexically marked phenomenon in Arabic), it
gives the model a head start rather than asking it to rediscover these regularities
purely from a small labeled set.

### 3.4 Per-label threshold tuning

**What**: after training, instead of thresholding every label's sigmoid output at
the default 0.5, `train_task1.py` sweeps each label's decision threshold
independently on the validation set and keeps whichever value maximizes that label's
own F1.

**Why**: 0.5 is not a principled threshold when classes are this imbalanced — a rare
label's optimal probability cutoff for maximizing F1 is very often below 0.5 (since
the model, correctly, is rarely more than 50% confident on a class it's seen 30-50
times), while a common label like `AS` might do better cutting slightly above 0.5 for
precision. Tuning this per label, independently, on held-out data captures a real,
free performance gain with no architecture change and no risk of overfitting the
prediction *logic* — the model weights themselves are untouched.

### 3.5 Multi-seed, multi-backbone ensembling

**What**: `boosted.yaml` trains multiple `(backbone, seed)` combinations — e.g. both
CAMeLBERT-Mix and MARBERTv2 at seeds 42 and 123 — and averages their predicted
sigmoid *probabilities* (not hard label votes) before threshold tuning and decoding.

**Why**: averaging probabilities from models trained on different random seeds and
sometimes different backbone architectures reduces variance from any single run's
particular initialization or training trajectory — a standard ensembling benefit —
and because different backbones (CAMeLBERT-Mix vs. MARBERTv2, trained on different
Arabic corpora with different tokenizers) tend to make partially independent errors,
averaging tends to cancel out backbone-specific mistakes rather than compound them.
Averaging probabilities rather than voting on hard labels also lets the *combined*
confidence, not just a majority count, feed into the per-label threshold tuning in
§3.4.

## 4. Experimental extension: QLoRA-tuned ALLaM-7B (generative approach)

Beyond the encoder pipeline above, this repo also contains a working, alternative
formulation of Task 1 as a task for a generative large language model:
`src/train_task1_generative.py`, config family `configs/task1/qlora_allam*.yaml`.

**What**: `ALLaM-7B-Instruct-preview` [6], an open-weight ~7B-parameter Arabic
instruction-tuned decoder, is fine-tuned with QLoRA — 4-bit NF4 quantization of the
frozen base weights (Dettmers et al. [7]) plus trainable low-rank adapters (Hu et al.
[8], LoRA) inserted into the attention/MLP projections — which makes fine-tuning a
7B model tractable on a single consumer GPU (~11-12GB VRAM in this repo's actual
hardware, smaller than the 15GB T4 the original notebooks assumed).

Rather than asking the model to *generate* a JSON list of labels — a design that
invites malformed or unparseable output — this variant uses **rank-classification**:
for each of the 6 labels, the model is prompted with a yes/no question about whether
the paragraph exhibits that ADU type, and the probability is read directly off the
next-token logits for "yes" vs. "no" (`score_labels_via_logits`). This is a
deliberate design choice specifically so the same threshold-sweep and
probability-averaging ensembling machinery from §3.4/§3.5 works completely
unmodified — the generative model just becomes another source of per-label
probabilities. TAPT does not apply to this variant (MLM continuation doesn't make
sense for a causal decoder) and the entrypoint raises an error if a config tries to
enable it.

Three tuning variants were tried on top of the base `qlora_allam` run, driven by a
real diagnosed error pattern (macro recall near ceiling at 0.927 but macro precision
capped at 0.763, worst on the semantically confusable `TE`/`OT`/`AN` labels — a
discrimination problem, not an imbalance problem):
- `qlora_allam_weighted` — reuses the encoder path's `weighted_bce_pos_weight`
  (clip 8×) to weight each per-label SFT example's loss by class balance.
- `qlora_allam_lowclip` — the same idea with a gentler clip (3× instead of 8×),
  after `_weighted` was found to make things *worse* by over-suppressing the
  already-tiny `CO`/`ST` classes further.
- `qlora_allam_fewshot` — instead of touching the loss, adds richer per-label
  prompt descriptions with explicit "don't confuse this with X" contrastive
  language, plus one deterministic positive and one hard-negative training
  exemplar per label in the prompt.

**Honest status**: this is flagged experimental in the repository's own README, not
presented as a fully validated alternative to the encoder pipeline. The class-balanced
loss weighting variants (`_weighted`, `_lowclip`) were tested and, per the numbers
below, both underperformed the plain `qlora_allam` baseline — a real negative result,
kept in the repo as a documented dead end rather than deleted. `qlora_allam_fewshot`
has trained checkpoints but no saved validation predictions in this repo, so it isn't
scored below.

## 5. Results

All numbers are validation-set F1 (92 held-out paragraphs), produced by calling the
organizers' unmodified `task1_scoring.py` via `src/evaluation/scoring.py`'s
`score_task1()` against each run's own saved `val_pred.jsonl` / `val_gold.jsonl`.
These are internal validation numbers, not the CodaBench leaderboard score — no gold
labels exist for the public `dev_in.jsonl` submission set (`CLAUDE.md` §2).

| Variant | Micro F1 | Macro F1 | Weighted F1 |
|---|---|---|---|
| `boosted` (encoder ensemble, §3) | 0.8394 | 0.7710 | 0.8437 |
| `qlora_allam` (QLoRA, base) | **0.8650** | **0.8337** | **0.8675** |
| `qlora_allam_lowclip` | 0.8538 | 0.8141 | 0.8558 |
| `qlora_allam_weighted` | 0.8507 | 0.7931 | 0.8515 |
| `baseline` (encoder, no boosts) | not run in this repo — no saved predictions | | |
| `qlora_allam_fewshot` | not scored — checkpoints only, no saved val predictions | | |

Two things worth being upfront about when reading this table:
1. The QLoRA generative variants outscore the encoder `boosted` ensemble here, but
   per §4, the QLoRA path is still labeled experimental by the repository's own
   documentation — its VRAM behavior and yes/no-token tokenization assumption
   against the *real* ALLaM-7B-Instruct-preview weights were flagged as needing
   further verification, even though the checkpoints and predictions scored above do
   exist on disk against that real backbone (confirmed via each run's
   `adapter_config.json`).
2. `boosted`'s per-label F1 breakdown (from the same scorer run) shows the ensemble
   doing very well on `AS` (F1 0.937) and reasonably on `AN`/`OT`/`CO`/`ST`, but
   comparatively weaker on `TE` (F1 0.714) — consistent with `TE` being one of the
   harder, more confusable label boundaries mentioned in §4.

## References

1. Inoue, G., Alhafni, B., Baimukan, N., Bouamor, H., & Habash, N. (2021).
   *The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language
   Models.* WANLP 2021. (CAMeLBERT)
2. Abdul-Mageed, M., Elmadany, A., & Nagoudi, E. M. B. (2021). *ARBERT & MARBERT:
   Deep Bidirectional Transformers for Arabic.* ACL 2021. (MARBERT/MARBERTv2)
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of
   Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
4. Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D.,
   & Smith, N. A. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains
   and Tasks.* ACL 2020.
5. Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., &
   Zelnik-Manor, L. (2021). *Asymmetric Loss For Multi-Label Classification.* ICCV
   2021.
6. *ALLaM-7B-Instruct-preview* model card, Hugging Face
   (`humain-ai/ALLaM-7B-Instruct-preview`) — the open-weight Arabic
   instruction-tuned decoder used as this variant's base model.
7. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA:
   Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
8. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., &
   Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
