"""Plain YAML + dataclass config loading -- no Hydra/OmegaConf.

Every experiment is one standalone YAML file (see CLAUDE.md section 6). Unknown or
missing-but-required keys fail loudly at load time rather than silently defaulting,
so a typo in a config file surfaces immediately instead of during a multi-hour run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class TaptConfig:
    enabled: bool = False
    epochs: int = 10
    learning_rate: float = 5.0e-5
    mlm_probability: float = 0.15


@dataclass
class ModelConfig:
    max_seq_len: int = 512
    # task1-specific
    loss: str = "weighted_bce"  # "weighted_bce" | "asymmetric"
    asl_gamma_pos: float = 1.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    # task1 qlora_allam-only: class-balanced per-example loss weighting for the SFT
    # yes/no rank-classification objective (see train_task1_generative.py's
    # WeightedQLoRATrainer). Ignored everywhere else.
    class_balanced_sft: bool = False
    # Clip for weighted_bce_pos_weight's inverse-frequency formula when
    # class_balanced_sft is true. Default 8.0 matches the encoder path
    # (models/losses.py), but that value was diagnosed as too aggressive for
    # QLoRA's per-example SFT weighting specifically on tiny-support labels
    # (CO/ST, 5 val instances each) -- a single heavily-upweighted example can
    # dominate one gradient step there, unlike the encoder path's batched BCE.
    class_balanced_sft_clip: float = 8.0
    # task1 qlora_allam-only: richer, explicitly contrastive per-label prompt
    # descriptions (LABEL_DESCRIPTIONS_RICH) plus a fixed positive/hard-negative
    # few-shot demonstration pair per label (train_task1_generative.py's
    # select_fewshot_exemplars), targeting a diagnosed precision (not recall)
    # bottleneck concentrated on the semantically fuzzy TE/OT/AN labels. Ignored
    # everywhere else.
    fewshot_prompt: bool = False
    # task2-specific (ignored for task1 configs)
    stride: int = 64
    use_crf: bool = False
    gradient_checkpointing: bool = False
    # task2 Track B stage B only (SpanTypeClassifier) -- adds a supervised contrastive
    # term (models/contrastive.py's SupConLoss) alongside the classifier's existing
    # class-weighted cross-entropy. Ignored everywhere else.
    contrastive_enabled: bool = False
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.1
    # task2 CRF variants only (baseline/boosted_crf/enhanced_track_a) -- adds a
    # per-tag class-weighted auxiliary cross-entropy alongside the CRF's own
    # unweighted sequence log-likelihood (models/crf_tagger.py's WeightedCRFTrainer).
    # Ignored everywhere else.
    use_aux_ce: bool = False
    aux_ce_weight: float = 0.3
    aux_ce_clip: float = 20.0
    # task2 enhanced_track_a_retyped only (train_task2_retype.py) -- a span already
    # typed by the source CRF ensemble is only re-typed by the dedicated
    # SpanTypeClassifier when its own prediction's softmax confidence clears this
    # threshold. Ignored everywhere else.
    retype_confidence_threshold: float = 0.6


@dataclass
class TrainingConfig:
    epochs: int = 8
    per_device_batch_size: int = 16
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    # When true (the default), evaluate every epoch and keep only the checkpoint with
    # the best macro-F1 on the validation set (a per-epoch proxy metric -- token-level
    # macro F1 for Task 2's CRF/boundary taggers, span-type macro F1 for Stage B, since
    # the true partial-overlap span metric is expensive to compute every epoch. This
    # proxy is never used for reported numbers -- see evaluation/scoring.py). The best
    # checkpoint is saved to <run_output_dir>/checkpoint. When false, matches the prior
    # behavior: no evaluation during training, nothing saved to disk.
    save_best_checkpoint: bool = True
    # Only populated for task2's enhanced_track_b, which genuinely needs two distinct
    # hyperparameter sets (Stage A boundary tagger vs. Stage B span-type classifier).
    # Every other config leaves these as None and uses the flat fields above.
    stage_a: Optional["TrainingConfig"] = None
    stage_b: Optional["TrainingConfig"] = None


@dataclass
class DataConfig:
    discourse_cues: bool = False
    jitter_augment: int = 0


@dataclass
class EnsemblingConfig:
    enabled: bool = False
    weighting: str = "uniform"  # "uniform" | "internal_f1"
    # Task 2 span-ensemble vote threshold (postprocessing/spans.py's
    # ensemble_decode_spans): a character's majority-voted label survives only if
    # the combined weight of runs agreeing on it clears min_weight_fraction * the
    # total weight of all runs. Default 0.5 = strict majority, matching every
    # config's behavior before this field existed.
    min_weight_fraction: float = 0.5
    # Task 2 decode strategy: "char_vote" (default, ensemble_decode_spans's
    # per-character majority vote) or "cluster" (cluster_ensemble_decode_spans --
    # groups each run's candidate spans by overlap first, outputs the union once a
    # cluster's total weight clears min_weight_fraction; fixes char-voting's bias
    # against short spans, see postprocessing/spans.py's docstring). cross_label_iou
    # only applies to "cluster" -- same IoU-threshold NMS pattern as
    # models/span_scorer.py's decode_overlap_iou_threshold.
    decode_strategy: str = "char_vote"
    cross_label_iou: float = 0.3


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "sanad-argmining"
    entity: Optional[str] = None


@dataclass
class SpanScorerConfig:
    # Only populated for task2's span_scorer variant -- an enumerate-and-classify
    # alternative to BIO+CRF (see models/span_scorer.py). max_span_width=48 tokens
    # excludes ~2% of real gold spans (empirically verified against train_task_2.jsonl
    # via the real tokenizer: p95=39 tok, p99=57 tok, p100=126 tok -- 48 was chosen as
    # the coverage/candidate-count tradeoff point, not a guess).
    max_span_width: int = 48
    negative_sampling_ratio: float = 10.0
    hard_negative_fraction: float = 0.5
    # Higher than Task 1's 8x clip (models/losses.py) -- span-level CO/ST imbalance is
    # far more extreme (~35x vs AS) than Task 1's paragraph-level imbalance.
    class_weight_clip: float = 20.0
    decode_score_threshold: float = 0.5
    decode_overlap_iou_threshold: float = 0.3


@dataclass
class QuantizationConfig:
    # Only populated for task1's qlora_allam variant. fp16, not bfloat16: the T4 this
    # repo is sized for is Turing architecture with no native bf16 tensor cores --
    # bf16 is QLoRA's more common default on newer GPUs, but would be a real
    # hardware mismatch here.
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoraConfigFields:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class SubmissionConfig:
    # Off by default: packaging a CodaBench submission zip is an explicit opt-in step,
    # not something every training run should produce. When enabled, team_name is
    # required (validated in evaluation/submission.py, not here, since an empty
    # string is a valid YAML default and shouldn't fail config loading itself).
    #
    # training_setting is NOT the closed/open-track distinction (that's CLAUDE.md
    # section 2's separate concept, and this repo is closed-track-only regardless).
    # It's CodaBench's "which domain(s) did you train on" tag, read from the zip
    # filename -- every real leaderboard submission uses "both" (our train/val split
    # always spans both domains, see data/loading.py's build_shared_split), so that's
    # the correct default. Never use "closed" here -- that was a naming mix-up.
    enabled: bool = False
    team_name: str = ""
    training_setting: str = "both"
    # "dev" for every train_*.py entrypoint (predicts on dev_in.jsonl); predict_eval.py
    # overrides this to "eval" for Evaluation-phase (test_in.jsonl) runs. Determines
    # which submissions/{phase}/ subdirectory write_submission_zip writes into -- keeps
    # dev-phase (217-row) and eval-phase (213-row) zips from ever colliding on the same
    # filename, which caused real upload mix-ups before this existed.
    phase: str = "dev"


@dataclass
class Config:
    task: str  # "task1" | "task2"
    variant: str
    backbones: list[str]
    seeds: list[int]
    output_dir: str
    tapt: TaptConfig = field(default_factory=TaptConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ensembling: EnsemblingConfig = field(default_factory=EnsemblingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    span_scorer: Optional[SpanScorerConfig] = None
    quantization: Optional[QuantizationConfig] = None
    lora: Optional[LoraConfigFields] = None
    # train_task2_retype.py only: path to the config that owns the already-trained
    # boundary ensemble (e.g. configs/task2/enhanced_track_a_weighted.yaml) whose
    # checkpoints get reloaded rather than retrained. None/unused everywhere else.
    source_config: Optional[str] = None


def _build(cls: Type[T], data: Any, path: str) -> T:
    if not dataclasses.is_dataclass(cls):
        return data

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping, got {type(data).__name__}")

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    required = {
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }

    unknown = set(data.keys()) - set(field_types.keys())
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; valid keys are {sorted(field_types.keys())}")

    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{path}: missing required key(s) {sorted(missing)}")

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_cls = _resolve_dataclass_type(f.type, cls)
        if field_cls is not None and isinstance(value, dict):
            kwargs[f.name] = _build(field_cls, value, f"{path}.{f.name}")
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _resolve_dataclass_type(annotation: Any, owner_cls: Type[Any]) -> Optional[Type[Any]]:
    """Resolve a field's type annotation to a dataclass, if it names one.

    `from __future__ import annotations` makes every annotation a plain string (e.g.
    "TaptConfig" or 'Optional["TrainingConfig"]'), including forward references like
    TrainingConfig.stage_a/stage_b's self-reference. Pull out the innermost identifier
    and look it up in this module's namespace.
    """
    if not isinstance(annotation, str):
        return annotation if dataclasses.is_dataclass(annotation) else None

    name = annotation.replace('"', "").replace("'", "")
    for wrapper in ("Optional[", "list[", "List["):
        if name.startswith(wrapper):
            name = name[len(wrapper) : -1]
    candidate = globals().get(name)
    return candidate if isinstance(candidate, type) and dataclasses.is_dataclass(candidate) else None


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _build(Config, raw, "config")
