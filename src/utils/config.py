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
    # task2-specific (ignored for task1 configs)
    stride: int = 64
    use_crf: bool = False
    gradient_checkpointing: bool = False


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


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "sanad-argmining"
    entity: Optional[str] = None


@dataclass
class SubmissionConfig:
    # Off by default: packaging a CodaBench submission zip is an explicit opt-in step,
    # not something every training run should produce. When enabled, team_name is
    # required (validated in evaluation/submission.py, not here, since an empty
    # string is a valid YAML default and shouldn't fail config loading itself).
    enabled: bool = False
    team_name: str = ""
    training_setting: str = "closed"


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
