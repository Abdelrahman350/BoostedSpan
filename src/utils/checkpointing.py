"""Shared best/last checkpoint finalization for HF-Trainer-based training runs.

Used with TrainingArguments(save_strategy="epoch", load_best_model_at_end=True,
save_total_limit=2) -- Trainer already writes numbered checkpoint-<step> dirs during
training and never evicts the best one under the rolling save_total_limit, so after
trainer.train() returns, trainer.state.best_model_checkpoint plus the highest-step
numbered dir are enough to reconstruct "best" and "last" without any custom
mid-training bookkeeping. Works uniformly for real PreTrainedModels and for the
bare-nn.Module custom models (TokenClassifierWithCRF, SpanScorerModel,
SpanTypeClassifier), since Trainer's internal per-epoch checkpointing already
handles both.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

_STEP_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def load_custom_state_dict(model: torch.nn.Module, checkpoint_dir: str) -> torch.nn.Module:
    """Reload a bare-nn.Module checkpoint (TokenClassifierWithCRF, SpanScorerModel,
    SpanTypeClassifier) saved by save_best_and_last_checkpoints -- these aren't HF
    PreTrainedModels, so trainer.save_model() falls back to a raw state_dict with
    no config.json/save_pretrained support. Caller must reconstruct the same model
    class/shape (e.g. TokenClassifierWithCRF(base_checkpoint, num_labels=...))
    before calling this."""
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    state_dict = load_file(st_path) if os.path.exists(st_path) else torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def _numbered_checkpoint_dirs(output_dir: Path) -> list[Path]:
    dirs = []
    for p in output_dir.iterdir():
        if p.is_dir() and _STEP_DIR_RE.match(p.name):
            dirs.append(p)
    return sorted(dirs, key=lambda p: int(_STEP_DIR_RE.match(p.name).group(1)))


def save_best_and_last_checkpoints(trainer, tokenizer, output_dir: str) -> tuple[str, str]:
    """Call once, right after trainer.train() (with load_best_model_at_end=True).

    trainer.model already holds the best epoch's weights at this point -- saved to
    checkpoint_best. The last-on-disk numbered checkpoint dir (highest step) is
    moved to checkpoint_last; if it's the same epoch as best (best == final epoch),
    checkpoint_best is copied instead. Every leftover numbered checkpoint-<step> dir
    is removed afterward so only these two directories remain.
    """
    out = Path(output_dir)
    best_dir = out / "checkpoint_best"
    last_dir = out / "checkpoint_last"

    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    numbered = _numbered_checkpoint_dirs(out)
    best_ckpt = Path(trainer.state.best_model_checkpoint).resolve() if trainer.state.best_model_checkpoint else None
    last_ckpt = numbered[-1] if numbered else None

    if last_ckpt is None or (best_ckpt is not None and last_ckpt.resolve() == best_ckpt):
        shutil.copytree(best_dir, last_dir)
    else:
        shutil.move(str(last_ckpt), str(last_dir))
        tokenizer.save_pretrained(str(last_dir))

    for d in numbered:
        if d.exists():
            shutil.rmtree(d)

    return str(best_dir), str(last_dir)
