"""Weights & Biases wrapper: the single point of wandb integration in the package.

Nothing in models/, pretraining/, or postprocessing/ imports wandb directly. When
disabled, every method is a true no-op and `wandb` is never imported at all.
"""

from __future__ import annotations

from typing import Any

from utils.config import WandbConfig


class RunTracker:
    def __init__(self, config: WandbConfig, run_name: str, run_config: dict[str, Any]):
        self._enabled = config.enabled
        self._run = None
        if self._enabled:
            import wandb

            self._run = wandb.init(project=config.project, entity=config.entity, name=run_name, config=run_config, reinit=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self._enabled:
            return
        self._run.log(metrics, step=step)

    def finish(self) -> None:
        if not self._enabled:
            return
        self._run.finish()
        self._run = None

    def __enter__(self) -> "RunTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.finish()


def make_trainer_callback(tracker: RunTracker):
    """A lightweight HF TrainerCallback that forwards on_log/on_evaluate events into
    tracker.log(...), so training scripts don't need to poll Trainer state themselves."""
    from transformers import TrainerCallback

    class _ForwardingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                tracker.log(logs, step=state.global_step)

    return _ForwardingCallback()
