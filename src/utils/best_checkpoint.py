"""Best-checkpoint selection for QLoRA generative training paths (train_task1_generative.py,
src/models/span_relabeler.py), where HF Trainer's native eval_strategy/
load_best_model_at_end/compute_metrics machinery doesn't apply -- those variants'
accuracy needs multiple per-example forward passes (rank-classification over several
yes/no prompts per row/span), not a single batched forward pass per eval step. The
encoder variants (train_task1.py, train_task2.py) already get real best-checkpoint
selection via HF's native mechanism and don't use this module.
"""

from __future__ import annotations


class BestStateTracker:
    def __init__(self):
        self.best_metric = float("-inf")
        self.best_state = None

    def update(self, metric: float, state_dict: dict) -> bool:
        """Returns True if this is a new best. Stores a detached CPU clone, so later
        mutation of the live state_dict never corrupts the stored best."""
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
            return True
        return False


def make_best_checkpoint_callback(eval_fn, tracker_state: BestStateTracker, run_tracker=None, metric_name: str = "eval_metric"):
    """eval_fn(model) -> float. Returns a TrainerCallback that evaluates at
    on_epoch_end (model.eval() -> eval_fn -> model.train()), logs to run_tracker if
    given, and records the best adapter-only state (via peft.get_peft_model_state_dict)
    into tracker_state."""
    from peft import get_peft_model_state_dict
    from transformers import TrainerCallback

    class _BestCheckpointCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            model.eval()
            metric = eval_fn(model)
            if run_tracker is not None:
                run_tracker.log({metric_name: metric}, step=state.global_step)
            tracker_state.update(metric, get_peft_model_state_dict(model))
            model.train()

    return _BestCheckpointCallback()
