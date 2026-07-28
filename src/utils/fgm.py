"""FGM (Fast Gradient Method) adversarial training for encoder fine-tuning.

After the normal backward pass, perturb the word-embedding weights along the gradient
direction (r = epsilon * g/||g||), run a second forward+backward so the adversarial
gradients accumulate onto the same .grad buffers, then restore the embeddings before
the optimizer steps. ~2x step time, no extra memory -- the best gain-per-line
regularizer in the low-resource NER/span shared-task literature (+0.3-1.0 F1 typical).

install_fgm() wraps an existing HF Trainer instance the same way
utils/logging.py's install_rounded_logging does -- so every Trainer subclass in this
repo (Trainer, Task1Trainer, ClassWeightedSpanTrainer, SpanScorerTrainer) gets FGM
without another subclass layer, and the wiring stays a one-line checklist item per
Trainer site.
"""

from __future__ import annotations

import torch


class FGM:
    def __init__(self, model: torch.nn.Module, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def install_fgm(trainer, epsilon: float) -> None:
    """Monkey-patch trainer.training_step to add the FGM attack->re-backward->restore
    cycle. A wrapper (not a subclass) so it composes with every existing Trainer
    subclass in this repo without touching their class definitions."""
    if epsilon <= 0:
        return

    fgm = FGM(trainer.model, epsilon=epsilon)
    original_training_step = trainer.training_step

    def training_step_with_fgm(model, inputs, *args, **kwargs):
        loss = original_training_step(model, inputs, *args, **kwargs)
        # original_training_step already ran backward, so grads are populated. The
        # attack direction g/||g|| is invariant to fp16 GradScaler scaling.
        fgm.attack()
        # Shallow-copy: every custom compute_loss in this repo pops keys (e.g.
        # "labels") from the dict it's given, and we must not mutate the caller's.
        adv_inputs = trainer._prepare_inputs({k: v for k, v in inputs.items()})
        model.train()
        adv_loss = trainer.compute_loss(model, adv_inputs)
        if trainer.args.gradient_accumulation_steps > 1:
            # Keeps the adversarial gradient's scale at-or-below the main loss's
            # regardless of which side of transformers' grad-accum scaling the main
            # pass took; any constant factor here is absorbed by epsilon tuning.
            adv_loss = adv_loss / trainer.args.gradient_accumulation_steps
        trainer.accelerator.backward(adv_loss)
        fgm.restore()
        return loss

    trainer.training_step = training_step_with_fgm
