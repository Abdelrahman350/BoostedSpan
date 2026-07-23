"""Round HF Trainer's console/W&B log output to a fixed number of significant digits.

Trainer.log() already rounds `loss` (4 decimals) and `epoch` (2 decimals) internally,
but leaves `grad_norm` and `learning_rate` at full float precision (e.g.
16.690906524658203, 1.5454545454545454e-05), which is what actually shows up in the
printed `{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}` lines.

install_rounded_logging inserts a callback at the FRONT of the trainer's callback
list (not the back, where callbacks passed via the `callbacks=` constructor arg
normally land) so it mutates the shared `logs` dict in place before the default
ProgressCallback prints it -- callbacks in a CallbackHandler all receive the same
dict object for a given on_log event, so an earlier callback's in-place edits are
visible to every later one.
"""

from __future__ import annotations

import math

from transformers import Trainer, TrainerCallback


def round_sig(x: float, sig_digits: int = 5) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, sig_digits - 1 - int(math.floor(math.log10(abs(x)))))


class RoundedLoggingCallback(TrainerCallback):
    def __init__(self, sig_digits: int = 5):
        self.sig_digits = sig_digits

    def on_log(self, args, state, control, logs: dict | None = None, **kwargs):
        if not logs:
            return
        for key, value in logs.items():
            if isinstance(value, float):
                logs[key] = round_sig(value, self.sig_digits)


def install_rounded_logging(trainer: Trainer, sig_digits: int = 5) -> None:
    trainer.callback_handler.callbacks.insert(0, RoundedLoggingCallback(sig_digits))
