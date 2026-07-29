from pathlib import Path

from utils.checkpointing import save_best_and_last_checkpoints


class _FakeState:
    def __init__(self, best_model_checkpoint):
        self.best_model_checkpoint = best_model_checkpoint


class _FakeTrainer:
    """Mimics just the Trainer API save_best_and_last_checkpoints calls: .state and
    .save_model(dir). Real numbered checkpoint-<step> dirs are pre-populated on disk
    by the test, matching what Trainer's own save_strategy="epoch" checkpointing
    would have written during training."""

    def __init__(self, best_model_checkpoint):
        self.state = _FakeState(best_model_checkpoint)

    def save_model(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.safetensors").write_text("best-weights")


class _FakeTokenizer:
    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tokenizer_config.json").write_text("{}")


def _make_numbered_checkpoint(output_dir: Path, step: int, content: str) -> Path:
    d = output_dir / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_text(content)
    return d


def test_best_and_last_differ(tmp_path):
    best_dir = _make_numbered_checkpoint(tmp_path, 100, "epoch-2-weights")
    last_dir = _make_numbered_checkpoint(tmp_path, 150, "epoch-3-weights")
    trainer = _FakeTrainer(best_model_checkpoint=str(best_dir))

    best_path, last_path = save_best_and_last_checkpoints(trainer, _FakeTokenizer(), str(tmp_path))

    assert Path(best_path) == tmp_path / "checkpoint_best"
    assert Path(last_path) == tmp_path / "checkpoint_last"
    assert (tmp_path / "checkpoint_best" / "model.safetensors").read_text() == "best-weights"  # from trainer.save_model
    assert (tmp_path / "checkpoint_last" / "model.safetensors").read_text() == "epoch-3-weights"  # moved from checkpoint-150
    assert (tmp_path / "checkpoint_last" / "tokenizer_config.json").exists()
    # numbered dirs cleaned up, only the two stable dirs remain
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"checkpoint_best", "checkpoint_last"}


def test_best_equals_last(tmp_path):
    only_dir = _make_numbered_checkpoint(tmp_path, 200, "final-epoch-weights")
    trainer = _FakeTrainer(best_model_checkpoint=str(only_dir))

    best_path, last_path = save_best_and_last_checkpoints(trainer, _FakeTokenizer(), str(tmp_path))

    assert (Path(best_path) / "model.safetensors").exists()
    assert (Path(last_path) / "model.safetensors").exists()
    assert (Path(best_path) / "model.safetensors").read_text() == (Path(last_path) / "model.safetensors").read_text()
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"checkpoint_best", "checkpoint_last"}


def test_no_numbered_checkpoints_falls_back_to_copy(tmp_path):
    trainer = _FakeTrainer(best_model_checkpoint=None)

    best_path, last_path = save_best_and_last_checkpoints(trainer, _FakeTokenizer(), str(tmp_path))

    assert (Path(best_path) / "model.safetensors").exists()
    assert (Path(last_path) / "model.safetensors").exists()
