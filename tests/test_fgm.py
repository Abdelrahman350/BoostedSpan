"""FGM regression tests: attack perturbs only the embedding weights, restore returns
them exactly, and the attack direction is gradient-scale invariant. Uses a tiny
synthetic model -- no GPU, no downloads."""

import torch

from utils.fgm import FGM, install_fgm


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(10, 4)
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, input_ids):
        return self.linear(self.word_embeddings(input_ids).mean(dim=1))


def _loss_of(model):
    logits = model(torch.tensor([[1, 2, 3]]))
    return logits.sum()


def test_attack_perturbs_embeddings_and_restore_is_exact():
    torch.manual_seed(0)
    model = _TinyModel()
    _loss_of(model).backward()

    original = model.word_embeddings.weight.data.clone()
    fgm = FGM(model, epsilon=1.0)

    fgm.attack()
    assert not torch.equal(model.word_embeddings.weight.data, original)

    fgm.restore()
    assert torch.equal(model.word_embeddings.weight.data, original)
    assert fgm.backup == {}


def test_attack_leaves_non_embedding_params_untouched():
    torch.manual_seed(0)
    model = _TinyModel()
    _loss_of(model).backward()

    linear_before = model.linear.weight.data.clone()
    fgm = FGM(model, epsilon=1.0)
    fgm.attack()
    assert torch.equal(model.linear.weight.data, linear_before)
    fgm.restore()


def test_attack_direction_is_gradient_scale_invariant():
    # g/||g|| is unchanged when grads are uniformly scaled (e.g. by a fp16 GradScaler),
    # so the perturbation must be identical for g and 1000*g.
    torch.manual_seed(0)
    model_a, model_b = _TinyModel(), _TinyModel()
    model_b.load_state_dict(model_a.state_dict())

    _loss_of(model_a).backward()
    (_loss_of(model_b) * 1000).backward()

    FGM(model_a, epsilon=0.5).attack()
    FGM(model_b, epsilon=0.5).attack()
    assert torch.allclose(model_a.word_embeddings.weight.data, model_b.word_embeddings.weight.data, atol=1e-5)


def test_install_fgm_zero_epsilon_is_noop():
    class _FakeTrainer:
        def __init__(self):
            self.model = _TinyModel()
            self.training_step = "sentinel"

    trainer = _FakeTrainer()
    install_fgm(trainer, epsilon=0.0)
    assert trainer.training_step == "sentinel"  # untouched when disabled
