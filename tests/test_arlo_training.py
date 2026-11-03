"""Tests for the Arlo decision training signal (choice_loss, binary_loss, Arlo.training_loss).
Run with: python -m pytest tests/test_arlo_training.py -q -p no:cacheprovider"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.arlo import Arlo, binary_loss, choice_loss, train_arlo  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510
B, N, S = 4, 3, 6


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512,
        scratch_open_id=OPEN,
        scratch_sep_id=SEP,
        scratch_close_id=CLOSE,
        hidden_size=64,
        num_attention_heads=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        q_lora_rank=32,
        kv_lora_rank=24,
        intermediate_size=96,
        num_routed_experts=4,
        num_active_routed=2,
        ssm_state_dim=16,
        max_recurrent_loops=4,
        min_recurrent_loops=2,
        loop_dropout_prob=0.0,
        max_position_embeddings=256,
        arlo_dim=16,
        arlo_pointer_rank=8,
        arlo_slots=4,
        arlo_heads=4,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def tiny_arlo(seed: int = 0) -> Arlo:
    torch.manual_seed(seed)
    return Arlo(tiny_config())


def random_batch(seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    query = torch.randn(B, 64, generator=gen)
    options = torch.randn(B, N, 64, generator=gen)
    states = torch.randn(B, S, 64, generator=gen)
    choice_gold = torch.randint(0, N, (B,), generator=gen)
    binary_gold = torch.randint(0, 2, (B,), generator=gen).float()
    return query, options, states, choice_gold, binary_gold


def test_choice_loss_matches_cross_entropy():
    torch.manual_seed(2)
    scores = torch.randn(B, N)
    gold = torch.randint(0, N, (B,))
    loss = choice_loss(scores, gold)
    assert loss.ndim == 0
    assert torch.allclose(loss, F.cross_entropy(scores, gold), atol=1e-6)


def test_binary_loss_matches_bce_with_logits():
    torch.manual_seed(3)
    logit = torch.randn(B)
    gold = torch.randint(0, 2, (B,)).float()
    loss = binary_loss(logit, gold)
    assert loss.ndim == 0
    assert torch.allclose(loss, F.binary_cross_entropy_with_logits(logit, gold), atol=1e-6)


def test_training_loss_backward_reaches_slots_and_pointer_heads():
    arlo = tiny_arlo()
    arlo.zero_grad()
    query, options, states, choice_gold, _ = random_batch()
    loss = arlo.training_loss(query, options, states, choice_gold, task="choice")
    assert loss.ndim == 0
    loss.backward()
    assert arlo.slot_base.grad is not None
    assert float(arlo.slot_base.grad.abs().sum()) > 0.0
    for name in ("q_projs", "o_projs"):
        head = getattr(arlo.pointer, name)[0]
        assert head.weight.grad is not None
        assert float(head.weight.grad.abs().sum()) > 0.0


def test_binary_training_loss_backward_reaches_binary_head():
    arlo = tiny_arlo()
    arlo.zero_grad()
    query, options, states, _, binary_gold = random_batch()
    loss = arlo.training_loss(query, options, states, binary_gold, task="binary")
    assert loss.ndim == 0
    loss.backward()
    assert arlo.binary_head.weight.grad is not None
    assert float(arlo.binary_head.weight.grad.abs().sum()) > 0.0


def test_unknown_task_raises_value_error():
    arlo = tiny_arlo()
    query, options, states, choice_gold, _ = random_batch()
    with pytest.raises(ValueError):
        arlo.training_loss(query, options, states, choice_gold, task="ranking")


def test_adam_steps_reduce_choice_loss_on_a_fixed_batch():
    arlo = tiny_arlo()
    query, options, states, choice_gold, _ = random_batch()
    optimiser = torch.optim.Adam(arlo.parameters(), lr=1e-2)
    initial = None
    for step in range(20):
        optimiser.zero_grad()
        loss = arlo.training_loss(query, options, states, choice_gold, task="choice")
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimiser.step()
    final = float(arlo.training_loss(query, options, states, choice_gold, task="choice").detach())
    assert final < initial


def decision_model() -> InkyModel:
    torch.manual_seed(0)
    cfg = tiny_config(vocab_size=512, scratch_open_id=OPEN, scratch_sep_id=SEP,
                      scratch_close_id=CLOSE, hidden_size=64, num_routed_experts=4,
                      num_active_routed=2, ssm_state_dim=16, max_recurrent_loops=4,
                      min_recurrent_loops=2, loop_dropout_prob=0.0, max_position_embeddings=256,
                      enable_arlo=True, pad_token_id=1)
    return InkyModel(cfg)


def decision_examples(seed: int = 7, n: int = 6):
    gen = torch.Generator().manual_seed(seed)
    examples = []
    for _ in range(n):
        context = torch.randint(2, 500, (1, 10), generator=gen)
        options = [torch.randint(2, 500, (1, L), generator=gen) for L in (2, 3, 4)]
        examples.append({"context": context, "options": options, "gold": int(torch.randint(0, 3, (1,), generator=gen))})
    return examples


def test_train_arlo_reduces_loss_and_leaves_the_language_model_unchanged():
    model = decision_model()
    model.train()
    examples = decision_examples()
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("arlo.")}
    arlo_before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("arlo.")}
    losses = train_arlo(model, [examples], task="choice", lr=1e-2, epochs=15)
    assert len(losses) == 15
    assert losses[-1] < losses[0], losses
    # Non-Arlo parameters are bit-for-bit unchanged, and Arlo parameters did move.
    for n, p in model.named_parameters():
        if n.startswith("arlo."):
            continue
        assert torch.equal(p.detach(), before[n]), n
    assert any(not torch.equal(p.detach(), arlo_before[n])
               for n, p in model.named_parameters() if n.startswith("arlo."))
    # Mode and requires_grad are restored.
    assert model.training is True
    assert all(p.requires_grad for p in model.parameters())


def test_train_arlo_binary_task_reduces_loss():
    model = decision_model()
    gen = torch.Generator().manual_seed(3)
    examples = [{"context": torch.randint(2, 500, (1, 8), generator=gen),
                 "gold": i % 2} for i in range(6)]
    losses = train_arlo(model, [examples], task="binary", lr=1e-2, epochs=15)
    assert losses[-1] < losses[0], losses


def test_train_arlo_restores_mode_and_flags_on_error():
    model = decision_model().eval()
    bad = [[{"context": torch.randint(2, 500, (1, 6)), "options": [torch.randint(2, 500, (1, 2))], "gold": 4}]]
    with pytest.raises(ValueError):
        train_arlo(model, bad, task="choice")
    assert model.training is False
    assert all(p.requires_grad for p in model.parameters())


def test_train_arlo_needs_an_enabled_engine():
    model = decision_model()
    model.arlo = None
    with pytest.raises(RuntimeError):
        train_arlo(model, [decision_examples(n=1)], task="choice")


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
