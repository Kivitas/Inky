"""Tests for the writing decision engine (Rhen Choice and Noul). Run with: python -m pytest tests -q"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel, split_parameters  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


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
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def seeded_model(config: InkyConfig) -> InkyModel:
    torch.manual_seed(0)
    model = InkyModel(config).eval()
    return model


def options(*lengths):
    torch.manual_seed(5)
    return [torch.randint(2, 500, (1, n)) for n in lengths]


def test_choice_returns_a_distribution_over_options():
    model = seeded_model(tiny_config())
    context = torch.randint(2, 500, (1, 10))
    probs = model.decide_choice(context, options(2, 3, 5))
    assert probs.shape == (3,)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert bool((probs > 0).all())


def test_choice_is_equivariant_to_option_order():
    model = seeded_model(tiny_config())
    context = torch.randint(2, 500, (1, 10))
    opts = options(2, 3, 5)
    forward = model.decide_choice(context, opts)
    backward = model.decide_choice(context, list(reversed(opts)))
    assert torch.allclose(forward, backward.flip(0), atol=1e-5)


def test_choice_depends_on_the_context():
    model = seeded_model(tiny_config())
    opts = options(3, 3)
    a = model.decide_choice(torch.randint(2, 500, (1, 12)), opts)
    b = model.decide_choice(torch.randint(2, 500, (1, 12)), opts)
    assert not torch.allclose(a, b)


def test_choice_accepts_a_question_between_context_and_options():
    model = seeded_model(tiny_config())
    context = torch.randint(2, 500, (1, 10))
    question = torch.randint(2, 500, (1, 4))
    probs = model.decide_choice(context, options(2, 2), question_ids=question)
    assert probs.shape == (2,)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


def test_binary_returns_a_probability():
    model = seeded_model(tiny_config())
    p = model.decide_binary(torch.randint(2, 500, (1, 9)), question_ids=torch.randint(2, 500, (1, 3)))
    assert p.ndim == 0
    assert 0.0 < float(p) < 1.0


def test_decisions_do_not_change_the_language_model_path():
    model = seeded_model(tiny_config())
    ids = torch.randint(2, 500, (1, 12))
    before = model(ids, num_loops=3).logits
    model.decide_choice(ids, options(2, 2))
    model.decide_binary(ids)
    after = model(ids, num_loops=3).logits
    assert torch.allclose(before, after, atol=1e-6)


def test_disabled_engine_is_absent_and_refuses_decisions():
    model = seeded_model(tiny_config(enable_arlo=False))
    assert model.arlo is None
    with pytest.raises(RuntimeError):
        model.decide_binary(torch.randint(2, 500, (1, 4)))


def test_slot_bank_is_routed_to_adamw():
    model = seeded_model(tiny_config())
    muon, _, _ = split_parameters(model)
    slot_ids = {id(p) for n, p in model.named_parameters() if n.endswith("slot_base")}
    assert slot_ids and not (slot_ids & {id(p) for p in muon})


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
