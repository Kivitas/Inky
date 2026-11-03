"""Regression tests for the Arlo audit findings ARLO-03, ARLO-04 and ARLO-05.
Run with: python -B -m pytest tests/test_arlo_fixes.py -q -p no:cacheprovider"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.arlo import Arlo  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510
D = 64


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512,
        scratch_open_id=OPEN,
        scratch_sep_id=SEP,
        scratch_close_id=CLOSE,
        hidden_size=D,
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
        enable_arlo=True,
        pad_token_id=1,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def tiny_arlo(slots: int = 4, seed: int = 0) -> Arlo:
    torch.manual_seed(seed)
    return Arlo(tiny_config(arlo_slots=slots))


def random_inputs(seed: int = 1, B: int = 2, N: int = 4, S: int = 6):
    gen = torch.Generator().manual_seed(seed)
    query = torch.randn(B, D, generator=gen)
    options = torch.randn(B, N, D, generator=gen)
    states = torch.randn(B, S, D, generator=gen)
    return query, options, states


# ARLO-04: slots are not averaged.

def test_single_slot_summary_matches_plain_projection():
    # With one slot the attention weight is 1, so the result equals the old mean-of-slots formula.
    arlo = tiny_arlo(slots=1)
    query, _, states = random_inputs()
    with torch.no_grad():
        expected = query + arlo.out_norm(arlo.out_proj(arlo.read(states).mean(dim=1)))
        assert torch.allclose(arlo.summarise(query, states), expected, atol=1e-6)


def test_multi_slot_summary_is_not_the_slot_mean():
    arlo = tiny_arlo(slots=4)
    query, _, states = random_inputs()
    with torch.no_grad():
        mean_version = query + arlo.out_norm(arlo.out_proj(arlo.read(states).mean(dim=1)))
        assert not torch.allclose(arlo.summarise(query, states), mean_version, atol=1e-4)


def test_slot_weighting_depends_on_the_query():
    # The slots are weighted by their match with the query, so different queries pick different mixes.
    arlo = tiny_arlo(slots=4)
    _, _, states = random_inputs()
    with torch.no_grad():
        slots = arlo.out_proj(arlo.read(states))                       # [B, K, d]
        steers = []
        for seed in (11, 12, 13):
            q = torch.randn(1, D, generator=torch.Generator().manual_seed(seed))
            w = torch.softmax(torch.einsum("bd,bkd->bk", q, slots[:1]) / D ** 0.5, dim=-1)
            steers.append(w)
        assert not all(torch.allclose(steers[0], s, atol=1e-4) for s in steers[1:])


def test_summary_gradients_reach_every_slot():
    arlo = tiny_arlo(slots=4)
    query, _, states = random_inputs()
    arlo.zero_grad()
    arlo.summarise(query, states).pow(2).sum().backward()
    assert arlo.slot_base.grad is not None
    assert float(arlo.slot_base.grad.abs().sum()) > 0.0


# ARLO-03: the option set influences the scores, and the scores stay order-invariant.

def test_option_scores_depend_on_the_other_options():
    arlo = tiny_arlo()
    query, options, states = random_inputs(B=1, N=2)
    gen = torch.Generator().manual_seed(21)
    other_a = torch.randn(1, 1, D, generator=gen)
    other_b = torch.randn(1, 1, D, generator=gen)
    with torch.no_grad():
        with_a = arlo.choice_scores(query, torch.cat([options[:, :1], other_a], dim=1), states)
        with_b = arlo.choice_scores(query, torch.cat([options[:, :1], other_b], dim=1), states)
    assert not torch.allclose(with_a[0, 0], with_b[0, 0], atol=1e-5)


def test_option_scores_are_permutation_equivariant():
    arlo = tiny_arlo()
    query, options, states = random_inputs(B=2, N=4)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        forward = arlo.choice_scores(query, options, states)
        permuted = arlo.choice_scores(query, options[:, perm], states)
    assert torch.allclose(forward[:, perm], permuted, atol=1e-5)


def test_choice_scores_gradients_reach_the_option_attention():
    arlo = tiny_arlo()
    query, options, states = random_inputs()
    options = options.clone().requires_grad_(True)
    arlo.zero_grad()
    arlo.choice_scores(query, options, states).sum().backward()
    assert options.grad is not None
    assert float(options.grad.abs().sum()) > 0.0


# ARLO-05: decision calls must restore the training mode.

def test_decide_binary_restores_training_mode():
    torch.manual_seed(0)
    model = InkyModel(tiny_config())
    model.train()
    model.decide_binary(torch.randint(2, 500, (1, 9)))
    assert model.training is True


def test_decide_choice_restores_training_mode():
    torch.manual_seed(0)
    model = InkyModel(tiny_config())
    model.train()
    model.decide_choice(torch.randint(2, 500, (1, 9)), [torch.randint(2, 500, (1, 2))])
    assert model.training is True


def test_decide_calls_restore_eval_mode_too_and_on_error():
    torch.manual_seed(0)
    model = InkyModel(tiny_config())
    model.eval()
    model.decide_choice(torch.randint(2, 500, (1, 9)), [torch.randint(2, 500, (1, 2))])
    assert model.training is False
    model.train()
    with pytest.raises(ValueError):
        model.decide_choice(torch.randint(2, 500, (1, 9)), [torch.zeros((1, 0), dtype=torch.long)])
    assert model.training is True


# ARLO-03 (remainder): options are mean-pooled over their own tokens, and empty options are rejected.

def test_option_representation_is_the_mean_of_its_own_tokens():
    from inky.arlo import mean_pool_options

    torch.manual_seed(3)
    h = torch.randn(2, 7, D)
    lengths = [6, 4]                       # prefix of 3 tokens; row 1 has 1 option token and right padding
    pooled = mean_pool_options(h, 3, lengths)
    assert torch.allclose(pooled[0], h[0, 3:6].mean(dim=0), atol=1e-6)
    assert torch.allclose(pooled[1], h[1, 3:4].mean(dim=0), atol=1e-6)


def test_empty_option_is_rejected():
    torch.manual_seed(0)
    model = InkyModel(tiny_config())
    with pytest.raises(ValueError, match="option 1 is empty"):
        model.decide_choice(torch.randint(2, 500, (1, 5)),
                            [torch.randint(2, 500, (1, 2)), torch.zeros((1, 0), dtype=torch.long)])


def test_option_order_does_not_change_the_probabilities_of_each_option():
    torch.manual_seed(0)
    model = InkyModel(tiny_config())
    model.eval()
    ctx = torch.randint(2, 500, (1, 9))
    opts = [torch.randint(2, 500, (1, n)) for n in (2, 3, 4)]
    forward = model.decide_choice(ctx, opts)
    backward = model.decide_choice(ctx, list(reversed(opts)))
    assert torch.allclose(forward, backward.flip(0), atol=1e-5)


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
