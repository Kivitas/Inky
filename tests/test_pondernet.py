"""Tests for PonderNet halting. Run with: python -m pytest tests -q"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.layers import PonderHalting, geometric_halting_weights, geometric_prior, pondernet_kl  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512, scratch_open_id=OPEN, scratch_sep_id=SEP, scratch_close_id=CLOSE,
        hidden_size=64, num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16,
        q_lora_rank=32, kv_lora_rank=24, intermediate_size=96, num_routed_experts=4, num_active_routed=2,
        ssm_state_dim=16, max_recurrent_loops=4, min_recurrent_loops=2, loop_dropout_prob=0.0,
        max_position_embeddings=256, arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4,
        eos_token_id=1, pad_token_id=1,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def test_weights_sum_to_one_and_match_the_geometric_form():
    lam = torch.full((1, 1), 0.5)
    w = geometric_halting_weights([lam, lam, lam])          # L = 3
    assert w.shape == (3, 1, 1)
    assert torch.allclose(w.sum(dim=0), torch.ones(1, 1))
    assert torch.allclose(w.flatten(), torch.tensor([0.5, 0.25, 0.25]))


def test_weights_sum_to_one_for_random_lambdas():
    torch.manual_seed(0)
    lambdas = [torch.rand(2, 5) for _ in range(6)]
    w = geometric_halting_weights(lambdas)
    assert torch.allclose(w.sum(dim=0), torch.ones(2, 5), atol=1e-6)
    assert bool((w >= 0).all())


def test_single_loop_takes_all_the_mass():
    w = geometric_halting_weights([torch.rand(1, 3)])
    assert torch.allclose(w, torch.ones(1, 1, 3))


def test_prior_is_a_distribution_with_the_requested_first_step():
    prior = geometric_prior(3, 1.0 / 3.0)
    assert torch.allclose(prior.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(prior, torch.tensor([1 / 3, 2 / 9, 4 / 9]), atol=1e-6)


def test_kl_is_zero_at_the_prior_and_positive_elsewhere():
    prior = geometric_prior(4, 0.4)
    at_prior = prior.view(-1, 1, 1).expand(4, 2, 3).contiguous()
    assert abs(float(pondernet_kl(at_prior, prior))) < 1e-6
    off = torch.zeros(4, 2, 3)
    off[0] = 1.0                                             # all mass on loop 1
    assert float(pondernet_kl(off, prior)) > 0.0


def test_halting_gate_starts_at_the_prior():
    model = InkyModel(tiny_config()).eval()
    expected = math.log((1 / 3) / (1 - 1 / 3))
    assert abs(float(model.halting.gate.bias.detach()) - expected) < 1e-6


def test_prior_steps_below_one_is_rejected():
    with pytest.raises(ValueError):
        PonderHalting(8, prior_steps=0.5)


def test_mean_loops_is_within_bounds_and_loss_reaches_the_halting_gate():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = InkyModel(cfg).train()
    ids = torch.randint(2, 500, (2, 10))
    out = model(ids, labels=ids, num_loops=cfg.max_recurrent_loops)
    assert 1.0 <= float(out.mean_loops) <= cfg.max_recurrent_loops
    out.loss.backward()
    grad = model.halting.gate.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_training_loss_decreases_on_a_fixed_batch():
    torch.manual_seed(0)
    model = InkyModel(tiny_config()).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ids = torch.randint(2, 500, (4, 16))
    first = None
    for _ in range(20):
        opt.zero_grad()
        loss = model(ids, labels=ids, num_loops=2).loss
        if first is None:
            first = float(loss.detach())
        loss.backward()
        opt.step()
    last = float(model(ids, labels=ids, num_loops=2).loss.detach())
    assert last < first - 0.1, (first, last)


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
