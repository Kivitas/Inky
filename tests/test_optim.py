"""Tests for the Muon + AdamW hybrid optimiser. Run with: python -m pytest tests -q"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import HybridOptimizer, InkyConfig, InkyModel, Muon, split_parameters  # noqa: E402
from inky.optim import muon_scale, muon_update_scale, zeropower_via_newtonschulz5  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


def tiny_model() -> InkyModel:
    torch.manual_seed(0)
    config = InkyConfig(
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
    )
    return InkyModel(config)


def test_split_sends_matrices_to_muon_and_the_rest_to_adamw():
    model = tiny_model()
    muon, decay, no_decay = split_parameters(model)
    muon_ids = {id(p) for p in muon}

    assert all(p.ndim == 2 for p in muon)
    # Token embedding (tied with the LM head) and the FiLM step table stay with AdamW.
    assert id(model.embed_tokens.weight) not in muon_ids
    step_table = [p for n, p in model.named_parameters() if "step_embedding" in n]
    assert step_table and all(id(p) not in muon_ids for p in step_table)
    # Every trainable parameter lands in exactly one group.
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grouped = sum(p.numel() for p in muon + decay + no_decay)
    assert grouped == total
    assert len(muon_ids) == len(muon)


def test_newton_schulz_pushes_singular_values_towards_one():
    torch.manual_seed(1)
    G = torch.randn(16, 8) * torch.logspace(0, 2, 8)  # badly conditioned on purpose
    U = zeropower_via_newtonschulz5(G, steps=5)
    svals = torch.linalg.svdvals(U)
    assert U.shape == G.shape
    assert float(svals.min()) > 0.5 and float(svals.max()) < 1.5, svals


def test_muon_step_changes_a_matrix():
    torch.manual_seed(2)
    W = torch.nn.Parameter(torch.randn(8, 4))
    opt = Muon([W], lr=0.02)
    before = W.detach().clone()
    W.grad = torch.randn_like(W)
    opt.step()
    assert not torch.allclose(before, W.detach())


def test_hybrid_steps_reduce_loss_on_a_fixed_batch():
    model = tiny_model()
    model.train()
    opt = HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02, adamw_weight_decay=0.0)
    ids = torch.randint(2, 512, (4, 16))

    first = None
    for _ in range(25):
        opt.zero_grad()
        loss = model(ids, labels=ids, num_loops=2).loss
        if first is None:
            first = float(loss.detach())
        loss.backward()
        opt.step(max_grad_norm=1.0)
        model.update_moe_balance()
    last = float(model(ids, labels=ids, num_loops=2).loss)
    assert last < first - 0.2, (first, last)


def test_muon_groups_have_decoupled_weight_decay_by_default():
    opt = HybridOptimizer(tiny_model())
    assert opt.muon.param_groups and all(g["weight_decay"] == 0.1 for g in opt.muon.param_groups)
    overridden = HybridOptimizer(tiny_model(), muon_weight_decay=0.0)
    assert all(g["weight_decay"] == 0.0 for g in overridden.muon.param_groups)


def test_clip_returns_pre_clip_norm():
    model = tiny_model()
    opt = HybridOptimizer(model)
    ids = torch.randint(2, 512, (2, 8))
    opt.zero_grad()
    model(ids, labels=ids, num_loops=2).loss.backward()
    norm = opt.step(max_grad_norm=1e-6)
    assert norm is not None and norm > 1e-6


def test_lr_scale_applies_to_both_groups_and_restores():
    model = tiny_model()
    opt = HybridOptimizer(model, adamw_lr=6e-4, muon_lr=0.02)
    opt.set_lr_scale(0.5)
    assert all(abs(g["lr"] - 0.01) < 1e-12 for g in opt.muon.param_groups)
    assert all(abs(g["lr"] - 3e-4) < 1e-12 for g in opt.adamw.param_groups)
    opt.set_lr_scale(1.0)
    assert all(abs(g["lr"] - 0.02) < 1e-12 for g in opt.muon.param_groups)


def test_state_dict_round_trip_keeps_momentum():
    model = tiny_model()
    opt = HybridOptimizer(model)
    ids = torch.randint(2, 512, (2, 8))
    opt.zero_grad()
    model(ids, labels=ids, num_loops=2).loss.backward()
    opt.step()

    fresh = HybridOptimizer(tiny_model())
    fresh.load_state_dict(opt.state_dict())
    saved = opt.muon.state[next(iter(opt.muon.param_groups[0]["params"]))]["momentum_buffer"]
    loaded = fresh.muon.state[next(iter(fresh.muon.param_groups[0]["params"]))]["momentum_buffer"]
    assert torch.allclose(saved, loaded)


@pytest.mark.parametrize("rows, cols", [(8, 4), (4, 8), (384, 64), (16, 16)])
def test_default_scale_mode_matches_previous_shape_scale(rows, cols):
    assert muon_scale(rows, cols) == muon_update_scale(rows, cols)
    assert muon_scale(rows, cols, "shape") == muon_update_scale(rows, cols)
    assert Muon([torch.nn.Parameter(torch.zeros(rows, cols))]).scale_mode == "shape"


def test_moonlight_scale_value():
    assert math.isclose(muon_scale(384, 64, "moonlight"), 0.2 * math.sqrt(384))
    assert math.isclose(muon_scale(64, 384, "moonlight"), 0.2 * math.sqrt(384))


def test_invalid_scale_mode_raises():
    with pytest.raises(ValueError):
        muon_scale(8, 4, "adamw")
    with pytest.raises(ValueError):
        Muon([torch.nn.Parameter(torch.zeros(8, 4))], scale_mode="rms")
    with pytest.raises(ValueError):
        HybridOptimizer(tiny_model(), scale_mode="bogus")


def test_scale_mode_survives_state_dict_and_old_dicts_still_load():
    opt = HybridOptimizer(tiny_model(), scale_mode="moonlight")
    state = opt.state_dict()
    assert state["scale_mode"] == "moonlight"

    fresh = HybridOptimizer(tiny_model())
    fresh.load_state_dict(state)
    assert fresh.scale_mode == "moonlight" and fresh.muon.scale_mode == "moonlight"

    legacy = {k: v for k, v in state.items() if k != "scale_mode"}
    legacy["muon"] = {k: v for k, v in state["muon"].items() if k != "scale_mode"}
    kept = HybridOptimizer(tiny_model())
    kept.load_state_dict(legacy)
    assert kept.scale_mode == "shape" and kept.muon.scale_mode == "shape"


def test_moonlight_steps_reduce_loss_on_a_fixed_batch():
    model = tiny_model()
    model.train()
    opt = HybridOptimizer(
        model, adamw_lr=3e-3, muon_lr=0.02, adamw_weight_decay=0.0, scale_mode="moonlight",
    )
    ids = torch.randint(2, 512, (4, 16))

    first = None
    for _ in range(25):
        opt.zero_grad()
        loss = model(ids, labels=ids, num_loops=2).loss
        if first is None:
            first = float(loss.detach())
        loss.backward()
        opt.step(max_grad_norm=1.0)
        model.update_moe_balance()
    last = float(model(ids, labels=ids, num_loops=2).loss)
    assert last < first - 0.2, (first, last)


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
