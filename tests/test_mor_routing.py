"""Tests for opt-in Mixture-of-Recursions (MoR) token-level depth routing. Run with: python -m pytest tests -q

Semantics under test (``routing_mode='mor'``, see ``InkyModel.forward``): a router assigns every token a
depth from its own state, loop n runs only the tokens with depth >= n, exited tokens keep their state and
their last latents, and the output is each token's state after its last active loop.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyCache, InkyConfig, InkyModel  # noqa: E402
from inky.speculative import self_speculative_generate  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512, scratch_open_id=OPEN, scratch_sep_id=SEP, scratch_close_id=CLOSE,
        hidden_size=64, num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16,
        q_lora_rank=32, kv_lora_rank=24, intermediate_size=96, num_routed_experts=4, num_active_routed=2,
        ssm_state_dim=16, max_recurrent_loops=4, min_recurrent_loops=2, loop_dropout_prob=0.0,
        max_position_embeddings=256, arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4,
        eos_token_id=1, pad_token_id=1, routing_mode="mor",
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def routed_model(seed: int = 0, router_scale: float = 30.0, **overrides) -> InkyModel:
    """A tiny MoR model whose router is sharp enough to produce a mix of depths (ties are not close)."""
    torch.manual_seed(seed)
    model = InkyModel(tiny_config(**overrides)).eval()
    with torch.no_grad():
        model.mor_router.router.weight.normal_(0.0, router_scale)
    return model


def depths_of(model: InkyModel, ids: torch.Tensor, loops=None) -> torch.Tensor:
    """The routed depth of every token, computed from the model's own pre layers (eval mode)."""
    with torch.no_grad():
        S = ids.shape[1]
        cos, sin = model.rope(0, S)
        h = model.embed_tokens(ids)
        for layer in model.pre_layers:
            h, _, _ = layer(h, cos, sin, None)
        depth, _, _ = model._mor_route(h, loops or model.config.max_recurrent_loops, None, None)
    return depth


# ------------------------------------------------------------------ config


def test_defaults_keep_ponder_and_the_documented_parameter_count():
    cfg = InkyConfig()
    assert cfg.routing_mode == "ponder"
    assert cfg.mor_router_hidden is None
    assert cfg.mor_capacity_aux_coef == pytest.approx(1e-2)
    assert cfg.mor_gate_scale is True
    assert cfg.mor_temperature == 1.0
    assert cfg.mor_target == cfg.ponder_prior_steps
    model = InkyModel(cfg)
    assert model.count_parameters()["total"] == 50_219_201
    assert model.halting is not None and model.mor_router is None
    assert not any(k.startswith("mor_router") for k in model.state_dict())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(routing_mode="moe"),
        dict(routing_mode="mor", mor_router_hidden=0),
        dict(mor_router_hidden=-3),
        dict(mor_capacity_aux_coef=-0.1),
        dict(mor_temperature=0.0),
        dict(mor_temperature=-1.0),
        dict(mor_target_depth=0.5),
        dict(routing_mode="mor", min_recurrent_loops=0),
        dict(routing_mode="mor", min_recurrent_loops=5, max_recurrent_loops=3),
        dict(routing_mode="mor", ssm_mixer_layers=(0,)),
    ],
)
def test_invalid_mor_config_is_refused(kwargs):
    with pytest.raises(ValueError):
        InkyConfig(**kwargs)


def test_mor_target_defaults_to_ponder_prior_and_is_clipped():
    assert InkyConfig(routing_mode="mor", ponder_prior_steps=4.0).mor_target == 4.0
    assert InkyConfig(routing_mode="mor", mor_target_depth=2.5).mor_target == 2.5
    assert InkyConfig(routing_mode="mor", mor_target_depth=99.0, max_recurrent_loops=6).mor_target == 6.0


def test_config_round_trips_through_dict():
    cfg = tiny_config(mor_router_hidden=16, mor_temperature=0.7, mor_gate_scale=False, mor_target_depth=3.5)
    again = InkyConfig.from_dict(cfg.to_dict())
    for name in ("routing_mode", "mor_router_hidden", "mor_temperature", "mor_gate_scale", "mor_target_depth",
                 "mor_capacity_aux_coef", "mor_sample_depth", "mor_gather"):
        assert getattr(again, name) == getattr(cfg, name), name


# ------------------------------------------------------------------ default (ponder) is unchanged


def test_explicit_ponder_flag_equals_flag_absent():
    kwargs = tiny_config().to_dict()
    kwargs.pop("routing_mode")
    absent = InkyConfig(**{k: v for k, v in kwargs.items() if k not in ("model_type", "transformers_version")})
    explicit = tiny_config(routing_mode="ponder")
    torch.manual_seed(3)
    a = InkyModel(absent).eval()
    torch.manual_seed(3)
    b = InkyModel(explicit).eval()
    assert absent.routing_mode == "ponder"
    ids = torch.randint(2, 500, (2, 10))
    with torch.no_grad():
        oa, ob = a(ids, return_hidden=True), b(ids, return_hidden=True)
    assert torch.equal(oa.hidden, ob.hidden)
    assert torch.equal(oa.mean_loops, ob.mean_loops)
    assert oa.loop_applications is None and ob.loop_applications is None     # ponder runs every loop


def test_ponder_model_has_no_router_and_mor_model_has_no_halting_gate():
    ponder = InkyModel(tiny_config(routing_mode="ponder"))
    mor = InkyModel(tiny_config())
    assert ponder.mor_router is None and ponder.halting is not None
    assert mor.mor_router is not None and mor.halting is None
    assert "mor_router" in mor.count_parameters()
    assert mor.count_parameters()["total"] == sum(p.numel() for p in mor.parameters())


# ------------------------------------------------------------------ training: loss, gradients, compute


def test_forward_backward_is_finite_and_reaches_router_blocks_and_embedding():
    torch.manual_seed(0)
    model = InkyModel(tiny_config(loop_dropout_prob=0.0)).train()
    ids = torch.randint(2, 500, (3, 14))
    out = model(ids, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    for name, g in grads.items():
        assert g is not None, f"no gradient for {name}"
        assert torch.isfinite(g).all(), name
    assert grads["embed_tokens.weight"].abs().sum() > 0
    assert grads["mor_router.router.weight"].abs().sum() > 0
    assert any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith("core.layers.0."))
    assert any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith("core.ssm."))
    parts = out.loss_parts
    assert set(parts) >= {"ce", "step_ce", "z_loss", "moe_aux", "ponder_kl", "mor_balance"}
    assert float(parts["step_ce"]) == 0.0 and float(parts["ponder_kl"]) == 0.0
    expected = (parts["ce"] + model.config.z_loss_coef * parts["z_loss"] + parts["moe_aux"]
                + model.config.mor_capacity_aux_coef * parts["mor_balance"])
    assert torch.allclose(out.loss.detach(), expected, atol=1e-5)
    assert 2.0 <= float(out.mean_loops) <= 4.0


def test_gate_scale_off_leaves_the_router_only_the_balance_gradient():
    torch.manual_seed(0)
    model = InkyModel(tiny_config(mor_gate_scale=False)).train()
    ids = torch.randint(2, 500, (2, 10))
    out = model(ids, labels=ids)
    out.loss.backward()
    assert model.mor_router.router.weight.grad.abs().sum() > 0       # from the balance loss
    off = InkyModel(tiny_config(mor_gate_scale=False, mor_capacity_aux_coef=0.0)).train()
    off(ids, labels=ids).loss.backward()
    g = off.mor_router.router.weight.grad
    assert g is None or float(g.abs().sum()) == 0.0


def test_training_loss_goes_down_with_a_few_steps():
    torch.manual_seed(0)
    model = InkyModel(tiny_config()).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = torch.randint(2, 100, (4, 16))
    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss = model(ids, labels=ids).loss
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_balance_loss_pulls_the_mean_depth_towards_the_target():
    torch.manual_seed(0)
    model = InkyModel(tiny_config(mor_target_depth=2.0)).train()
    with torch.no_grad():
        model.mor_router.router.bias.copy_(torch.tensor([0.0, 0.0, 3.0]))      # starts far above the target
    opt = torch.optim.SGD(model.mor_router.parameters(), lr=1.0)
    ids = torch.randint(2, 500, (2, 8))
    h = model.embed_tokens(ids).detach()
    first = float(model._mor_balance(torch.softmax(model.mor_router(h), -1), None).detach())
    for _ in range(60):
        opt.zero_grad()
        model._mor_balance(torch.softmax(model.mor_router(h), -1), None).backward()
        opt.step()
    last = float(model._mor_balance(torch.softmax(model.mor_router(h), -1), None))
    assert first > 1.0 and last < 0.1 * first


def test_initial_depth_distribution_matches_the_target_depth():
    model = InkyModel(tiny_config(mor_target_depth=3.0))
    probs = torch.softmax(model.mor_router.router.bias.detach(), -1)
    mean = float((probs * torch.arange(2, 5)).sum())
    assert abs(mean - 3.0) < 0.35


def test_gather_path_equals_masked_path_in_training():
    ids = torch.randint(2, 500, (3, 13))
    results = {}
    for gather in (True, False):
        torch.manual_seed(1)
        model = InkyModel(tiny_config(mor_gather=gather)).train()
        with torch.no_grad():
            model.mor_router.router.weight.normal_(0.0, 5.0)
        torch.manual_seed(7)                                   # same Gumbel draws
        out = model(ids, labels=ids)
        out.loss.backward()
        results[gather] = (out, {n: p.grad.clone() for n, p in model.named_parameters()})
    (og, gg), (om, gm) = results[True], results[False]
    assert og.mean_loops == om.mean_loops
    assert float(og.mean_loops) < 4.0                           # a real mix of depths
    assert og.loop_applications < om.loop_applications          # the masked path computes every row
    assert torch.allclose(og.loss, om.loss, atol=1e-5), (float(og.loss), float(om.loss))
    for name in gg:
        assert torch.allclose(gg[name], gm[name], atol=1e-4, rtol=1e-3), name


def test_gather_path_equals_masked_path_in_eval_with_cache_off():
    ids = torch.randint(2, 500, (3, 13))
    a = routed_model(mor_gather=True)
    b = routed_model(mor_gather=False)
    with torch.no_grad():
        oa, ob = a(ids, return_hidden=True), b(ids, return_hidden=True)
    assert torch.allclose(oa.hidden, ob.hidden, atol=1e-5)
    assert torch.allclose(oa.moe_aux_loss, ob.moe_aux_loss, atol=1e-6)
    assert float(oa.mean_loops) < 4.0


def test_compute_drops_when_the_mean_depth_is_below_the_maximum():
    ids = torch.randint(2, 500, (2, 16))
    mor = routed_model()
    ponder_cfg = tiny_config(routing_mode="ponder").to_dict()
    torch.manual_seed(0)
    ponder = InkyModel(InkyConfig(**{k: v for k, v in ponder_cfg.items() if k != "model_type"})).eval()

    def spy(model, attr, rows_of):
        rows = []
        for layer in model.core.layers:
            original = getattr(layer, attr)

            def wrapped(*a, _orig=original, **k):
                rows.append(rows_of(a[0]))
                return _orig(*a, **k)

            setattr(layer, attr, wrapped)
        return rows

    mor_rows = spy(mor, "forward_mor", lambda x: x.shape[0])
    ponder_rows = spy(ponder, "_forward", lambda x: x.shape[0] * x.shape[1])
    with torch.no_grad():
        out = mor(ids, return_hidden=True)
        ponder(ids, return_hidden=True)
    depth = depths_of(mor, ids)
    n_layers = mor.config.num_recurrent_layers
    assert float(depth.float().mean()) < mor.config.max_recurrent_loops
    assert sum(mor_rows) == int(depth.sum()) * n_layers          # exactly the routed token-loop pairs
    assert sum(mor_rows) < sum(ponder_rows)
    assert sum(ponder_rows) == ids.numel() * mor.config.max_recurrent_loops * n_layers
    assert out.loop_applications == int(depth.sum())
    assert out.full_loop_applications == ids.numel() * mor.config.max_recurrent_loops
    assert float(out.mean_loops) == pytest.approx(float(depth.float().mean()))


def test_mor_training_does_not_draw_a_random_loop_count():
    model = InkyModel(tiny_config()).train()
    assert model._choose_loops(None, None) == model.config.max_recurrent_loops
    assert model._choose_loops(None, 3) == 3


def test_checkpointing_gives_the_same_loss_and_gradients():
    ids = torch.randint(2, 500, (2, 11))
    got = {}
    for ckpt in (False, True):
        torch.manual_seed(2)
        model = InkyModel(tiny_config(gradient_checkpointing=ckpt)).train()
        with torch.no_grad():
            model.mor_router.router.weight.normal_(0.0, 5.0)
        torch.manual_seed(9)
        out = model(ids, labels=ids)
        out.loss.backward()
        got[ckpt] = (out.loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()})
    assert torch.allclose(got[False][0], got[True][0], atol=1e-5)
    for name, g in got[False][1].items():
        assert torch.allclose(g, got[True][1][name], atol=1e-4, rtol=1e-3), name


def test_loop_dropout_and_dropouts_train_without_error():
    torch.manual_seed(0)
    model = InkyModel(tiny_config(loop_dropout_prob=0.5, residual_dropout=0.1, embedding_dropout=0.1)).train()
    ids = torch.randint(2, 500, (2, 12))
    for _ in range(3):
        out = model(ids, labels=ids)
        assert torch.isfinite(out.loss)
        out.loss.backward()


def test_dense_ffn_and_no_highway_and_no_step_modulation_variants_work():
    ids = torch.randint(2, 500, (2, 10))
    for kw in (dict(use_moe=False), dict(use_highway=False), dict(use_step_modulation=False),
               dict(qk_norm=True), dict(qk_norm=True, qk_norm_mode="latent")):
        model = routed_model(**kw)
        with torch.no_grad():
            g = model(ids, return_hidden=True)
            model.config.mor_gather = False
            m = model(ids, return_hidden=True)
        assert torch.allclose(g.hidden, m.hidden, atol=1e-5), kw
        model.train()
        model.config.mor_gather = True
        loss = model(ids, labels=ids).loss
        assert torch.isfinite(loss), kw
        loss.backward()


def test_bf16_autocast_forward_is_finite():
    ids = torch.randint(2, 500, (2, 10))
    model = InkyModel(tiny_config()).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()


# ------------------------------------------------------------------ equivalence with hand-built references


def test_all_depths_at_maximum_equals_a_plain_recurrent_run_without_mixture():
    model = routed_model(mor_gate_scale=False)
    with torch.no_grad():
        model.mor_router.router.weight.zero_()
        model.mor_router.router.bias.copy_(torch.tensor([0.0, 0.0, 50.0]))      # depth 4 = max, always
    ids = torch.randint(2, 500, (2, 12))
    with torch.no_grad():
        got = model(ids, return_hidden=True)
        # Reference built by hand from the pieces: every token, every loop, no halting mixture.
        S = ids.shape[1]
        cos, sin = model.rope(0, S)
        h = model.embed_tokens(ids)
        for layer in model.pre_layers:
            h, _, _ = layer(h, cos, sin, None)
        prev_mem = None
        for step in range(1, model.config.max_recurrent_loops + 1):
            h, prev_mem, _, _, _ = model.core(h, step, cos, sin, prev_mem=prev_mem)
        for layer in model.post_layers:
            h, _, _ = layer(h, cos, sin, None)
        want = model.norm(h)
    assert float(got.mean_loops) == 4.0
    assert got.loop_applications == got.full_loop_applications
    assert torch.allclose(got.hidden, want, atol=1e-5), float((got.hidden - want).abs().max())


def test_gate_scale_with_a_saturated_router_equals_the_plain_run_too():
    model = routed_model(mor_gate_scale=True)
    with torch.no_grad():
        model.mor_router.router.weight.zero_()
        model.mor_router.router.bias.copy_(torch.tensor([0.0, 0.0, 80.0]))
    ids = torch.randint(2, 500, (1, 9))
    other = routed_model(mor_gate_scale=False)
    other.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.allclose(model(ids, return_hidden=True).hidden, other(ids, return_hidden=True).hidden,
                              atol=1e-5)


@pytest.mark.parametrize("extra", [dict(), dict(use_highway=False), dict(use_step_modulation=False)])
def test_mixed_depths_match_the_forward_subset_reference(extra):
    """Independent reference: the batch-size-1 inference machinery of forward_early_exit (forward_subset)."""
    model = routed_model(mor_gate_scale=False, **extra)
    ids = torch.randint(2, 500, (1, 15))
    depth = depths_of(model, ids)[0]
    assert depth.min() < depth.max(), "the test needs a mix of depths"
    with torch.no_grad():
        got = model(ids, return_hidden=True).hidden
        cfg = model.config
        S = ids.shape[1]
        cos, sin = model.rope(0, S)
        h = model.embed_tokens(ids)
        for layer in model.pre_layers:
            h, _, _ = layer(h, cos, sin, None)
        state = h.clone()
        stores = [
            (torch.zeros(1, S, cfg.kv_lora_rank), torch.zeros(1, 1, S, cfg.qk_rope_head_dim))
            for _ in model.core.layers
        ]
        mem = None
        for n in range(1, cfg.max_recurrent_loops + 1):
            active = torch.nonzero(depth >= n).squeeze(-1)
            if active.numel() == 0:
                break
            out_act, mem = model.core.forward_subset(state, n, cos, sin, active, stores, mem)
            state[0, active] = out_act[0]
        for layer in model.post_layers:
            state, _, _ = layer(state, cos, sin, None)
        want = model.norm(state)
    assert torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())


def test_exited_tokens_keep_their_state_and_deeper_tokens_see_their_stale_latents():
    """Frozen-state contract: a token with depth 2 is unaffected by what happens in loops 3 and 4."""
    model = routed_model()
    ids = torch.randint(2, 500, (1, 12))
    depth = depths_of(model, ids)[0]
    first_exit = int(torch.nonzero(depth == depth.min())[0])
    with torch.no_grad():
        full = model(ids, return_hidden=True).hidden
        capped = model(ids, num_loops=int(depth.min()), return_hidden=True).hidden
    # Capping the loops at the shallowest depth cannot change that token's state before the post layers,
    # but the post layers mix in deeper neighbours, so only check the cap really changes deeper tokens.
    assert not torch.allclose(full, capped, atol=1e-6)
    assert first_exit >= 0


# ------------------------------------------------------------------ cached decoding


@pytest.mark.parametrize("extra", [dict(), dict(use_highway=False), dict(use_step_modulation=False),
                                   dict(use_moe=False), dict(qk_norm=True, qk_norm_mode="latent")])
def test_cached_decode_equals_the_full_forward(extra):
    model = routed_model(**extra)
    ids = torch.randint(2, 500, (2, 14))
    with torch.no_grad():
        full = model(ids).logits
        cache = InkyCache(model.config)
        steps = [model(ids[:, t:t + 1], past=cache).logits for t in range(ids.shape[1])]
    assert float(depths_of(model, ids).float().mean()) < model.config.max_recurrent_loops
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4), float((torch.cat(steps, 1) - full).abs().max())
    assert cache.seq_len == ids.shape[1]


def test_cached_prefill_chunks_and_mixed_chunk_sizes_equal_the_full_forward():
    model = routed_model(seed=4)
    ids = torch.randint(2, 500, (2, 17))
    with torch.no_grad():
        full = model(ids).logits
        cache = InkyCache(model.config)
        parts = []
        pos = 0
        for size in (5, 1, 1, 6, 3, 1):
            parts.append(model(ids[:, pos:pos + size], past=cache).logits)
            pos += size
    assert pos == ids.shape[1]
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-4)


def test_cache_fills_every_loop_with_full_length_entries():
    model = routed_model()
    ids = torch.randint(2, 500, (1, 9))
    with torch.no_grad():
        cache = InkyCache(model.config)
        model(ids, past=cache)
        model(ids[:, :1], past=cache)
    assert cache.num_loops == model.config.max_recurrent_loops
    for entries in cache.core:
        for entry in entries:
            assert entry[0].shape[1] == 10 and entry[1].shape[2] == 10
    assert all(p is not None for p in cache.prefix)


def test_num_loops_is_a_cap_and_is_fixed_by_the_first_cached_call():
    model = routed_model()
    ids = torch.randint(2, 500, (1, 12))
    with torch.no_grad():
        capped = model(ids, num_loops=2, return_hidden=True)
        assert float(capped.mean_loops) == 2.0
        full = model(ids, return_hidden=True)
        assert float(full.mean_loops) > 2.0
        cache = InkyCache(model.config)
        model(ids[:, :6], past=cache, num_loops=2)
        spare = InkyCache(model.config)
        model(ids[:, :6], past=spare, num_loops=2)
        with pytest.raises(ValueError):
            model(ids[:, 6:], past=spare, num_loops=3)
        rest = model(ids[:, 6:], past=cache).logits                         # keeps the cap of 2
        whole = model(ids, num_loops=2).logits
    assert torch.allclose(rest, whole[:, 6:], atol=1e-4)


def test_generate_with_and_without_cache_agree_under_mor():
    model = routed_model(seed=5)
    prompt = torch.randint(2, 500, (1, 6))
    out = model.generate(prompt, max_new_tokens=10, do_sample=False, repetition_penalty=1.0,
                         no_repeat_ngram_size=0, eos_token_id=None)
    seq = prompt
    with torch.no_grad():
        for _ in range(10):
            logits = model(seq).logits[:, -1]
            logits[:, 508:511] = float("-inf")
            logits[:, 49974:] = float("-inf")
            seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, seq)


def test_sampling_generate_and_batched_generate_run_under_mor():
    model = routed_model(seed=6)
    prompt = torch.randint(2, 500, (2, 5))
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=8, do_sample=True, eos_token_id=None)
    assert out.shape == (2, 13)


def test_scratch_reduction_runs_under_mor():
    model = routed_model(seed=7)
    with torch.no_grad():
        model.embed_tokens.weight[OPEN].zero_()
    prompt = torch.randint(2, 500, (1, 5))
    out = model.generate(prompt, max_new_tokens=12, do_sample=False, reduce_scratch_blocks=True,
                         eos_token_id=None, repetition_penalty=1.0, no_repeat_ngram_size=0)
    assert out.shape[0] == 1 and out.shape[1] >= 5


def test_scratch_rollback_restores_the_mor_cache_exactly():
    """Snapshot / restore (used by scratch reduction) must give the same logits as never having fed the block."""
    from inky.modeling import _cache_restore, _cache_snapshot
    model = routed_model(seed=8)
    ids = torch.randint(2, 500, (1, 10))
    with torch.no_grad():
        cache = InkyCache(model.config)
        model(ids[:, :4], past=cache)
        snap = _cache_snapshot(cache)
        model(ids[:, 4:8], past=cache)                      # the "notes"
        _cache_restore(cache, snap)
        after = model(ids[:, 8:], past=cache).logits
        ref_cache = InkyCache(model.config)
        model(ids[:, :4], past=ref_cache)
        ref = model(ids[:, 8:], past=ref_cache).logits
    assert torch.allclose(after, ref, atol=1e-5)


def test_self_speculative_generate_matches_greedy_under_mor():
    model = routed_model(seed=9)
    prompt = torch.randint(2, 500, (1, 6))
    plain = model.generate(prompt, max_new_tokens=12, do_sample=False, repetition_penalty=1.0,
                           no_repeat_ngram_size=0, eos_token_id=None,
                           num_loops=model.config.max_recurrent_loops)
    spec, stats = self_speculative_generate(model, prompt, 12, draft_loops=2, draft_len=3, eos_token_id=None)
    assert torch.equal(plain, spec)
    assert stats["drafted"] > 0


# ------------------------------------------------------------------ early exit follows the router


def test_forward_early_exit_uses_the_router_depth():
    model = routed_model(seed=10)
    ids = torch.randint(2, 500, (1, 14))
    depth = depths_of(model, ids)
    with torch.no_grad():
        full = model(ids, return_hidden=True)
        early = model.forward_early_exit(ids, return_hidden=True)
        logits = model.forward_early_exit(ids, last_only=True).logits
    assert torch.allclose(full.hidden, early.hidden, atol=1e-6)
    assert early.loop_applications == int(depth.sum())
    assert early.full_loop_applications == ids.shape[1] * model.config.max_recurrent_loops
    assert float(early.mean_exit_loop) == pytest.approx(float(depth.float().mean()))
    assert float(early.mean_loops) == pytest.approx(float(depth.float().mean()))
    assert logits.shape == (1, 1, model.config.vocab_size)
    with pytest.raises(ValueError):
        model.forward_early_exit(ids, halt_threshold=0.5)
    with pytest.raises(ValueError):
        model.forward_early_exit(torch.randint(2, 500, (2, 6)))


# ------------------------------------------------------------------ packing


def test_attention_mask_right_padding_matches_the_unpadded_run():
    model = routed_model(seed=11)
    ids = torch.randint(2, 500, (1, 9))
    pad = torch.cat([ids, torch.full((1, 4), 1)], dim=1)
    mask = torch.cat([torch.ones(1, 9), torch.zeros(1, 4)], dim=1).bool()
    with torch.no_grad():
        plain = model(ids, return_hidden=True)
        padded = model(pad, attention_mask=mask, return_hidden=True)
    assert torch.allclose(padded.hidden[:, :9], plain.hidden, atol=1e-4), float(
        (padded.hidden[:, :9] - plain.hidden).abs().max())
    assert float(padded.mean_loops) == pytest.approx(float(plain.mean_loops))      # real positions only
    assert padded.loop_applications == plain.loop_applications                     # pads never run


def test_attention_mask_left_padding_matches_the_unpadded_run():
    model = routed_model(seed=12)
    ids = torch.randint(2, 500, (1, 8))
    pad = torch.cat([torch.full((1, 3), 1), ids], dim=1)
    mask = torch.cat([torch.zeros(1, 3), torch.ones(1, 8)], dim=1).bool()
    with torch.no_grad():
        plain = model(ids, return_hidden=True).hidden
        padded = model(pad, attention_mask=mask, return_hidden=True).hidden
    assert torch.allclose(padded[:, 3:], plain, atol=1e-4), float((padded[:, 3:] - plain).abs().max())


def test_segment_ids_packed_documents_match_separate_runs():
    model = routed_model(seed=13)
    a = torch.randint(2, 500, (1, 7))
    b = torch.randint(2, 500, (1, 6))
    packed = torch.cat([a, b], dim=1)
    seg = torch.cat([torch.zeros(1, 7), torch.ones(1, 6)], dim=1).long()
    with torch.no_grad():
        got = model(packed, segment_ids=seg, return_hidden=True).hidden
    # The post layers are masked per segment too, so each document equals its own run exactly.
    with torch.no_grad():
        want = torch.cat([model(a, return_hidden=True).hidden, model(b, return_hidden=True).hidden], dim=1)
    assert torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())


def test_packing_arguments_are_refused_with_a_cache():
    model = routed_model()
    ids = torch.randint(2, 500, (1, 6))
    with pytest.raises(ValueError):
        model(ids, past=InkyCache(model.config), attention_mask=torch.ones(1, 6).bool())


def test_ssm_mixer_layers_are_refused_for_mor():
    with pytest.raises(ValueError):
        tiny_config(ssm_mixer_layers=(0,))
    # A model built with ponder routing and an SSM mixer cannot be switched to mor afterwards either.
    model = InkyModel(tiny_config(routing_mode="ponder", ssm_mixer_layers=(0,))).eval()
    model.config.routing_mode = "mor"
    model.mor_router = None
    with pytest.raises((NotImplementedError, TypeError, AttributeError)):
        model(torch.randint(2, 500, (1, 6)))


# ------------------------------------------------------------------ save / load


def test_save_and_load_round_trip(tmp_path):
    torch.manual_seed(14)
    model = InkyModel(tiny_config(mor_router_hidden=16, mor_temperature=0.8)).eval()
    with torch.no_grad():
        model.mor_router.router[0].weight.normal_(0.0, 3.0)
        model.mor_router.router[2].weight.normal_(0.0, 3.0)
    ids = torch.randint(2, 500, (2, 11))
    with torch.no_grad():
        before = model(ids, return_hidden=True)
    model.save_pretrained(tmp_path)
    loaded = InkyModel.from_pretrained(tmp_path).eval()
    assert loaded.config.routing_mode == "mor"
    assert loaded.config.mor_router_hidden == 16 and loaded.config.mor_temperature == 0.8
    with torch.no_grad():
        after = loaded(ids, return_hidden=True)
    assert torch.equal(before.hidden, after.hidden)
    assert float(before.mean_loops) == float(after.mean_loops)


def test_router_with_a_hidden_layer_trains():
    torch.manual_seed(0)
    model = InkyModel(tiny_config(mor_router_hidden=8)).train()
    ids = torch.randint(2, 500, (2, 9))
    model(ids, labels=ids).loss.backward()
    assert all(p.grad is not None for p in model.mor_router.parameters())
