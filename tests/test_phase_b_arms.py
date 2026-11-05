"""Tests for the Phase B comparison arms: attention output gate, grouped-query attention, the last-loop output,
prelude re-injection and a separate shared-expert width. Every arm defaults to the existing architecture.

Run with: python -B -m pytest tests/test_phase_b_arms.py -q -p no:cacheprovider
"""

import pytest
import torch

from inky import HybridOptimizer, InkyConfig, InkyModel, assign_parameter_groups, loop_diagnostics
from inky.layers import GroupedQueryAttention, LatentKV, MultiLatentAttention, SharedExpertMoE
from inky.quantize import quantize_model
from inky.train import train

OPEN, SEP, CLOSE = 508, 509, 510


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512, scratch_open_id=OPEN, scratch_sep_id=SEP, scratch_close_id=CLOSE, hidden_size=64,
        num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16, q_lora_rank=32,
        kv_lora_rank=24, intermediate_size=96, num_routed_experts=4, num_active_routed=2, ssm_state_dim=16,
        max_recurrent_loops=4, min_recurrent_loops=2, loop_dropout_prob=0.0, max_position_embeddings=256,
        arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4, eos_token_id=1, pad_token_id=1,
        highway_gain_init=0.3,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def make_model(**overrides) -> InkyModel:
    torch.manual_seed(0)
    return InkyModel(tiny_config(**overrides)).eval()


def ids(rows=1, seq=20, seed=1):
    return torch.randint(2, 500, (rows, seq), generator=torch.Generator().manual_seed(seed))


def quiet(_msg):
    return None


def randomise(model, *name_suffixes, std=0.5):
    """Give the zero-initialised arm parameters (gains, gates) non-trivial values."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(name_suffixes):
                p.normal_(0.0, std)


def cached_hidden(model, x, first=8, **kw):
    """Hidden states of x computed as a prefill of ``first`` tokens, then single-token cached steps."""
    out = model(x[:, :first], use_cache=True, return_hidden=True, **kw)
    parts = [out.hidden]
    for t in range(first, x.shape[1]):
        parts.append(model(x[:, t:t + 1], past=out.past, return_hidden=True, **kw).hidden)
    return torch.cat(parts, 1)


# ----------------------------------------------------------------------------- defaults and validation


def test_defaults_are_the_existing_architecture():
    cfg = InkyConfig()
    assert (cfg.attention_type, cfg.attention_gate, cfg.loop_input_injection) == ("mla", False, False)
    assert cfg.routing_mode == "ponder" and cfg.shared_intermediate_size is None
    model = make_model()
    names = [n for n, _ in model.named_parameters()]
    assert not any("out_gate" in n or "inject_gain" in n for n in names)
    assert isinstance(model.pre_layers[0].attn, MultiLatentAttention)
    assert model.count_parameters()["total"] == make_model(attention_type="mla", shared_intermediate_size=None) \
        .count_parameters()["total"]


@pytest.mark.parametrize("kwargs, message", [
    (dict(attention_type="flash"), "attention_type"),
    (dict(attention_type="gqa", num_kv_heads=4), "divisible"),            # the default has 6 heads
    (dict(num_kv_heads=0), "num_kv_heads"),
    (dict(routing_mode="mor", attention_type="gqa"), "attention_type='mla'"),
    (dict(routing_mode="sometimes"), "routing_mode"),
    (dict(shared_intermediate_size=0), "shared_intermediate_size"),
])
def test_invalid_arm_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        InkyConfig(**kwargs)


# ----------------------------------------------------------------------------- attention output gate


@pytest.mark.parametrize("attention_type", ["mla", "gqa"])
def test_gate_adds_one_linear_per_attention_and_nothing_else(attention_type):
    base = make_model(attention_type=attention_type)
    gated = make_model(attention_type=attention_type, attention_gate=True)
    heads, d = 4, 64
    attention_layers = 1 + 2 + 1                                            # pre + 2 recurrent + post
    assert gated.count_parameters()["total"] - base.count_parameters()["total"] == \
        attention_layers * (heads * d + heads)
    assert gated.pre_layers[0].attn.out_gate.weight.shape == (heads, d)


@pytest.mark.parametrize("attention_type", ["mla", "gqa"])
def test_closed_gate_removes_the_attention_output(attention_type):
    plain = make_model(attention_type=attention_type)
    gated = make_model(attention_type=attention_type, attention_gate=True)
    gated.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        for m in gated.modules():
            if getattr(m, "out_gate", None) is not None:
                m.out_gate.weight.zero_()
                m.out_gate.bias.fill_(-60.0)                              # sigmoid -> 0
        for m in plain.modules():
            if isinstance(m, (MultiLatentAttention, GroupedQueryAttention)):
                m.o_proj.weight.zero_()
    x = ids()
    with torch.no_grad():
        assert torch.allclose(gated(x, return_hidden=True).hidden, plain(x, return_hidden=True).hidden, atol=1e-5)


def test_open_gate_matches_a_fixed_half_scale_on_the_attention_output():
    # sigmoid(0) = 0.5: with a zero weight and bias, the gate halves every head's output.
    plain = make_model()
    gated = make_model(attention_gate=True)
    gated.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        for m in gated.modules():
            if getattr(m, "out_gate", None) is not None:
                m.out_gate.weight.zero_()
                m.out_gate.bias.zero_()
        for m in plain.modules():
            if isinstance(m, MultiLatentAttention):
                m.o_proj.weight.mul_(0.5)
    x = ids()
    with torch.no_grad():
        assert torch.allclose(gated(x, return_hidden=True).hidden, plain(x, return_hidden=True).hidden, atol=1e-5)


@pytest.mark.parametrize("extra", [dict(), dict(attention_type="gqa"), dict(routing_mode="mor")])
def test_gate_keeps_cached_decode_equal_to_the_full_forward(extra):
    model = make_model(attention_gate=True, **extra)
    randomise(model, "out_gate.weight", "out_gate.bias")
    x = ids()
    with torch.no_grad():
        assert torch.allclose(cached_hidden(model, x), model(x, return_hidden=True).hidden, atol=1e-5)


def test_gate_keeps_early_exit_equal_to_the_full_forward():
    model = make_model(attention_gate=True)
    randomise(model, "out_gate.weight", "out_gate.bias")
    x = ids()
    with torch.no_grad():
        assert torch.allclose(model.forward_early_exit(x, return_hidden=True).hidden,
                              model(x, return_hidden=True).hidden, atol=1e-5)


def test_gate_receives_gradient():
    model = make_model(attention_gate=True).train()
    x = ids(rows=2)
    model(x, labels=x, num_loops=3).loss.backward()
    grads = [m.out_gate.weight.grad for m in model.modules() if getattr(m, "out_gate", None) is not None]
    assert grads and all(g is not None and float(g.abs().max()) > 0 for g in grads)


# ----------------------------------------------------------------------------- grouped-query attention


def test_gqa_uses_the_grouped_module_with_the_expected_parameters():
    model = make_model(attention_type="gqa", num_kv_heads=2)
    attn = model.pre_layers[0].attn
    assert isinstance(attn, GroupedQueryAttention)
    d, H, Hk, qk, v = 64, 4, 2, 16, 16
    assert attn.q_proj.weight.shape == (H * qk, d) and attn.k_proj.weight.shape == (Hk * qk, d)
    assert attn.v_proj.weight.shape == (Hk * v, d) and attn.o_proj.weight.shape == (d, H * v)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_gqa_cached_decode_matches_the_full_forward(kv_heads, qk_norm):
    model = make_model(attention_type="gqa", num_kv_heads=kv_heads, qk_norm=qk_norm)
    x = ids(seq=24)
    with torch.no_grad():
        full = model(x, return_hidden=True).hidden
        assert torch.allclose(cached_hidden(model, x), full, atol=1e-5)
        # a multi-token chunk after the prefill takes the masked path
        first = model(x[:, :10], use_cache=True, return_hidden=True)
        chunk = model(x[:, 10:17], past=first.past, return_hidden=True).hidden
        assert torch.allclose(chunk, full[:, 10:17], atol=1e-5)


def test_gqa_cache_is_a_latent_kv_with_the_flattened_keys_and_values():
    model = make_model(attention_type="gqa", num_kv_heads=2)
    with torch.no_grad():
        cache = model(ids(seq=9), use_cache=True).past
    entry = cache.pre[0]
    assert isinstance(entry, LatentKV) and entry.length == 9
    assert entry[0].shape == (1, 9, 2 * 16) and entry[1].shape == (1, 1, 9, 2 * 16)


def test_gqa_packed_documents_and_right_padding_match_separate_runs():
    model = make_model(attention_type="gqa")
    a, b = ids(seq=5, seed=2), ids(seq=4, seed=3)
    segments = torch.tensor([[0] * 5 + [1] * 4])
    with torch.no_grad():
        together = model(torch.cat([a, b], 1), segment_ids=segments, num_loops=3).logits
        assert torch.allclose(together[:, :5], model(a, num_loops=3).logits, atol=1e-4)
        assert torch.allclose(together[:, 5:], model(b, num_loops=3).logits, atol=1e-4)
        padded = torch.cat([a, torch.zeros(1, 3, dtype=torch.long)], 1)
        mask = torch.tensor([[1] * 5 + [0] * 3], dtype=torch.bool)
        assert torch.allclose(model(padded, attention_mask=mask, num_loops=3).logits[:, :5],
                              model(a, num_loops=3).logits, atol=1e-4)


def test_gqa_generation_matches_the_stateless_reference():
    model = make_model(attention_type="gqa", initializer_range=0.5)
    prompt = torch.tensor([[2, 3, 4]])
    out = model.generate(prompt, max_new_tokens=8, do_sample=False, repetition_penalty=1.0,
                         no_repeat_ngram_size=0, num_loops=2)
    seq = prompt.clone()
    with torch.no_grad():
        for _ in range(8):
            seq = torch.cat([seq, model(seq, num_loops=2, last_only=True).logits[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, seq)


def test_gqa_rejects_early_exit_and_trains():
    model = make_model(attention_type="gqa")
    with pytest.raises(NotImplementedError, match="attention_type='mla'"):
        model.forward_early_exit(ids())
    model.train()
    x = ids(rows=2)
    out = model(x, labels=x, num_loops=3)
    out.loss.backward()
    assert torch.isfinite(out.loss) and all(p.grad is None or torch.isfinite(p.grad).all()
                                            for p in model.parameters())


# ----------------------------------------------------------------------------- routing_mode="last"


def test_last_mode_has_no_gate_and_matches_ponder_with_one_loop():
    ponder = make_model()
    last = make_model(routing_mode="last")
    assert last.halting is None and not any("halting" in n for n, _ in last.named_parameters())
    last.load_state_dict(ponder.state_dict(), strict=False)
    x = ids()
    with torch.no_grad():
        # with one loop the halting distribution is [1], so the mixture is that loop's state
        assert torch.allclose(last(x, return_hidden=True, num_loops=1).hidden,
                              ponder(x, return_hidden=True, num_loops=1).hidden, atol=1e-5)
        assert not torch.allclose(last(x, return_hidden=True, num_loops=3).hidden,
                                  ponder(x, return_hidden=True, num_loops=3).hidden, atol=1e-3)


def test_last_mode_loss_is_ce_plus_the_small_terms_only():
    model = make_model(routing_mode="last").train()
    x = ids(rows=2)
    out = model(x, labels=x, num_loops=3)
    p = out.loss_parts
    assert float(p["step_ce"]) == 0.0 and float(p["ponder_kl"]) == 0.0
    expected = p["ce"] + model.config.z_loss_coef * p["z_loss"] + p["moe_aux"]
    assert float(out.loss) == pytest.approx(float(expected), rel=1e-5)
    out.loss.backward()
    assert float(out.mean_loops) == 3.0


def test_last_mode_cached_decode_early_exit_and_loop_count():
    model = make_model(routing_mode="last")
    x = ids()
    with torch.no_grad():
        full = model(x, return_hidden=True).hidden
        assert torch.allclose(cached_hidden(model, x), full, atol=1e-5)
        out = model.forward_early_exit(x, return_hidden=True)
        assert torch.allclose(out.hidden, full, atol=1e-5)
        assert out.loop_applications == out.full_loop_applications == x.shape[1] * model.config.max_recurrent_loops
        assert float(model(x).mean_loops) == model.config.max_recurrent_loops
    with pytest.raises(ValueError, match="halt_threshold"):
        model.forward_early_exit(x, halt_threshold=0.5)


def test_last_mode_diagnostics_put_all_weight_on_the_last_loop():
    d = loop_diagnostics(make_model(routing_mode="last"), ids(), num_loops=3)
    assert [r["weight"] for r in d["loops"]] == [0.0, 0.0, 1.0]


def test_last_mode_trains_through_the_loop():
    model = make_model(routing_mode="last")
    batches = [ids(rows=2, seed=s) for s in range(3)]
    history = train(model, HybridOptimizer(model, adamw_lr=1e-3), batches, steps=3, warmup_steps=0, log_fn=quiet)
    assert all(r["skipped"] is False for r in history if r["kind"] == "train")


# ----------------------------------------------------------------------------- loop input injection


def test_injection_starts_identical_to_the_model_without_it():
    plain = make_model()
    injected = make_model(loop_input_injection=True)
    assert injected.core.inject_gain.shape == (64,) and float(injected.core.inject_gain.abs().sum()) == 0.0
    injected.load_state_dict(plain.state_dict(), strict=False)
    x = ids()
    with torch.no_grad():
        assert torch.equal(injected(x, return_hidden=True).hidden, plain(x, return_hidden=True).hidden)


def test_injection_changes_the_output_once_the_gain_is_nonzero_and_gets_a_gradient():
    plain = make_model()
    injected = make_model(loop_input_injection=True)
    injected.load_state_dict(plain.state_dict(), strict=False)
    x = ids(rows=2)
    injected.train()
    injected(x, labels=x, num_loops=3).loss.backward()
    assert float(injected.core.inject_gain.grad.abs().max()) > 0.0
    injected.eval()
    randomise(injected, "inject_gain")
    with torch.no_grad():
        assert not torch.allclose(injected(x, return_hidden=True).hidden, plain(x, return_hidden=True).hidden,
                                  atol=1e-3)


@pytest.mark.parametrize("extra", [dict(), dict(routing_mode="mor"), dict(routing_mode="last")])
def test_injection_keeps_cached_decode_equal_to_the_full_forward(extra):
    model = make_model(loop_input_injection=True, **extra)
    randomise(model, "inject_gain")
    x = ids()
    with torch.no_grad():
        assert torch.allclose(cached_hidden(model, x), model(x, return_hidden=True).hidden, atol=1e-5)


def test_injection_keeps_early_exit_equal_to_the_full_forward():
    model = make_model(loop_input_injection=True)
    randomise(model, "inject_gain")
    x = ids()
    with torch.no_grad():
        assert torch.allclose(model.forward_early_exit(x, return_hidden=True).hidden,
                              model(x, return_hidden=True).hidden, atol=1e-5)


# ----------------------------------------------------------------------------- shared-expert width


def test_shared_width_changes_only_the_shared_expert():
    model = make_model(shared_intermediate_size=40)
    moe = next(m for m in model.modules() if isinstance(m, SharedExpertMoE))
    assert moe.shared_experts[0].gate_proj.weight.shape == (40, 64)
    assert moe.routed_experts[0].gate_proj.weight.shape == (96, 64)
    cfg = model.config
    assert cfg.dense_ffn_size == 40 + 2 * 96 and cfg.shared_ffn_size == 40


def test_fewer_larger_experts_keep_active_width_and_expert_parameters():
    base = make_model()
    wide = make_model(shared_intermediate_size=96, intermediate_size=192, num_routed_experts=2, num_active_routed=1)

    def expert_params(model):
        return sum(p.numel() for n, p in model.named_parameters() if "_experts." in n)

    assert wide.config.dense_ffn_size == base.config.dense_ffn_size == 96 + 2 * 96
    assert expert_params(wide) == expert_params(base)
    x = ids()
    with torch.no_grad():
        assert torch.isfinite(wide(x, return_hidden=True).hidden).all()


# ----------------------------------------------------------------------------- optimiser and quantisation


@pytest.mark.parametrize("kwargs", [
    dict(attention_type="gqa", attention_gate=True),
    dict(attention_gate=True, loop_input_injection=True),
    dict(routing_mode="last", loop_input_injection=True),
])
def test_arms_are_routed_to_the_optimiser_and_take_a_step(kwargs):
    model = make_model(**kwargs).train()
    groups = {name: group for name, group, _ in assign_parameter_groups(model)}
    for name, group in groups.items():
        if name.endswith("inject_gain"):
            assert group == "adamw_no_decay"
        if name.endswith("out_gate.weight"):
            assert group != "muon"                          # (heads x hidden) is a tiny matrix: AdamW
        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            assert group == "muon"
    opt = HybridOptimizer(model, adamw_lr=1e-3)
    x = ids(rows=2)
    model(x, labels=x, num_loops=3).loss.backward()
    opt.step(max_grad_norm=1.0)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_gqa_with_a_gate_quantises_and_still_runs():
    model = make_model(attention_type="gqa", attention_gate=True)
    quantize_model(model, bits=8)
    with torch.no_grad():
        assert torch.isfinite(model(ids(), return_hidden=True).hidden).all()


def test_out_gate_stays_with_adamw_even_when_it_is_large_enough_for_muon():
    # (heads x hidden) = 16 x 64 passes the shape rule for Muon, so the name rule has to keep it with AdamW
    cfg = tiny_config(attention_gate=True, num_attention_heads=16, hidden_size=64, qk_nope_head_dim=4,
                      qk_rope_head_dim=4, v_head_dim=4, q_lora_rank=32)
    groups = {n: g for n, g, _ in assign_parameter_groups(InkyModel(cfg))}
    gate_groups = {g for n, g in groups.items() if n.endswith("out_gate.weight")}
    assert gate_groups == {"adamw_decay"}
