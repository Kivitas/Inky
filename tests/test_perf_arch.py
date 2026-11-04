"""Tests for the growable decode cache, latent QK-norm, and the dense / no-highway ablation switches.

Run with: python -m pytest tests/test_perf_arch.py -q -p no:cacheprovider
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import (  # noqa: E402
    InkyConfig,
    InkyModel,
    modeling,
)
from inky import layers as layers_mod  # noqa: E402
from inky.layers import DenseFFN, LatentKV, MultiLatentAttention, RotaryEmbedding, SharedExpertMoE  # noqa: E402
from inky.speculative import self_speculative_generate  # noqa: E402

DEFAULT_PARAM_COUNT = 50_219_201
KV_RANK = 24   # distinct from every other width in the tiny config, so cats on it are recognisable


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512, scratch_open_id=508, scratch_sep_id=509, scratch_close_id=510,
        hidden_size=64, num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16,
        q_lora_rank=32, kv_lora_rank=KV_RANK, intermediate_size=96, num_routed_experts=4, num_active_routed=2,
        ssm_state_dim=16, max_recurrent_loops=3, min_recurrent_loops=2, loop_dropout_prob=0.0,
        max_position_embeddings=64, initializer_range=0.3, eos_token_id=1, pad_token_id=1,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def make_model(**overrides) -> InkyModel:
    torch.manual_seed(0)
    return InkyModel(tiny_config(**overrides)).eval()


def decode_logits(model, ids, prefix, loops=3):
    """Prefill ``prefix`` tokens, then feed the rest one token at a time. Returns (logits [B, S, V], cache)."""
    with torch.no_grad():
        out = model(ids[:, :prefix], use_cache=True, num_loops=loops)
        cache = out.past
        steps = [out.logits]
        for t in range(prefix, ids.shape[1]):
            steps.append(model(ids[:, t:t + 1], past=cache).logits)
    return torch.cat(steps, dim=1), cache


def cat_append(self, past, c_new, k_new):
    """The previous cache update (concatenate the whole cache), used as the reference."""
    c = torch.cat([past[0], c_new], dim=1)
    k = torch.cat([past[1], k_new], dim=2)
    return LatentKV(c, k, c.shape[1])


# ---------------------------------------------------------------- 1. growable decode cache

def test_cached_decode_matches_full_forward_over_many_tokens():
    model = make_model()
    ids = torch.randint(2, 500, (2, 48))
    with torch.no_grad():
        full = model(ids, num_loops=3).logits
    incremental, cache = decode_logits(model, ids, prefix=5)
    assert torch.allclose(full, incremental, atol=1e-4), float((full - incremental).abs().max())
    assert cache.seq_len == 48


def test_buffer_cache_is_identical_to_the_concatenating_cache(monkeypatch):
    model = make_model()
    ids = torch.randint(2, 500, (2, 30))
    new, _ = decode_logits(model, ids, prefix=7)
    monkeypatch.setattr(MultiLatentAttention, "_append", cat_append)
    ref, _ = decode_logits(model, ids, prefix=7)
    assert torch.allclose(new, ref, atol=1e-6, rtol=0), float((new - ref).abs().max())


def test_entries_still_unpack_as_the_latent_pair():
    model = make_model()
    ids = torch.randint(2, 500, (2, 9))
    with torch.no_grad():
        cache = model(ids, use_cache=True, num_loops=3).past
    entry = cache.pre[0]
    c_kv, k_pe = entry
    assert isinstance(entry, tuple) and len(entry) == 2
    assert c_kv.shape == (2, 9, KV_RANK) and k_pe.shape == (2, 1, 9, 4)
    assert entry.length == 9 and entry.capacity >= 9


def test_decode_does_not_reallocate_or_concatenate_the_cache(monkeypatch):
    model = make_model()
    ids = torch.randint(2, 500, (1, 40))
    with torch.no_grad():
        out = model(ids[:, :20], use_cache=True, num_loops=3)
        cache = out.past
        model(ids[:, 20:21], past=cache)         # first decode step: the exact-fit prefill buffer grows once

    def entries():
        return list(cache.pre) + [e for loop in cache.core for e in loop] + list(cache.post)

    ptrs = [e.c_buf.data_ptr() for e in entries()]
    kptrs = [e.k_buf.data_ptr() for e in entries()]
    caps = [e.capacity for e in entries()]
    assert all(cap > e.length for cap, e in zip(caps, entries()))     # spare room after the growth

    real_cat = torch.cat
    cache_cats = []

    def spy(tensors, *args, **kwargs):
        if any(t.ndim == 3 and t.shape[-1] == KV_RANK for t in tensors):
            cache_cats.append(1)
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(layers_mod.torch, "cat", spy)
    with torch.no_grad():
        for t in range(21, 36):
            model(ids[:, t:t + 1], past=cache)
    monkeypatch.undo()

    assert cache_cats == []                                           # the latent cache is never concatenated
    assert [e.c_buf.data_ptr() for e in entries()] == ptrs            # same storage across 15 steps
    assert [e.k_buf.data_ptr() for e in entries()] == kptrs
    assert [e.capacity for e in entries()] == caps
    assert all(e.length == 36 for e in entries())


def test_capacity_grows_geometrically_and_stays_within_the_context_limit():
    model = make_model(max_position_embeddings=64)
    ids = torch.randint(2, 500, (1, 64))
    caps = []
    with torch.no_grad():
        cache = model(ids[:, :3], use_cache=True, num_loops=2).past
        for t in range(3, 64):
            model(ids[:, t:t + 1], past=cache)
            caps.append(cache.pre[0].capacity)
    assert caps[0] == 6                      # 3 -> doubled on the first append
    assert sorted(set(caps)) == [6, 12, 24, 48, 64]   # doubles, then clamps at max_position_embeddings
    assert max(caps) == 64


def test_fork_does_not_corrupt_the_parent_cache():
    """A fork shares buffers with its parent. Feeding the fork must leave the parent's later output unchanged."""
    model = make_model()
    ids = torch.randint(2, 500, (1, 24))
    other = torch.randint(2, 500, (1, 5))
    with torch.no_grad():
        parent = model(ids[:, :10], use_cache=True, num_loops=3).past
        model(ids[:, 10:11], past=parent)                     # parent now owns spare capacity
        fork = copy.copy(parent)
        fork.pre, fork.post = list(parent.pre), list(parent.post)
        fork.core, fork.prefix = list(parent.core), list(parent.prefix)
        for t in range(other.shape[1]):                       # the fork writes different tokens
            model(other[:, t:t + 1], past=fork)
        got = model(ids[:, 11:], past=parent).logits
        want = model(ids, num_loops=3).logits[:, 11:]
    assert torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())


def test_snapshot_restore_roundtrip():
    model = make_model()
    ids = torch.randint(2, 500, (1, 20))
    alt = torch.randint(2, 500, (1, 4))
    with torch.no_grad():
        cache = model(ids[:, :8], use_cache=True, num_loops=3).past
        model(ids[:, 8:9], past=cache)
        snap = modeling._cache_snapshot(cache)
        for t in range(alt.shape[1]):
            model(alt[:, t:t + 1], past=cache)
        modeling._cache_restore(cache, snap)
        got = model(ids[:, 9:], past=cache).logits
        want = model(ids, num_loops=3).logits[:, 9:]
    assert torch.allclose(got, want, atol=1e-4)


def test_speculative_decoding_still_matches_greedy():
    model = make_model(max_position_embeddings=128)
    prompt = torch.tensor([[2, 3, 4, 5]])
    kw = dict(do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0, eos_token_id=None)
    plain = model.generate(prompt, max_new_tokens=24, num_loops=3, **kw)
    spec, _ = self_speculative_generate(model, prompt, 24, draft_loops=2, draft_len=3, eos_token_id=None)
    assert torch.equal(plain, spec)


def test_layer_level_past_tuple_is_not_mutated():
    """Passing the same past twice (or a plain tuple) must not let one call overwrite the other's data."""
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = MultiLatentAttention(cfg).eval()
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, 64, cfg.rope_theta)
    cos, sin = rope(0, 20)
    x = torch.randn(1, 20, cfg.hidden_size)
    with torch.no_grad():
        _, past = attn(x[:, :8], cos[:8], sin[:8])
        a, pa = attn(x[:, 8:9], cos[8:9], sin[8:9], past=past)
        b, pb = attn(torch.randn(1, 1, cfg.hidden_size), cos[8:9], sin[8:9], past=past)
        a2, _ = attn(x[:, 8:9], cos[8:9], sin[8:9], past=(past[0].clone(), past[1].clone()))
    assert torch.equal(a, a2)
    assert past[0].shape[1] == 8 and pa[0].shape[1] == 9 and pb[0].shape[1] == 9
    assert not torch.equal(pa[0][:, 8], pb[0][:, 8])


def test_cached_forward_with_grad_enabled_still_works():
    model = make_model()
    model.train()
    ids = torch.randint(2, 500, (1, 8))
    out = model(ids[:, :5], use_cache=True, num_loops=2)
    out2 = model(ids[:, 5:6], past=out.past)
    out2.logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


# ---------------------------------------------------------------- 2. latent QK-norm

def qk_attn(mode, dtype=torch.float32):
    torch.manual_seed(1)
    cfg = tiny_config(qk_norm=True, qk_norm_mode=mode)
    attn = MultiLatentAttention(cfg).to(dtype).eval()
    for p in attn.parameters():
        p.data.normal_(0.0, 0.3)
    return cfg, attn


def test_latent_mode_has_no_key_nope_norm_and_head_mode_does():
    _, latent = qk_attn("latent")
    _, head = qk_attn("head")
    names = {n for n, _ in latent.named_parameters()}
    assert "k_nope_norm.weight" not in names
    assert {"q_nope_norm.weight", "q_pe_norm.weight", "k_pe_norm.weight"} <= names
    assert "k_nope_norm.weight" in {n for n, _ in head.named_parameters()}


def test_latent_mode_absorbed_path_equals_full_path_exactly():
    cfg, attn = qk_attn("latent")
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, 64, cfg.rope_theta)
    cos, sin = rope(0, 17)
    x = torch.randn(2, 17, cfg.hidden_size)
    with torch.no_grad():
        full, _ = attn(x, cos, sin)                                  # SDPA over up-projected keys
        _, past = attn(x[:, :16], cos[:16], sin[:16])
        step, _ = attn(x[:, 16:], cos[16:17], sin[16:17], past=past)  # absorbed decode (S == 1)
    assert torch.allclose(full[:, 16:], step, atol=1e-6, rtol=1e-5), float((full[:, 16:] - step).abs().max())


def test_latent_mode_cached_decode_matches_full_forward_in_the_model():
    model = make_model(qk_norm=True, qk_norm_mode="latent")
    ids = torch.randint(2, 500, (2, 30))
    with torch.no_grad():
        full = model(ids, num_loops=3).logits
    incremental, _ = decode_logits(model, ids, prefix=6)
    assert torch.allclose(full, incremental, atol=1e-4), float((full - incremental).abs().max())


def test_latent_mode_bf16_is_close_to_fp32():
    cfg, attn32 = qk_attn("latent")
    attn16 = copy.deepcopy(attn32).bfloat16()
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, 64, cfg.rope_theta)
    cos, sin = rope(0, 21)
    x = torch.randn(1, 21, cfg.hidden_size)
    with torch.no_grad():
        _, p32 = attn32(x[:, :20], cos[:20], sin[:20])
        ref, _ = attn32(x[:, 20:], cos[20:21], sin[20:21], past=p32)
        _, p16 = attn16(x[:, :20].bfloat16(), cos[:20], sin[:20])
        out, _ = attn16(x[:, 20:].bfloat16(), cos[20:21], sin[20:21], past=p16)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), ref, atol=0.25, rtol=0.1), float((out.float() - ref).abs().max())


def test_qk_norm_modes_are_validated_and_default_is_unchanged():
    with pytest.raises(ValueError):
        InkyConfig(qk_norm=True, qk_norm_mode="bogus")
    cfg = InkyConfig()
    assert cfg.qk_norm is False and cfg.qk_norm_mode == "head"
    assert InkyModel(InkyConfig()).count_parameters()["total"] == DEFAULT_PARAM_COUNT


def test_head_mode_still_matches_cached_vs_full():
    model = make_model(qk_norm=True, qk_norm_mode="head")
    ids = torch.randint(2, 500, (1, 24))
    with torch.no_grad():
        full = model(ids, num_loops=3).logits
    incremental, _ = decode_logits(model, ids, prefix=6)
    assert torch.allclose(full, incremental, atol=1e-4)


# ---------------------------------------------------------------- 3. dense baseline and highway ablation

FLAGS = [
    pytest.param(dict(use_moe=False), id="dense"),
    pytest.param(dict(use_highway=False), id="no_highway"),
    pytest.param(dict(use_moe=False, use_highway=False), id="dense_no_highway"),
]


def test_default_parameter_count_is_unchanged():
    model = InkyModel(InkyConfig())
    assert model.count_parameters()["total"] == DEFAULT_PARAM_COUNT
    assert all(isinstance(b.mlp, SharedExpertMoE) for b in model.pre_layers)
    assert model.core.ssm is not None


def test_dense_width_matches_the_moe_active_parameters():
    cfg = InkyConfig()
    assert cfg.dense_ffn_size == (cfg.num_shared_experts + cfg.num_active_routed) * cfg.intermediate_size
    moe = InkyModel(tiny_config()).pre_layers[0].mlp
    dense = InkyModel(tiny_config(use_moe=False)).pre_layers[0].mlp
    assert isinstance(dense, DenseFFN)
    count = lambda m: sum(p.numel() for p in m.parameters())   # noqa: E731
    active = sum(count(e) for e in moe.shared_experts) + moe.top_k * count(moe.routed_experts[0])
    assert count(dense) == active
    assert tiny_config(use_moe=False, dense_intermediate_size=50).dense_ffn_size == 50
    with pytest.raises(ValueError):
        InkyConfig(dense_intermediate_size=0)


def test_ablation_parameter_counts_reflect_the_switches():
    base = InkyModel(tiny_config()).count_parameters()["total"]
    no_hw = InkyModel(tiny_config(use_highway=False))
    assert no_hw.core.ssm is None
    assert not any(".ssm." in n or n.startswith("core.ssm") for n, _ in no_hw.named_parameters())
    assert no_hw.count_parameters()["total"] < base
    dense = InkyModel(tiny_config(use_moe=False))
    assert not any("router" in n or "routed_experts" in n for n, _ in dense.named_parameters())
    both = InkyModel(tiny_config(use_moe=False, use_highway=False))
    assert both.count_parameters()["total"] < min(no_hw.count_parameters()["total"], dense.count_parameters()["total"])


@pytest.mark.parametrize("flags", FLAGS)
def test_forward_loss_backward(flags):
    torch.manual_seed(0)
    model = InkyModel(tiny_config(**flags)).train()
    ids = torch.randint(2, 500, (2, 12))
    out = model(ids, labels=ids, num_loops=3)
    assert torch.isfinite(out.loss) and torch.isfinite(out.moe_aux_loss)
    if flags.get("use_moe") is False:
        assert float(out.moe_aux_loss) == 0.0
    out.loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, no_grad
    model.update_moe_balance()                      # a no-op without MoE, must not fail either way
    assert set(out.loss_parts) >= {"ce", "step_ce", "z_loss", "moe_aux", "ponder_kl"}


@pytest.mark.parametrize("flags", FLAGS)
def test_cached_decode_matches_full_forward(flags):
    model = make_model(**flags)
    ids = torch.randint(2, 500, (2, 26))
    with torch.no_grad():
        full = model(ids, num_loops=3).logits
    incremental, _ = decode_logits(model, ids, prefix=6)
    assert torch.allclose(full, incremental, atol=1e-4), float((full - incremental).abs().max())


@pytest.mark.parametrize("flags", FLAGS)
def test_early_exit_and_generate(flags):
    model = make_model(**flags)
    ids = torch.randint(2, 500, (1, 10))
    with torch.no_grad():
        full = model(ids, return_hidden=True, num_loops=3).hidden
        early = model.forward_early_exit(ids, halt_threshold=None, num_loops=3, return_hidden=True).hidden
        exited = model.forward_early_exit(ids, halt_threshold=0.5, num_loops=3)
    assert torch.allclose(full, early, atol=1e-4), float((full - early).abs().max())
    assert exited.loop_applications <= exited.full_loop_applications
    seq = model.generate(ids[:, :4], max_new_tokens=6, do_sample=False, num_loops=3, eos_token_id=None)
    assert seq.shape == (1, 10)


@pytest.mark.parametrize("flags", FLAGS)
def test_save_and_load_roundtrip(flags, tmp_path):
    model = make_model(**flags)
    ids = torch.randint(2, 500, (1, 9))
    model.save_pretrained(tmp_path)
    loaded = InkyModel.from_pretrained(tmp_path).eval()
    for key, value in flags.items():
        assert getattr(loaded.config, key) == value
    assert loaded.count_parameters() == model.count_parameters()
    with torch.no_grad():
        assert torch.allclose(model(ids, num_loops=3).logits, loaded(ids, num_loops=3).logits, atol=1e-6)


@pytest.mark.parametrize("flags", FLAGS)
def test_packing_masks_work_without_moe_or_highway(flags):
    model = make_model(**flags)
    ids = torch.randint(2, 500, (2, 10))
    keep = torch.ones(2, 10, dtype=torch.bool)
    keep[1, :3] = False
    seg = torch.tensor([[0] * 5 + [1] * 5] * 2)
    with torch.no_grad():
        assert torch.isfinite(model(ids, attention_mask=keep, num_loops=2).logits).all()
        assert torch.isfinite(model(ids, segment_ids=seg, num_loops=2).logits).all()
