"""Tests for the open core findings CORE-06, CORE-07, CORE-10 and CORE-14 (inky/layers.py).

Run with: python -m pytest tests/test_layers_open_fixes.py -q -p no:cacheprovider
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig  # noqa: E402
from inky.layers import (  # noqa: E402
    MultiLatentAttention,
    RecurrentCore,
    RotaryEmbedding,
    StateSpaceMemoryHighway,
    apply_rope,
)


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512,
        scratch_open_id=508,
        scratch_sep_id=509,
        scratch_close_id=510,
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
        eos_token_id=1,
        pad_token_id=1,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def rope_tables(config: InkyConfig, length: int):
    return RotaryEmbedding(config.qk_rope_head_dim, config.max_position_embeddings, config.rope_theta)(0, length)


def rotate_half_reference(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------- CORE-06: RoPE and cache dtypes

def test_apply_rope_fp32_is_the_plain_formula():
    torch.manual_seed(0)
    config = tiny_config()
    cos, sin = rope_tables(config, 5)
    x = torch.randn(1, 2, 5, config.qk_rope_head_dim)
    ref = x * cos + rotate_half_reference(x) * sin
    assert torch.equal(apply_rope(x, cos, sin), ref)


def test_apply_rope_bf16_input_returns_bf16_within_one_rounding_of_fp32():
    torch.manual_seed(1)
    config = tiny_config()
    cos, sin = rope_tables(config, 16)
    x = torch.randn(1, 2, 16, config.qk_rope_head_dim).bfloat16()
    out = apply_rope(x, cos, sin)
    assert out.dtype == torch.bfloat16
    # Same inputs in fp32: the only error left is the final cast to bf16 (half an ulp, 2**-9 relative).
    ref = apply_rope(x.float(), cos, sin)
    assert bool(((out.float() - ref).abs() <= 2.0 ** -8 * ref.abs() + 1e-12).all())


def test_rope_tables_stay_fp32_when_the_module_is_cast_to_bfloat16():
    config = tiny_config()
    emb = RotaryEmbedding(config.qk_rope_head_dim, config.max_position_embeddings, config.rope_theta)
    cos_before, sin_before = emb.cos.clone(), emb.sin.clone()
    emb.bfloat16()
    assert emb.cos.dtype == torch.float32 and emb.sin.dtype == torch.float32
    assert torch.equal(emb.cos, cos_before) and torch.equal(emb.sin, sin_before)
    cos, _ = emb(0, 8)
    assert cos.dtype == torch.float32


def test_bf16_weights_with_fp32_tables_prefill_and_decode_run_and_agree():
    torch.manual_seed(2)
    config = tiny_config()
    attn = MultiLatentAttention(config).bfloat16().eval()   # tables below stay fp32: a partial cast
    x = torch.randn(1, 9, config.hidden_size).bfloat16()
    cos, sin = rope_tables(config, 9)
    with torch.no_grad():
        full, full_cache = attn(x, cos, sin)
        _, past = attn(x[:, :8], cos[:8], sin[:8])
        step, step_cache = attn(x[:, 8:], cos[8:9], sin[8:9], past=past)
    assert full.dtype == torch.bfloat16 and step.dtype == torch.bfloat16
    assert all(t.dtype == torch.bfloat16 for t in full_cache + step_cache)
    # Prefill uses SDPA, decode the absorbed path; both in bf16, so they agree to bf16 rounding.
    scale = full.float().abs().max().item()
    assert (step.float() - full[:, 8:].float()).abs().max().item() <= 0.05 * scale


def test_autocast_keeps_the_latent_cache_in_the_activation_dtype():
    torch.manual_seed(3)
    config = tiny_config()
    attn = MultiLatentAttention(config).eval()    # fp32 weights, as in training under autocast
    x = torch.randn(1, 6, config.hidden_size)
    cos, sin = rope_tables(config, 6)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, (c_kv, k_pe) = attn(x, cos, sin)
    assert c_kv.dtype == torch.bfloat16
    assert k_pe.dtype == torch.bfloat16


# ---------------------------------------------------------------- CORE-07: absorbed decode precision

def test_absorbed_attention_bf16_error_is_only_the_output_rounding():
    torch.manual_seed(4)
    config = tiny_config(hidden_size=384, num_attention_heads=6, qk_nope_head_dim=48, qk_rope_head_dim=16,
                         v_head_dim=64, q_lora_rank=192, kv_lora_rank=128, max_position_embeddings=1024)
    attn_b = MultiLatentAttention(config).bfloat16().eval()
    attn_32 = MultiLatentAttention(config).eval()
    attn_32.load_state_dict({k: v.float() for k, v in attn_b.state_dict().items()})   # same bf16 values, exact
    T, H = 1500, config.num_attention_heads
    q_nope = (torch.randn(1, H, 1, config.qk_nope_head_dim) * 2).bfloat16()
    q_pe = (torch.randn(1, H, 1, config.qk_rope_head_dim) * 2).bfloat16()
    c_kv = (torch.randn(1, T, config.kv_lora_rank) * 2).bfloat16()
    k_pe = (torch.randn(1, 1, T, config.qk_rope_head_dim) * 2).bfloat16()
    with torch.no_grad():
        out_b = attn_b._absorbed_attention(q_nope, q_pe, c_kv, k_pe)
        ref = attn_32._absorbed_attention(q_nope.float(), q_pe.float(), c_kv.float(), k_pe.float())
    assert out_b.dtype == torch.bfloat16
    err = (out_b.float() - ref).abs()
    assert bool((err <= 2.0 ** -8 * ref.abs().max()).all()), float(err.max())


def test_bf16_decode_step_matches_fp32_decode_step_closely():
    torch.manual_seed(5)
    config = tiny_config()
    attn_b = MultiLatentAttention(config).bfloat16().eval()
    attn_32 = MultiLatentAttention(config).eval()
    attn_32.load_state_dict({k: v.float() for k, v in attn_b.state_dict().items()})
    x = torch.randn(1, 20, config.hidden_size)
    cos, sin = rope_tables(config, 20)
    with torch.no_grad():
        _, past_b = attn_b(x[:, :19].bfloat16(), cos[:19], sin[:19])
        out_b, _ = attn_b(x[:, 19:].bfloat16(), cos[19:20], sin[19:20], past=past_b)
        _, past_32 = attn_32(x[:, :19], cos[:19], sin[:19])
        out_32, _ = attn_32(x[:, 19:], cos[19:20], sin[19:20], past=past_32)
    scale = out_32.abs().max().item()
    assert (out_b.float() - out_32).abs().max().item() <= 0.03 * scale


# ---------------------------------------------------------------- CORE-10: valid mask in the highway

def highway(config: InkyConfig) -> StateSpaceMemoryHighway:
    torch.manual_seed(6)
    return StateSpaceMemoryHighway(config.hidden_size, config.ssm_state_dim, config.rms_norm_eps).eval()


def test_highway_all_true_valid_mask_matches_default():
    config = tiny_config()
    hw = highway(config)
    h = torch.randn(2, 11, config.hidden_size)
    with torch.no_grad():
        out_def, mem_def, _ = hw(h, None, None, 0)
        out_m, mem_m, prefix = hw(h, None, None, 0, valid_mask=torch.ones(2, 11, dtype=torch.bool))
    assert torch.allclose(out_m, out_def, atol=1e-6)
    assert torch.allclose(mem_m, mem_def, atol=1e-6)
    assert prefix is None


def test_highway_left_padding_is_excluded_from_the_running_mean():
    config = tiny_config()
    hw = highway(config)
    D = config.hidden_size
    h_valid = torch.randn(1, 5, D)
    pads = torch.randn(1, 3, D) * 1e3                       # large garbage: would dominate an unmasked mean
    h_pad = torch.cat([pads, h_valid], dim=1)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        out_ref, mem_ref, _ = hw(h_valid, None, None, 0)
        out_pad, mem_pad, _ = hw(h_pad, None, None, 0, valid_mask=mask)
        out_unmasked, _, _ = hw(h_pad, None, None, 0)
    assert torch.allclose(out_pad[:, 3:], out_ref, atol=1e-5)
    assert torch.allclose(mem_pad[:, 3:], mem_ref, atol=1e-5)
    # Sanity: without the mask the pads do leak into the mean, so the test above is meaningful.
    assert not torch.allclose(out_unmasked[:, 3:], out_ref, atol=1e-3)


def test_highway_right_padding_contents_do_not_change_valid_outputs():
    config = tiny_config()
    hw = highway(config)
    D = config.hidden_size
    h_valid = torch.randn(1, 4, D)
    with torch.no_grad():
        out_ref, _, _ = hw(h_valid, None, None, 0)
        for junk in (torch.randn(1, 3, D), torch.randn(1, 3, D) * 1e4):
            h_pad = torch.cat([h_valid, junk], dim=1)
            mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
            out_pad, _, _ = hw(h_pad, None, None, 0, valid_mask=mask)
            assert torch.allclose(out_pad[:, :4], out_ref, atol=1e-5)


def test_highway_valid_mask_combined_with_segments_ignores_pads_inside_a_document():
    config = tiny_config()
    hw = highway(config)
    D = config.hidden_size
    doc_a = torch.randn(1, 3, D)
    doc_b = torch.randn(1, 4, D)
    pads = torch.randn(1, 2, D) * 1e3                       # pads sit inside document 0
    h_pad = torch.cat([doc_a, pads, doc_b], dim=1)          # length 9
    seg_pad = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    seg_ref = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    h_ref = torch.cat([doc_a, doc_b], dim=1)
    with torch.no_grad():
        out_ref, _, _ = hw(h_ref, None, None, 0, segment_ids=seg_ref)
        out_pad, _, prefix = hw(h_pad, None, None, 0, segment_ids=seg_pad, valid_mask=mask)
    keep = mask[0]
    assert torch.allclose(out_pad[0, keep], out_ref[0], atol=1e-5)
    assert prefix is None


def test_highway_valid_mask_rejects_cache_and_bad_shapes():
    config = tiny_config()
    hw = highway(config)
    D = config.hidden_size
    h = torch.randn(1, 5, D)
    with pytest.raises(ValueError):
        hw(h, None, None, 2, valid_mask=torch.ones(1, 5, dtype=torch.bool))
    with pytest.raises(ValueError):
        hw(h, None, torch.zeros(1, 1, D), 0, valid_mask=torch.ones(1, 5, dtype=torch.bool))
    with pytest.raises(ValueError):
        hw(h, None, None, 0, valid_mask=torch.ones(1, 4, dtype=torch.bool))


def test_recurrent_core_left_padded_rows_match_their_unpadded_runs_over_loops():
    torch.manual_seed(7)
    config = tiny_config()
    core = RecurrentCore(config).eval()
    D = config.hidden_size
    # Row 0 has no padding. Row 1 has 2 left pads followed by 4 real tokens.
    row0 = torch.randn(1, 6, D)
    row1_valid = torch.randn(1, 4, D)
    row1 = torch.cat([torch.randn(1, 2, D) * 50.0, row1_valid], dim=1)
    h = torch.cat([row0, row1], dim=0)                      # [2, 6, D]
    valid = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    eye = torch.eye(6, dtype=torch.bool).unsqueeze(0)
    attn_mask = (valid.unsqueeze(1) | eye)                  # keys must be valid; pad queries see themselves
    cos, sin = rope_tables(config, 6)

    def run(x, mask, attn, cos_, sin_, loops=3):
        mem, out = None, x
        with torch.no_grad():
            for step in range(1, loops + 1):
                out, mem, _, _, _ = core(out, step, cos_, sin_, prev_mem=mem,
                                         attn_mask=attn, valid_mask=mask)
        return out

    with torch.no_grad():
        batched = run(h, valid, attn_mask, cos, sin)
        ref0 = run(row0, None, None, cos, sin)
        ref1 = run(row1_valid, None, None, cos[:4], sin[:4])
    assert torch.allclose(batched[0], ref0[0], atol=1e-4)
    assert torch.allclose(batched[1, 2:], ref1[0], atol=1e-4)


def test_recurrent_core_all_true_valid_mask_matches_default():
    torch.manual_seed(8)
    config = tiny_config()
    core = RecurrentCore(config).eval()
    h = torch.randn(2, 7, config.hidden_size)
    cos, sin = rope_tables(config, 7)
    with torch.no_grad():
        out_def, mem_def, _, _, _ = core(h, 1, cos, sin)
        out_m, mem_m, _, _, _ = core(h, 1, cos, sin, valid_mask=torch.ones(2, 7, dtype=torch.bool))
    assert torch.allclose(out_m, out_def, atol=1e-6)
    assert torch.allclose(mem_m, mem_def, atol=1e-6)


# ---------------------------------------------------------------- CORE-14: highway normalisation (design check)

def test_highway_added_term_has_fixed_rms_by_design_regardless_of_input_scale():
    """CORE-14 is NOT-A-BUG: the RMSNorm sets the magnitude of the highway's added term to 0.1 * weight RMS
    by design (the default output is unchanged). This test pins that property so a change is deliberate."""
    config = tiny_config()
    hw = highway(config)
    for scale in (1.0, 30.0):
        h = torch.randn(1, 40, config.hidden_size) * scale
        with torch.no_grad():
            out, _, _ = hw(h, None, None, 0)
        added_rms = (out - h).pow(2).mean(-1).sqrt()
        assert torch.allclose(added_rms, torch.full_like(added_rms, 0.1), rtol=1e-3)


def test_highway_norm_stays_finite_when_the_input_is_zero():
    config = tiny_config()
    hw = highway(config)
    h = torch.zeros(1, 6, config.hidden_size)
    with torch.no_grad():
        out, mem, _ = hw(h, None, None, 0)
    assert torch.isfinite(out).all() and torch.isfinite(mem).all()
    assert torch.equal(out, h)
