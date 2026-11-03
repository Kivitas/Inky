"""Equivalence and behaviour tests for the layer fixes in inky/layers.py.

Each optimised path is checked against a copy of the previous implementation kept below as a reference.
Run with: python -m pytest tests/test_layers_fixes.py -q -p no:cacheprovider
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig  # noqa: E402
from inky.layers import (  # noqa: E402
    InkyBlock,
    MultiLatentAttention,
    RecurrentCore,
    RotaryEmbedding,
    SelectiveSSMMixer,
    SharedExpertMoE,
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


def rope(config: InkyConfig):
    emb = RotaryEmbedding(config.qk_rope_head_dim, config.max_position_embeddings, 10000.0)
    return emb(0, config.max_position_embeddings)


# ---------------------------------------------------------------- reference implementations (old code)

def old_moe_forward(moe: SharedExpertMoE, x: torch.Tensor):
    """The previous MoE forward: one where() and index_add per routed expert."""
    B, S, D = x.shape
    flat = x.reshape(-1, D)
    logits = moe.router(flat)
    scores = torch.sigmoid(logits.float())
    _, idx = torch.topk(scores + moe.balance_bias, moe.top_k, dim=-1)
    weights = scores.gather(-1, idx)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)
    out = torch.zeros_like(flat)
    for expert in moe.shared_experts:
        out = out + expert(flat)
    for e, expert in enumerate(moe.routed_experts):
        token_idx, slot = torch.where(idx == e)
        if token_idx.numel() == 0:
            continue
        y = expert(flat[token_idx]) * weights[token_idx, slot].unsqueeze(-1).to(flat.dtype)
        out = out.index_add(0, token_idx, y)
    z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean() * moe.z_loss_coef
    return out.view(B, S, D), z_loss


def old_forward_subset(attn: MultiLatentAttention, x_act, active, cos, sin, store):
    """The previous subset attention: up-projects the whole latent store through kv_b_proj."""
    _, K, _ = x_act.shape
    H = attn.num_heads
    q = attn.q_b_proj(attn.q_a_norm(attn.q_a_proj(x_act))).view(1, K, H, attn.qk_dim).transpose(1, 2)
    q_nope, q_pe = q.split([attn.nope_dim, attn.rope_dim], dim=-1)
    q_pe = apply_rope(q_pe, cos[active], sin[active])
    c, k_pe = attn.kv_a_proj(x_act).split([attn.kv_rank, attn.rope_dim], dim=-1)
    c = attn.kv_a_norm(c)
    k_pe = apply_rope(k_pe.unsqueeze(1), cos[active], sin[active])
    c_store, kpe_store = store
    c_store[:, active] = c.to(c_store.dtype)
    kpe_store[:, :, active] = k_pe.to(kpe_store.dtype)
    S = c_store.shape[1]
    kv = attn.kv_b_proj(c_store).view(1, S, H, attn.nope_dim + attn.v_dim).transpose(1, 2)
    k_nope, v = kv.split([attn.nope_dim, attn.v_dim], dim=-1)
    k = torch.cat([k_nope, kpe_store.expand(1, H, S, attn.rope_dim)], dim=-1)
    q_full = torch.cat([q_nope, q_pe], dim=-1)
    positions = torch.arange(S)
    mask = positions.unsqueeze(0) <= active.unsqueeze(1)
    out = F.scaled_dot_product_attention(q_full, k, v, attn_mask=mask)
    out = out.transpose(1, 2).reshape(1, K, H * attn.v_dim)
    return attn.o_proj(out)


def loop_ssm_forward(mixer: SelectiveSSMMixer, x, past_state=None):
    """The previous SSM mixer: a Python loop over tokens."""
    B, S, _ = x.shape
    u = mixer.in_proj(x)
    a = torch.sigmoid(mixer.decay_proj(x))
    g = torch.sigmoid(mixer.gate_proj(x))
    h = u.new_zeros(B, u.shape[-1], mixer.state_dim) if past_state is None else past_state
    ys = []
    for t in range(S):
        h = a[:, t, None, :] * h + g[:, t, None, :] * u[:, t, :, None]
        ys.append((h * mixer.readout).sum(-1))
    y = torch.stack(ys, dim=1) + mixer.skip * u
    return mixer.out_proj(y), h


# ---------------------------------------------------------------- 1. expert load accumulation

def test_expert_load_accumulates_over_calls_and_resets_on_update():
    torch.manual_seed(1)
    config = tiny_config(balance_rate=0.1)
    moe = SharedExpertMoE(config).eval()
    moe.balance_bias.normal_(0.0, 0.5)
    x1, x2 = torch.randn(2, 5, 64), torch.randn(3, 4, 64)

    with torch.no_grad():
        moe(x1)
        after_first = moe.load_counts.clone()
        moe(x2)
        total = moe.load_counts.sum()
    # Every (token, slot) assignment is counted: (2*5 + 3*4) tokens times top_k slots.
    assert float(total) == (10 + 12) * config.num_active_routed
    assert torch.all(moe.load_counts >= after_first)
    assert torch.allclose(moe.expert_load, moe.load_counts / total)

    frac = moe.load_counts / total
    bias_before = moe.balance_bias.clone()
    moe.update_balance_bias()
    expected = bias_before + config.balance_rate * (1.0 / config.num_routed_experts - frac)
    assert torch.allclose(moe.balance_bias, expected, atol=1e-7)
    assert float(moe.load_counts.sum()) == 0.0


def test_single_forward_per_update_matches_previous_behaviour():
    # Non-looped use: one forward between updates. The bias step uses that forward's load fraction.
    torch.manual_seed(2)
    config = tiny_config(balance_rate=0.05)
    moe = SharedExpertMoE(config).eval()
    x = torch.randn(1, 9, 64)
    with torch.no_grad():
        _, idx = torch.topk(torch.sigmoid(moe.router(x.reshape(-1, 64)).float()) + moe.balance_bias,
                            moe.top_k, dim=-1)
        counts = torch.bincount(idx.reshape(-1), minlength=config.num_routed_experts).float()
        moe(x)
    bias_before = moe.balance_bias.clone()
    moe.update_balance_bias()
    expected = bias_before + config.balance_rate * (1.0 / config.num_routed_experts - counts / counts.sum())
    assert torch.allclose(moe.balance_bias, expected, atol=1e-7)


def test_update_without_new_load_leaves_bias_unchanged():
    moe = SharedExpertMoE(tiny_config()).eval()
    moe.balance_bias.normal_()
    before = moe.balance_bias.clone()
    moe.update_balance_bias()
    assert torch.equal(moe.balance_bias, before)


def test_reset_expert_load_discards_counts():
    moe = SharedExpertMoE(tiny_config()).eval()
    with torch.no_grad():
        moe(torch.randn(2, 3, 64))
    assert float(moe.load_counts.sum()) > 0
    moe.reset_expert_load()
    assert float(moe.load_counts.sum()) == 0.0


def test_load_counts_are_not_in_the_state_dict():
    moe = SharedExpertMoE(tiny_config())
    assert "load_counts" not in moe.state_dict()
    assert "expert_load" in moe.state_dict()


# ---------------------------------------------------------------- 2. no host sync in loop dropout

def test_loop_dropout_drops_all_layers_at_probability_one_in_training():
    config = tiny_config(loop_dropout_prob=1.0)
    core = RecurrentCore(config).train()
    cos, sin = rope(config)
    h = torch.randn(1, 6, config.hidden_size)
    _, _, _, presents, aux = core(h, 0, cos[:6], sin[:6], None, None, None, 0)
    assert presents == [None] * config.num_recurrent_layers
    assert float(aux) == 0.0


def test_loop_dropout_is_inactive_in_eval_and_at_zero():
    cos, sin = rope(tiny_config())
    h = torch.randn(1, 6, 64)
    for prob, training in [(1.0, False), (0.0, True)]:
        config = tiny_config(loop_dropout_prob=prob)
        core = RecurrentCore(config)
        core.train(training)
        _, _, _, presents, _ = core(h, 0, cos[:6], sin[:6], None, None, None, 0)
        assert all(p is not None for p in presents)


def test_loop_dropout_rate_is_about_the_configured_probability():
    torch.manual_seed(3)
    config = tiny_config(loop_dropout_prob=0.3, num_pre_layers=1)
    core = RecurrentCore(config).train()
    cos, sin = rope(config)
    h = torch.randn(1, 4, 64)
    dropped = total = 0
    for _ in range(300):
        _, _, _, presents, _ = core(h, 0, cos[:4], sin[:4], None, None, None, 0)
        dropped += sum(p is None for p in presents)
        total += len(presents)
    assert 0.25 < dropped / total < 0.35, dropped / total


# ---------------------------------------------------------------- 3. MoE dispatch equivalence

@pytest.mark.parametrize("bias_scale", [0.0, 2.0])
def test_sorted_moe_dispatch_matches_per_expert_loop(bias_scale):
    torch.manual_seed(4)
    config = tiny_config()
    moe = SharedExpertMoE(config).eval()
    moe.balance_bias.normal_(0.0, bias_scale)
    x = torch.randn(3, 7, 64)
    with torch.no_grad():
        new, new_z = moe(x)
        ref, ref_z = old_moe_forward(moe, x)
    assert new.shape == ref.shape == x.shape
    assert torch.allclose(new, ref, atol=1e-5), float((new - ref).abs().max())
    assert torch.allclose(new_z, ref_z)


def test_sorted_moe_dispatch_gradients_match_old_implementation():
    torch.manual_seed(5)
    config = tiny_config()
    moe = SharedExpertMoE(config).train()
    x = torch.randn(2, 6, 64)

    moe.zero_grad()
    new, _ = moe(x)
    new.pow(2).sum().backward()
    new_grads = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}

    moe.zero_grad()
    ref, _ = old_moe_forward(moe, x)
    ref.pow(2).sum().backward()
    ref_grads = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}

    assert new_grads.keys() == ref_grads.keys()
    for name in new_grads:
        assert torch.allclose(new_grads[name], ref_grads[name], atol=1e-4), name


# ---------------------------------------------------------------- 4. absorbed subset decode

def test_absorbed_subset_attention_matches_full_up_projection():
    torch.manual_seed(6)
    config = tiny_config()
    attn = MultiLatentAttention(config).eval()
    cos, sin = rope(config)
    S = 12
    active = torch.tensor([2, 5, 11])
    x_act = torch.randn(1, len(active), 64)
    c_store = torch.randn(1, S, config.kv_lora_rank)
    kpe_store = torch.randn(1, 1, S, config.qk_rope_head_dim)

    store_new = (c_store.clone(), kpe_store.clone())
    store_old = (c_store.clone(), kpe_store.clone())
    with torch.no_grad():
        new = attn.forward_subset(x_act, active, cos, sin, store_new)
        old = old_forward_subset(attn, x_act, active, cos, sin, store_old)
    assert torch.allclose(new, old, atol=1e-5), float((new - old).abs().max())
    assert torch.equal(store_new[0], store_old[0])
    assert torch.equal(store_new[1], store_old[1])


def test_absorbed_decode_single_token_matches_full_attention():
    torch.manual_seed(7)
    config = tiny_config()
    attn = MultiLatentAttention(config).eval()
    cos, sin = rope(config)
    x = torch.randn(2, 9, 64)
    with torch.no_grad():
        full, _ = attn(x, cos[:9], sin[:9])
        # The last position decoded against a cache of the first 8 positions.
        _, past = attn(x[:, :8], cos[:8], sin[:8])
        step, _ = attn(x[:, 8:], cos[8:9], sin[8:9], past=past)
    assert torch.allclose(full[:, 8:], step, atol=1e-5)


# ---------------------------------------------------------------- 5. chunked SSM scan

def test_chunked_ssm_scan_matches_token_loop():
    torch.manual_seed(8)
    config = tiny_config()
    mixer = SelectiveSSMMixer(config).double().eval()
    x = torch.randn(2, 150, 64, dtype=torch.float64)   # spans several 64-token chunks
    with torch.no_grad():
        out, (h_new, _) = mixer(x)
        ref, h_ref = loop_ssm_forward(mixer, x)
    assert torch.allclose(out, ref, atol=1e-10), float((out - ref).abs().max())
    assert torch.allclose(h_new, h_ref, atol=1e-10)


def test_chunked_ssm_scan_small_chunks_and_carried_state_match_token_loop():
    torch.manual_seed(9)
    config = tiny_config()
    mixer = SelectiveSSMMixer(config).double().eval()
    mixer.scan_chunk = 7
    x = torch.randn(1, 40, 64, dtype=torch.float64)
    with torch.no_grad():
        ref, _ = loop_ssm_forward(mixer, x)
        whole, (h_whole, _) = mixer(x)
        first, (h_mid, _) = mixer(x[:, :23])
        second, (h_end, _) = mixer(x[:, 23:], past=(h_mid, None))
    assert torch.allclose(whole, ref, atol=1e-10)
    assert torch.allclose(torch.cat([first, second], dim=1), whole, atol=1e-10)
    assert torch.allclose(h_end, h_whole, atol=1e-10)


def test_chunked_ssm_scan_is_finite_on_a_long_sequence():
    torch.manual_seed(10)
    mixer = SelectiveSSMMixer(tiny_config()).eval()
    x = torch.randn(1, 1024, 64) * 3.0
    with torch.no_grad():
        out, (h, _) = mixer(x)
    assert torch.isfinite(out).all() and torch.isfinite(h).all()


# ---------------------------------------------------------------- 6. attention masks and segments

def test_segment_masked_concatenation_matches_separate_documents():
    torch.manual_seed(11)
    config = tiny_config()
    block = InkyBlock(config).eval()
    cos, sin = rope(config)
    doc_a, doc_b = torch.randn(1, 5, 64), torch.randn(1, 7, 64)
    packed = torch.cat([doc_a, doc_b], dim=1)
    segments = torch.tensor([[0] * 5 + [1] * 7])
    with torch.no_grad():
        out_packed, _, _ = block(packed, cos[:12], sin[:12], segment_ids=segments)
        out_a, _, _ = block(doc_a, cos[:5], sin[:5])
        out_b, _, _ = block(doc_b, cos[5:12], sin[5:12])          # same absolute positions as in the pack
    assert torch.allclose(out_packed[:, :5], out_a, atol=1e-5)
    assert torch.allclose(out_packed[:, 5:], out_b, atol=1e-5)


def test_segment_mask_blocks_cross_document_influence():
    torch.manual_seed(12)
    config = tiny_config()
    attn = MultiLatentAttention(config).eval()
    cos, sin = rope(config)
    x = torch.randn(1, 10, 64)
    segments = torch.tensor([[0] * 4 + [1] * 6])
    changed = x.clone()
    changed[:, :4] = torch.randn(1, 4, 64)                        # rewrite document 0 only
    with torch.no_grad():
        a, _ = attn(x, cos[:10], sin[:10], segment_ids=segments)
        b, _ = attn(changed, cos[:10], sin[:10], segment_ids=segments)
    assert torch.allclose(a[:, 4:], b[:, 4:], atol=1e-5)
    assert not torch.allclose(a[:, :4], b[:, :4], atol=1e-5)


def test_all_true_attn_mask_matches_default():
    torch.manual_seed(13)
    config = tiny_config()
    attn = MultiLatentAttention(config).eval()
    cos, sin = rope(config)
    x = torch.randn(2, 8, 64)
    ones = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    with torch.no_grad():
        default, _ = attn(x, cos[:8], sin[:8])
        masked, _ = attn(x, cos[:8], sin[:8], attn_mask=ones)
        masked3d, _ = attn(x, cos[:8], sin[:8], attn_mask=ones[:, 0])   # [B, S, T] is accepted too
    assert torch.allclose(default, masked, atol=1e-5)
    assert torch.allclose(default, masked3d, atol=1e-5)


def test_attn_mask_with_past_hides_the_masked_cache_entries():
    torch.manual_seed(14)
    config = tiny_config()
    attn = MultiLatentAttention(config).eval()
    cos, sin = rope(config)
    x = torch.randn(1, 9, 64)
    with torch.no_grad():
        _, past = attn(x[:, :8], cos[:8], sin[:8])
        # Key 0 is hidden from the single new query; the output must change compared with no mask.
        mask = torch.ones(1, 1, 1, 9, dtype=torch.bool)
        mask[..., 0] = False
        plain, _ = attn(x[:, 8:], cos[8:9], sin[8:9], past=past)
        hidden, _ = attn(x[:, 8:], cos[8:9], sin[8:9], past=past, attn_mask=mask)
    assert not torch.allclose(plain, hidden, atol=1e-5)


def test_segment_ids_with_past_is_rejected():
    config = tiny_config()
    attn = MultiLatentAttention(config).eval()
    cos, sin = rope(config)
    x = torch.randn(1, 3, 64)
    _, past = attn(x, cos[:3], sin[:3])
    with pytest.raises(ValueError):
        attn(x[:, :1], cos[3:4], sin[3:4], past=past, segment_ids=torch.zeros(1, 1, dtype=torch.long))


def test_segment_ids_shape_is_checked():
    config = tiny_config()
    attn = MultiLatentAttention(config).eval()
    cos, sin = rope(config)
    with pytest.raises(ValueError):
        attn(torch.randn(1, 4, 64), cos[:4], sin[:4], segment_ids=torch.zeros(1, 3, dtype=torch.long))


def test_ssm_mixer_rejects_masks_it_cannot_honour():
    mixer = SelectiveSSMMixer(tiny_config()).eval()
    x = torch.randn(1, 4, 64)
    with pytest.raises(NotImplementedError, match="ssm_mixer_layers"):
        mixer(x, segment_ids=torch.zeros(1, 4, dtype=torch.long))


# ---------------------------------------------------------------- 7. packed documents in the highway and core

def naive_segment_mean(h: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """Reference: for each position, the mean of h over earlier-or-equal positions with the same id."""
    B, S, _ = h.shape
    out = torch.zeros_like(h)
    for b in range(B):
        for t in range(S):
            same = [s for s in range(t + 1) if seg[b, s] == seg[b, t]]
            out[b, t] = h[b, same].mean(dim=0)
    return out


def test_segment_mean_matches_naive_reference_on_several_segments():
    torch.manual_seed(15)
    h = torch.randn(2, 9, 8, dtype=torch.float64)
    seg = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2],
                        [0, 1, 1, 1, 2, 2, 2, 2, 2]])
    fast = StateSpaceMemoryHighway._segment_mean(h, seg)
    assert torch.allclose(fast, naive_segment_mean(h, seg), atol=1e-12)


def test_highway_packed_documents_match_separate_runs():
    torch.manual_seed(16)
    config = tiny_config()
    ssm = StateSpaceMemoryHighway(config.hidden_size, config.ssm_state_dim, config.rms_norm_eps).eval()
    doc_a, doc_b = torch.randn(1, 5, 64), torch.randn(1, 7, 64)
    packed = torch.cat([doc_a, doc_b], dim=1)
    seg = torch.tensor([[0] * 5 + [1] * 7])
    with torch.no_grad():
        out_p, mem_p, prefix_p = ssm(packed, None, None, 0, segment_ids=seg)
        out_a, mem_a, _ = ssm(doc_a, None, None, 0)
        out_b, mem_b, _ = ssm(doc_b, None, None, 0)
    assert prefix_p is None
    assert torch.allclose(out_p[:, :5], out_a, atol=1e-5)
    assert torch.allclose(out_p[:, 5:], out_b, atol=1e-5)
    assert torch.allclose(mem_p[:, :5], mem_a, atol=1e-5)
    assert torch.allclose(mem_p[:, 5:], mem_b, atol=1e-5)


def test_highway_segments_reject_carried_prefix_and_cache():
    config = tiny_config()
    ssm = StateSpaceMemoryHighway(config.hidden_size, config.ssm_state_dim, config.rms_norm_eps).eval()
    h = torch.randn(1, 4, 64)
    seg = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        ssm(h, None, torch.zeros(1, 1, 64), 0, segment_ids=seg)
    with pytest.raises(ValueError):
        ssm(h, None, None, 3, segment_ids=seg)


def test_single_segment_matches_default_path():
    torch.manual_seed(17)
    config = tiny_config()
    core = RecurrentCore(config).eval()
    cos, sin = rope(config)
    h = torch.randn(2, 6, 64)
    one = torch.zeros(2, 6, dtype=torch.long)
    with torch.no_grad():
        default = core(h, 1, cos[:6], sin[:6])
        segmented = core(h, 1, cos[:6], sin[:6], segment_ids=one, attn_mask=None)
    assert torch.allclose(default[0], segmented[0], atol=1e-5)
    assert torch.allclose(default[1], segmented[1], atol=1e-5)


def test_recurrent_core_packed_documents_match_separate_over_loops():
    torch.manual_seed(18)
    config = tiny_config(num_recurrent_layers=2, max_recurrent_loops=4, ssm_mixer_layers=())
    core = RecurrentCore(config).eval()
    cos, sin = rope(config)
    len_a, len_b, loops = 5, 7, 3
    doc_a, doc_b = torch.randn(1, len_a, 64), torch.randn(1, len_b, 64)
    packed = torch.cat([doc_a, doc_b], dim=1)
    seg = torch.tensor([[0] * len_a + [1] * len_b])

    def run(h, seg_ids, cos_s, sin_s):
        # Same driver as the model's loop: each loop gets the previous loop's state and highway memory.
        mem = None
        for step in range(1, loops + 1):
            h, mem, _, _, _ = core(h, step, cos_s, sin_s, None, None, mem, 0, segment_ids=seg_ids)
        return h, mem

    with torch.no_grad():
        out_p, mem_p = run(packed, seg, cos[:len_a + len_b], sin[:len_a + len_b])
        out_a, mem_a = run(doc_a, None, cos[:len_a], sin[:len_a])
        out_b, mem_b = run(doc_b, None, cos[:len_b], sin[:len_b])
    assert torch.allclose(out_p[:, :len_a], out_a, atol=1e-5), float((out_p[:, :len_a] - out_a).abs().max())
    assert torch.allclose(out_p[:, len_a:], out_b, atol=1e-5), float((out_p[:, len_a:] - out_b).abs().max())
    assert torch.allclose(mem_p[:, :len_a], mem_a, atol=1e-5)
    assert torch.allclose(mem_p[:, len_a:], mem_b, atol=1e-5)


def test_recurrent_core_all_true_attn_mask_matches_default():
    torch.manual_seed(19)
    config = tiny_config(num_recurrent_layers=2, ssm_mixer_layers=())
    core = RecurrentCore(config).eval()
    cos, sin = rope(config)
    h = torch.randn(2, 6, 64)
    ones = torch.ones(2, 6, 6, dtype=torch.bool)
    with torch.no_grad():
        default = core(h, 2, cos[:6], sin[:6])
        masked = core(h, 2, cos[:6], sin[:6], attn_mask=ones)
    assert torch.allclose(default[0], masked[0], atol=1e-5)


def test_recurrent_core_rejects_packing_with_ssm_layers():
    config = tiny_config(num_recurrent_layers=2, ssm_mixer_layers=(1,))
    core = RecurrentCore(config).eval()
    cos, sin = rope(config)
    h = torch.randn(1, 4, 64)
    with pytest.raises(NotImplementedError, match="ssm_mixer_layers"):
        core(h, 1, cos[:4], sin[:4], segment_ids=torch.zeros(1, 4, dtype=torch.long))


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        if test.__code__.co_argcount == 0:
            test()
            print(f"ok  {test.__name__}")
    print("done")
