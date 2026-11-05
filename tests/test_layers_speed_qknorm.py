"""Tests for the MoE dispatch speed-up and the opt-in QK-norm (inky/layers.py, inky/config.py).

Run with: python -m pytest tests/test_layers_speed_qknorm.py -q -p no:cacheprovider
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig  # noqa: E402
from inky.layers import MultiLatentAttention, RotaryEmbedding, SharedExpertMoE  # noqa: E402

DEFAULT_PARAM_COUNT = 50_219_202


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        highway_gain_init=0.1,  # keeps the highway active in these checks (the default gain is 0)
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
        num_routed_experts=8,
        num_active_routed=2,
        num_shared_experts=1,
        ssm_state_dim=16,
        max_position_embeddings=64,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def naive_moe(moe: SharedExpertMoE, x: torch.Tensor) -> torch.Tensor:
    """Reference: loop over tokens and selected experts, no grouping."""
    B, S, D = x.shape
    flat = x.reshape(-1, D)
    scores = torch.sigmoid(moe.router(flat).float())
    _, idx = torch.topk(scores + moe.balance_bias, moe.top_k, dim=-1)
    weights = scores.gather(-1, idx)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)
    out = torch.zeros_like(flat)
    for expert in moe.shared_experts:
        out = out + expert(flat)
    rows = []
    for t in range(flat.shape[0]):
        acc = out[t].clone()
        for s in range(moe.top_k):
            e = int(idx[t, s])
            acc = acc + weights[t, s] * moe.routed_experts[e](flat[t:t + 1])[0]
        rows.append(acc)
    return torch.stack(rows).view(B, S, D)


def test_moe_output_matches_naive_reference():
    torch.manual_seed(0)
    cfg = tiny_config()
    moe = SharedExpertMoE(cfg).eval()
    # Non-zero balance bias so that the biased top-k selection is exercised.
    moe.balance_bias.copy_(torch.linspace(-0.3, 0.3, cfg.num_routed_experts))
    x = torch.randn(3, 5, cfg.hidden_size)
    with torch.no_grad():
        out, _ = moe(x)
        ref = naive_moe(moe, x)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-6, rtol=0), (out - ref).abs().max()


def test_dense_dispatch_matches_grouped_dispatch():
    """The sync-free accelerator path must give the same routed output as the CPU grouped path."""
    torch.manual_seed(1)
    cfg = tiny_config()
    moe = SharedExpertMoE(cfg).eval()
    x = torch.randn(4, 7, cfg.hidden_size)
    flat = x.reshape(-1, cfg.hidden_size)
    with torch.no_grad():
        scores = torch.sigmoid(moe.router(flat).float())
        _, idx = torch.topk(scores + moe.balance_bias, moe.top_k, dim=-1)
        weights = scores.gather(-1, idx)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)
        base = torch.randn_like(flat)
        grouped = moe._dispatch_routed(flat, idx, weights, base.clone())
        dense = moe._dispatch_routed_dense(flat, idx, weights, base.clone())
    assert torch.allclose(grouped, dense, atol=1e-6, rtol=0), (grouped - dense).abs().max()


def _route(moe, flat):
    scores = torch.sigmoid(moe.router(flat).float())
    _, idx = torch.topk(scores + moe.balance_bias, moe.top_k, dim=-1)
    weights = scores.gather(-1, idx)
    return idx, weights / (weights.sum(-1, keepdim=True) + 1e-9)


def test_use_dense_only_for_small_n_on_non_cpu():
    moe = SharedExpertMoE(tiny_config())
    assert moe.dense_max_tokens == 64
    assert moe._use_dense(1, "cuda") and moe._use_dense(64, "cuda") and moe._use_dense(8, "mps")
    assert not moe._use_dense(65, "cuda") and not moe._use_dense(4096, "cuda")
    for n in (1, 8, 64, 65, 4096):
        assert not moe._use_dense(n, "cpu")
    off = SharedExpertMoE(tiny_config(moe_dense_max_tokens=0))
    assert not off._use_dense(1, "cuda")
    assert off.dense_max_tokens == 0


def test_moe_dense_max_tokens_validated():
    import pytest

    assert InkyConfig().moe_dense_max_tokens == 64
    with pytest.raises(ValueError):
        InkyConfig(moe_dense_max_tokens=-1)
    assert InkyConfig(moe_dense_max_tokens=0).moe_dense_max_tokens == 0


def test_grouped_and_dense_dispatch_gradients_match():
    torch.manual_seed(4)
    cfg = tiny_config()
    moe = SharedExpertMoE(cfg)
    flat0 = torch.randn(11, cfg.hidden_size)
    base0 = torch.randn_like(flat0)
    results = []
    for fn in (moe._dispatch_routed, moe._dispatch_routed_dense):
        moe.zero_grad()
        flat = flat0.clone().requires_grad_(True)
        # The weights come from the router so that router gradients flow through the dispatch.
        idx, weights = _route(moe, flat)
        out = fn(flat, idx, weights, base0.clone())
        (out * torch.linspace(0.5, 1.5, out.numel()).view_as(out)).sum().backward()
        results.append((
            out.detach().clone(),
            flat.grad.clone(),
            moe.router.weight.grad.clone(),
            [p.grad.clone() for e in moe.routed_experts for p in e.parameters()],
        ))
    (o1, gx1, gr1, ge1), (o2, gx2, gr2, ge2) = results
    assert torch.allclose(o1, o2, atol=1e-6, rtol=0)
    assert torch.allclose(gx1, gx2, atol=1e-5, rtol=1e-4)
    assert gr1.abs().sum() > 0 and torch.allclose(gr1, gr2, atol=1e-5, rtol=1e-4)
    assert len(ge1) == len(ge2) and any(g.abs().sum() > 0 for g in ge1)
    for a, b in zip(ge1, ge2):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_dense_dispatch_inf_from_unselected_expert_stays_finite():
    torch.manual_seed(5)
    cfg = tiny_config()
    moe = SharedExpertMoE(cfg).eval()
    flat = torch.randn(6, cfg.hidden_size)
    with torch.no_grad():
        idx, weights = _route(moe, flat)
        # Pick an expert and make it return inf for every token; only tokens that do not select it are checked.
        target = 0
        unselected = ~(idx == target).any(dim=1)
        if not unselected.any():  # make sure at least one token does not select the target expert
            idx = torch.tensor([[1, 2]] * 6)
            weights = torch.full((6, 2), 0.5)
            unselected = torch.ones(6, dtype=torch.bool)
        moe.routed_experts[target].down_proj.weight.fill_(float("inf"))
        selects = (idx == target).any(dim=1)
        out = moe._dispatch_routed_dense(flat, idx, weights, torch.zeros_like(flat))
    assert torch.isfinite(out[~selects]).all()


def test_qk_norm_cached_decode_matches_full_forward_fp32():
    torch.manual_seed(2)
    cfg = tiny_config(qk_norm=True)
    attn = MultiLatentAttention(cfg).eval()
    # Non-trivial norm gains, so that the test notices a norm that is skipped on one path.
    for name, module in attn.named_modules():
        if name.endswith("_norm") and hasattr(module, "weight") and module.weight.numel() > 0:
            module.weight.data.uniform_(0.5, 1.5)
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.rope_theta)
    B, S, P = 2, 9, 4
    x = torch.randn(B, S, cfg.hidden_size)
    cos, sin = rope(0, S)
    with torch.no_grad():
        full, _ = attn(x, cos, sin)
        prefix, past = attn(x[:, :P], cos[:P], sin[:P])
        steps = [prefix]
        for t in range(P, S):
            o, past = attn(x[:, t:t + 1], cos[t:t + 1], sin[t:t + 1], past=past)
            steps.append(o)
        cached = torch.cat(steps, dim=1)
    assert torch.allclose(full, cached, atol=1e-5, rtol=0), (full - cached).abs().max()


def test_qk_norm_forward_subset_matches_full_forward():
    """The early-exit subset path must apply the same QK-norm as the full forward."""
    torch.manual_seed(3)
    cfg = tiny_config(qk_norm=True)
    attn = MultiLatentAttention(cfg).eval()
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.rope_theta)
    S = 6
    x = torch.randn(1, S, cfg.hidden_size)
    cos, sin = rope(0, S)
    store = (
        torch.zeros(1, S, cfg.kv_lora_rank),
        torch.zeros(1, 1, S, cfg.qk_rope_head_dim),
    )
    active = torch.arange(S)
    with torch.no_grad():
        full, _ = attn(x, cos, sin)
        subset = attn.forward_subset(x, active, cos, sin, store)
    assert torch.allclose(full, subset, atol=1e-5, rtol=0), (full - subset).abs().max()


def test_qk_norm_default_is_off_and_adds_no_parameters_when_off():
    cfg = InkyConfig()
    assert cfg.qk_norm is False
    attn = MultiLatentAttention(tiny_config())
    assert attn.qk_norm is False
    assert not any("norm" in name and name.startswith(("q_nope", "q_pe", "k_nope", "k_pe"))
                   for name, _ in attn.named_parameters())


def test_default_parameter_count_unchanged():
    from inky.modeling import InkyModel

    model = InkyModel(InkyConfig())
    assert sum(p.numel() for p in model.parameters()) == DEFAULT_PARAM_COUNT


def test_qk_norm_cached_decode_bf16_autocast_close_to_full_forward():
    """qk_norm cached decode (absorbed path, materialised k_nope) runs under bf16 and tracks the full forward."""
    torch.manual_seed(6)
    cfg = tiny_config(qk_norm=True)
    attn = MultiLatentAttention(cfg).eval()
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.rope_theta)
    B, S, P = 2, 8, 4
    x = torch.randn(B, S, cfg.hidden_size)
    cos, sin = rope(0, S)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        full, _ = attn(x, cos, sin)
        prefix, past = attn(x[:, :P], cos[:P], sin[:P])
        steps = [prefix]
        for t in range(P, S):
            o, past = attn(x[:, t:t + 1], cos[t:t + 1], sin[t:t + 1], past=past)
            steps.append(o)
        cached = torch.cat(steps, dim=1)
    assert cached.shape == full.shape
    assert torch.isfinite(cached.float()).all()
    assert torch.allclose(full.float(), cached.float(), atol=0.1, rtol=0.1), (full.float() - cached.float()).abs().max()
