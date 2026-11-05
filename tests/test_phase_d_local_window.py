"""Tests for Phase D: local (sliding-window) attention in selected loops.

Run with: python -B -m pytest tests/test_phase_d_local_window.py -q -p no:cacheprovider
"""

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from inky import InkyConfig, InkyModel
from inky.layers import GroupedQueryAttention, MultiLatentAttention, causal_band_mask

OPEN, SEP, CLOSE = 508, 509, 510


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512, scratch_open_id=OPEN, scratch_sep_id=SEP, scratch_close_id=CLOSE, hidden_size=64,
        num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16, q_lora_rank=32,
        kv_lora_rank=24, intermediate_size=96, num_routed_experts=4, num_active_routed=2, ssm_state_dim=16,
        max_recurrent_loops=4, min_recurrent_loops=2, loop_dropout_prob=0.0, max_position_embeddings=256,
        arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4, eos_token_id=1, pad_token_id=1, highway_gain_init=0.3,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def make_model(**overrides) -> InkyModel:
    torch.manual_seed(0)
    return InkyModel(tiny_config(**overrides)).eval()


def ids(rows=1, seq=24, seed=1):
    return torch.randint(2, 500, (rows, seq), generator=torch.Generator().manual_seed(seed))


def cached_hidden(model, x, first=10):
    out = model(x[:, :first], use_cache=True, return_hidden=True)
    parts = [out.hidden]
    for t in range(first, x.shape[1]):
        parts.append(model(x[:, t:t + 1], past=out.past, return_hidden=True).hidden)
    return torch.cat(parts, 1)


# ----------------------------------------------------------------------------- the mask


def test_causal_band_mask_by_hand():
    # 3 new queries after 2 cached positions, window 3: query at position 2 sees 0..2, 3 sees 1..3, 4 sees 2..4
    expected = torch.tensor([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(causal_band_mask(3, 5, 2, 3), expected)
    assert torch.equal(causal_band_mask(3, 5, 2, None), torch.tensor([
        [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool))
    assert torch.equal(causal_band_mask(3, 5, 2, 1), torch.eye(5, dtype=torch.bool)[2:])      # window 1: itself only


# ----------------------------------------------------------------------------- config


def test_defaults_are_global_and_validation_is_strict():
    cfg = InkyConfig()
    assert cfg.local_window is None and cfg.local_window_loops == ()
    assert tiny_config(local_window=8).local_window_loops == ()
    assert tiny_config(local_window=8, local_window_loops=[3, 1, 3]).local_window_loops == (1, 3)
    for kwargs, message in [
        (dict(local_window=0), "local_window"),
        (dict(local_window_loops=(1,)), "needs local_window"),
        (dict(local_window=8, local_window_loops=(5,)), "out of range"),
        (dict(local_window=8, local_window_loops=(0,)), "out of range"),
        (dict(local_window=8, local_window_loops=(1,), routing_mode="mor"), "mor"),
    ]:
        with pytest.raises(ValueError, match=message):
            tiny_config(**kwargs)


def test_early_exit_is_rejected_when_loops_are_local():
    model = make_model(local_window=8, local_window_loops=(1, 2))
    with pytest.raises(NotImplementedError, match="local_window_loops"):
        model.forward_early_exit(ids())


# ----------------------------------------------------------------------------- the attention itself


@pytest.mark.parametrize("kind", ["mla", "gqa"])
@pytest.mark.parametrize("window", [1, 3, 7])
def test_windowed_attention_equals_global_attention_on_the_last_window_tokens(kind, window):
    cfg = tiny_config(attention_type=kind)
    torch.manual_seed(0)
    attn = (MultiLatentAttention if kind == "mla" else GroupedQueryAttention)(cfg).eval()
    S = 12
    x = torch.randn(1, S, 64)
    from inky.layers import RotaryEmbedding
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, 64, cfg.rope_theta)
    cos, sin = rope(0, S)
    with torch.no_grad():
        windowed, _ = attn(x, cos, sin, window=window)
        for p in range(S):
            lo = max(0, p - window + 1)
            # RoPE scores depend only on relative positions, so the slice can use positions 0..len-1
            c, s = rope(0, p - lo + 1)
            ref, _ = attn(x[:, lo:p + 1], c, s)
            assert torch.allclose(windowed[:, p], ref[:, -1], atol=1e-5), (kind, window, p)


@pytest.mark.parametrize("kind", ["mla", "gqa"])
def test_a_window_that_covers_the_sequence_is_global(kind):
    cfg = tiny_config(attention_type=kind)
    torch.manual_seed(0)
    attn = (MultiLatentAttention if kind == "mla" else GroupedQueryAttention)(cfg).eval()
    from inky.layers import RotaryEmbedding
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, 64, cfg.rope_theta)
    x = torch.randn(1, 9, 64)
    cos, sin = rope(0, 9)
    with torch.no_grad():
        assert torch.equal(attn(x, cos, sin, window=9)[0], attn(x, cos, sin)[0])
        assert torch.equal(attn(x, cos, sin, window=100)[0], attn(x, cos, sin)[0])


# ----------------------------------------------------------------------------- the model


def test_loops_without_a_window_or_a_window_without_loops_change_nothing():
    base = make_model()
    x = ids(seq=40)
    with torch.no_grad():
        ref = base(x, return_hidden=True).hidden
        for model in (make_model(local_window=6), make_model(local_window=6, local_window_loops=())):
            assert torch.equal(model(x, return_hidden=True).hidden, ref)


def test_a_local_loop_changes_the_output_and_only_when_the_window_is_short():
    base = make_model()
    x = ids(seq=40)
    with torch.no_grad():
        ref = base(x, return_hidden=True).hidden
        short = make_model(local_window=6, local_window_loops=(1, 2, 3))(x, return_hidden=True).hidden
        long = make_model(local_window=64, local_window_loops=(1, 2, 3))(x, return_hidden=True).hidden
    assert not torch.allclose(short, ref, atol=1e-3)
    assert torch.equal(long, ref)


@pytest.mark.parametrize("kind", ["mla", "gqa"])
def test_cached_decode_equals_the_full_forward_with_local_loops(kind):
    model = make_model(attention_type=kind, local_window=5, local_window_loops=(1, 2, 3))
    x = ids(seq=30)
    with torch.no_grad():
        full = model(x, return_hidden=True).hidden
        assert torch.allclose(cached_hidden(model, x), full, atol=1e-5)
        first = model(x[:, :12], use_cache=True, return_hidden=True)                      # a multi-token chunk
        chunk = model(x[:, 12:20], past=first.past, return_hidden=True).hidden
        assert torch.allclose(chunk, full[:, 12:20], atol=1e-5)


def test_packed_documents_match_separate_runs_with_local_loops():
    model = make_model(local_window=4, local_window_loops=(1, 3))
    a, b = ids(seq=9, seed=2), ids(seq=8, seed=3)
    segments = torch.tensor([[0] * 9 + [1] * 8])
    with torch.no_grad():
        together = model(torch.cat([a, b], 1), segment_ids=segments, num_loops=4).logits
        assert torch.allclose(together[:, :9], model(a, num_loops=4).logits, atol=1e-4)
        assert torch.allclose(together[:, 9:], model(b, num_loops=4).logits, atol=1e-4)


def test_local_loops_train_with_gradient_checkpointing():
    model = make_model(local_window=4, local_window_loops=(1, 2), gradient_checkpointing=True).train()
    x = ids(rows=2, seq=20)
    out = model(x, labels=x, num_loops=3)
    out.loss.backward()
    assert torch.isfinite(out.loss) and all(p.grad is None or torch.isfinite(p.grad).all()
                                            for p in model.parameters())


def test_the_ssm_mixer_layer_ignores_the_window():
    model = make_model(ssm_mixer_layers=(1,), local_window=4, local_window_loops=(1, 2))
    x = ids(seq=16)
    with torch.no_grad():
        assert torch.isfinite(model(x, return_hidden=True).hidden).all()


def test_local_loops_cut_the_decode_flops_at_long_context():
    x = ids(seq=200)

    def decode_flops(model):
        with torch.no_grad():
            cache = model(x, use_cache=True, last_only=True).past
            with FlopCounterMode(display=False) as fc:
                model(ids(seq=1, seed=9), past=cache, last_only=True)
        return fc.get_total_flops()

    global_flops = decode_flops(make_model())
    local_flops = decode_flops(make_model(local_window=16, local_window_loops=(1, 2, 3)))
    last_global = decode_flops(make_model(local_window=16, local_window_loops=(1, 2, 3, 4)))
    assert local_flops < global_flops
    assert last_global < local_flops                       # one more local loop, one more saving
