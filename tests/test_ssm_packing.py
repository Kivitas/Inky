"""SelectiveSSMMixer with packed documents (segment_ids), and the unchanged default path."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.layers import SelectiveSSMMixer  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        highway_gain_init=0.1,  # keeps the highway active in these checks (the default gain is 0)
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
        eos_token_id=1,
        pad_token_id=1,
        arlo_dim=16,
        arlo_pointer_rank=8,
        arlo_slots=4,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def make_mixer(seed: int = 0, chunk: int = 4) -> SelectiveSSMMixer:
    torch.manual_seed(seed)
    mixer = SelectiveSSMMixer(tiny_config()).eval()
    mixer.scan_chunk = chunk
    return mixer


def segs(values, batch: int = 1) -> torch.Tensor:
    return torch.tensor([values] * batch, dtype=torch.long)


def reference_mixer(mixer: SelectiveSSMMixer, x: torch.Tensor) -> torch.Tensor:
    """Token-by-token reference for the default path (no packing): the same maths as forward, no chunks."""
    u = mixer.in_proj(x)                                  # [B, S, d]
    a = torch.exp(F.logsigmoid(mixer.decay_proj(x)))      # [B, S, N]
    g = torch.sigmoid(mixer.gate_proj(x))                 # [B, S, N]
    B, S, d = u.shape
    h = torch.zeros(B, d, mixer.state_dim, dtype=u.dtype)
    ys = []
    for t in range(S):
        h = a[:, t, None, :] * h + g[:, t, None, :] * u[:, t, :, None]    # [B, d, N]
        ys.append((h * mixer.readout).sum(-1))                              # [B, d]
    y = torch.stack(ys, dim=1) + mixer.skip * u
    return mixer.out_proj(y)


# ---------------------------------------------------------------- a. packed segments match separate runs

@pytest.mark.parametrize("layout", [
    [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2],   # boundaries inside chunks, one at a chunk edge (token 8 = 2 * 4)
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],   # boundary exactly at a chunk edge (token 4)
])
def test_packed_segments_match_separate_runs(layout):
    torch.manual_seed(1)
    mixer = make_mixer()
    x = torch.randn(1, 11, 64)
    with torch.no_grad():
        packed, _ = mixer(x, segment_ids=segs(layout))
        ids = torch.tensor(layout)
        for sid in sorted(set(layout)):
            idx = (ids == sid).nonzero().flatten()
            start, stop = int(idx[0]), int(idx[-1]) + 1
            alone, _ = mixer(x[:, start:stop])
            diff = float((packed[:, start:stop] - alone).abs().max())
            assert torch.allclose(packed[:, start:stop], alone, atol=1e-5), (sid, diff)


def test_packed_segments_match_separate_runs_in_a_batch_with_different_layouts():
    torch.manual_seed(2)
    mixer = make_mixer(seed=3)
    x = torch.randn(2, 11, 64)
    layouts = [[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]
    seg = torch.tensor(layouts, dtype=torch.long)
    with torch.no_grad():
        packed, _ = mixer(x, segment_ids=seg)
        for b, layout in enumerate(layouts):
            ids = torch.tensor(layout)
            for sid in sorted(set(layout)):
                idx = (ids == sid).nonzero().flatten()
                start, stop = int(idx[0]), int(idx[-1]) + 1
                alone, _ = mixer(x[b:b + 1, start:stop])
                assert torch.allclose(packed[b:b + 1, start:stop], alone, atol=1e-5)


# ---------------------------------------------------------------- b. all-equal ids are the default path

def test_all_equal_segment_ids_match_no_segment_ids():
    torch.manual_seed(4)
    mixer = make_mixer()
    x = torch.randn(2, 11, 64)
    with torch.no_grad():
        plain, plain_present = mixer(x)
        seg_out, seg_present = mixer(x, segment_ids=torch.zeros(2, 11, dtype=torch.long))
    assert torch.allclose(seg_out, plain, atol=1e-6)
    assert torch.allclose(seg_present[0], plain_present[0], atol=1e-6)


# ---------------------------------------------------------------- c. argument checks

def test_segment_ids_with_past_raises_value_error():
    mixer = make_mixer()
    x = torch.randn(1, 1, 64)
    past = (torch.zeros(1, 64, 16), torch.zeros(1, 1, 0))
    with pytest.raises(ValueError, match="prefill"):
        mixer(x, past=past, segment_ids=torch.zeros(1, 1, dtype=torch.long))


def test_segment_ids_wrong_shape_raises_value_error():
    mixer = make_mixer()
    x = torch.randn(1, 4, 64)
    with pytest.raises(ValueError, match="segment_ids must have shape"):
        mixer(x, segment_ids=torch.zeros(1, 3, dtype=torch.long))


def test_attn_mask_still_raises_not_implemented():
    mixer = make_mixer()
    x = torch.randn(1, 4, 64)
    with pytest.raises(NotImplementedError, match="attn_mask"):
        mixer(x, attn_mask=torch.ones(1, 4, 4, dtype=torch.bool))


# ---------------------------------------------------------------- d. model level: packed == separate

def test_hybrid_model_packed_documents_match_documents_run_separately():
    torch.manual_seed(0)
    model = InkyModel(tiny_config(highway_gain_init=0.3, ssm_mixer_layers=(1,))).eval()
    a = torch.randint(2, 500, (1, 5))
    b = torch.randint(2, 500, (1, 4))
    packed = torch.cat([a, b], dim=1)
    segments = torch.tensor([[0] * 5 + [1] * 4])
    with torch.no_grad():
        together = model(packed, segment_ids=segments, num_loops=3).logits
        alone_a = model(a, num_loops=3).logits
        alone_b = model(b, num_loops=3).logits
    assert torch.allclose(together[:, :5], alone_a, atol=1e-4), float((together[:, :5] - alone_a).abs().max())
    assert torch.allclose(together[:, 5:], alone_b, atol=1e-4), float((together[:, 5:] - alone_b).abs().max())


# ---------------------------------------------------------------- e. packed training step

def test_hybrid_model_packed_training_step_is_finite():
    torch.manual_seed(0)
    model = InkyModel(tiny_config(highway_gain_init=0.3, ssm_mixer_layers=(1,))).train()
    ids = torch.randint(2, 500, (2, 9))
    seg = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1]])
    out = model(ids, labels=ids, segment_ids=seg, num_loops=2)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    mixers = [m for m in model.modules() if isinstance(m, SelectiveSSMMixer)]
    assert mixers, "expected one SelectiveSSMMixer in the hybrid model"
    for mixer in mixers:
        for name, p in mixer.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------- f. default path matches a token-by-token reference

def test_default_path_matches_token_by_token_reference():
    torch.manual_seed(5)
    mixer = make_mixer(chunk=4)
    x = torch.randn(2, 11, 64)
    with torch.no_grad():
        out, _ = mixer(x)
        ref = reference_mixer(mixer, x)
    assert torch.allclose(out, ref, atol=1e-4), float((out - ref).abs().max())


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        if test.__code__.co_argcount:
            print(f"skip {test.__name__} (needs pytest fixtures)")
        else:
            test()
            print(f"ok {test.__name__}")
