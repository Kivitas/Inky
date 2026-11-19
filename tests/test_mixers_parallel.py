"""Chunked GatedDeltaNet prefill / SSM scan vs naive per-token references, cache handoff, edge cases, MoE."""
import pytest
import torch
import torch.nn.functional as F

from inky.config import InkyConfig
from inky.layers import GatedDeltaNetMixer, SelectiveSSMMixer, SharedExpertMoE


def _cfg():
    return InkyConfig(hidden_size=64, num_attention_heads=4, ssm_state_dim=8, intermediate_size=64,
                      num_routed_experts=6, num_active_routed=2)


def _segs(B, S, returning=True):
    seg = torch.zeros(B, S, dtype=torch.long)
    seg[0, S // 3:] = 1
    seg[0, 2 * S // 3:] = 0 if returning else 2   # a returning id is a NEW run for the GDN recurrence
    seg[-1, min(5, S - 1):] = 3
    return seg


@pytest.fixture(scope="module")
def gdn():
    torch.manual_seed(0)
    m = GatedDeltaNetMixer(_cfg()).double().eval()
    return m


@pytest.mark.parametrize("S", [1, 2, 63, 64, 65, 150])
@pytest.mark.parametrize("use_seg", [False, True])
def test_gdn_chunked_matches_recurrent(gdn, S, use_seg):
    x = torch.randn(2, S, 64, dtype=torch.double)
    seg = _segs(2, S) if use_seg else None
    with torch.no_grad():
        gdn.recurrent_max_tokens = 10 ** 9
        y0, (s0, _) = gdn(x, segment_ids=seg)
        gdn.recurrent_max_tokens = 1
        y1, (s1, _) = gdn(x, segment_ids=seg)
    torch.testing.assert_close(y1, y0, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(s1, s0, atol=1e-10, rtol=1e-8)


def test_gdn_chunk_sizes_agree(gdn):
    x = torch.randn(1, 100, 64, dtype=torch.double)
    with torch.no_grad():
        ref = gdn(x)[0]
        for cs in (1, 7, 32):
            gdn.chunk_size = cs
            torch.testing.assert_close(gdn(x)[0], ref, atol=1e-10, rtol=1e-8)
        del gdn.chunk_size


def test_gdn_fp32_state_and_prefill_decode_handoff():
    torch.manual_seed(1)
    m = GatedDeltaNetMixer(_cfg()).eval()
    x = torch.randn(2, 90, 64)
    with torch.no_grad():
        full, _ = m(x)
        y_pre, past = m(x[:, :70])
        assert past[0].dtype == torch.float32
        outs = [y_pre]
        for t in range(70, 90):                       # single-token decode
            y, past = m(x[:, t:t + 1], past=past)
            outs.append(y)
        torch.testing.assert_close(torch.cat(outs, 1), full, atol=2e-5, rtol=1e-4)
        # multi-token continuation (chunked with a carried state)
        _, p1 = m(x[:, :40])
        y2, _ = m(x[:, 40:], past=p1)
        torch.testing.assert_close(y2, full[:, 40:], atol=2e-5, rtol=1e-4)


def test_gdn_bf16_finite_and_close():
    torch.manual_seed(2)
    m = GatedDeltaNetMixer(_cfg()).eval()
    x = torch.randn(2, 80, 64)
    with torch.no_grad():
        ref = m(x)[0]
        mb = GatedDeltaNetMixer(_cfg()).eval()
        mb.load_state_dict(m.state_dict())
        mb = mb.bfloat16()
        y, (st, _) = mb(x.bfloat16())
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()
    assert st.dtype == torch.float32
    assert (y.float() - ref).abs().max() < 0.1 * ref.abs().max() + 1e-2


def test_gdn_gradients_match_reference():
    torch.manual_seed(3)
    m = GatedDeltaNetMixer(_cfg()).double()
    x = torch.randn(1, 70, 64, dtype=torch.double, requires_grad=True)
    grads = []
    for mx in (10 ** 9, 1):
        m.recurrent_max_tokens = mx
        m.zero_grad()
        x.grad = None
        m(x)[0].pow(2).sum().backward()
        grads.append((x.grad.clone(), m.q_proj.weight.grad.clone()))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-6)


def _ssm_reference(m, x, seg=None):
    u = m.in_proj(x)
    a = torch.exp(F.logsigmoid(m.decay_proj(x)))
    g = torch.sigmoid(m.gate_proj(x))
    B, S, d = u.shape
    h = u.new_zeros(B, d, m.state_dim)
    ys = []
    for t in range(S):
        if seg is not None and t > 0:
            h = torch.where((seg[:, t] == seg[:, t - 1]).view(B, 1, 1), h, torch.zeros_like(h))
        h = a[:, t].unsqueeze(1) * h + u[:, t].unsqueeze(-1) * g[:, t].unsqueeze(1)
        ys.append((h * m.readout).sum(-1))
    y = torch.stack(ys, 1) + m.skip * u
    return m.out_proj(y), h


@pytest.mark.parametrize("S", [1, 2, 31, 32, 33, 100])
@pytest.mark.parametrize("use_seg", [False, True])
def test_ssm_matches_naive(S, use_seg):
    torch.manual_seed(4)
    m = SelectiveSSMMixer(_cfg()).double().eval()
    m.skip.data.normal_()
    x = torch.randn(2, S, 64, dtype=torch.double)
    seg = _segs(2, S, returning=False) if use_seg else None
    with torch.no_grad():
        y, (h, _) = m(x, segment_ids=seg)
        yr, hr = _ssm_reference(m, x, seg)
    torch.testing.assert_close(y, yr, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(h, hr, atol=1e-10, rtol=1e-8)


def test_ssm_handoff_and_bf16():
    torch.manual_seed(5)
    m = SelectiveSSMMixer(_cfg()).eval()
    x = torch.randn(2, 75, 64)
    with torch.no_grad():
        full, _ = m(x)
        y1, p = m(x[:, :50])
        y2, _ = m(x[:, 50:], past=p)
        torch.testing.assert_close(torch.cat([y1, y2], 1), full, atol=2e-5, rtol=1e-4)
        mb = SelectiveSSMMixer(_cfg()).eval()
        mb.load_state_dict(m.state_dict())
        yb, (hb, _) = mb.bfloat16()(x.bfloat16())
    assert yb.dtype == torch.bfloat16 and torch.isfinite(yb.float()).all()


def test_moe_edge_cases():
    torch.manual_seed(6)
    m = SharedExpertMoE(_cfg()).eval()
    for shape in [(1, 1, 64), (3, 5, 64)]:
        out, z = m(torch.randn(*shape))
        assert out.shape == shape and torch.isfinite(out).all() and torch.isfinite(z)
    with pytest.raises(ValueError):
        m(torch.randn(1, 4, 64), expert_bias=torch.zeros(1, 4, 3))
    out, _ = m(torch.randn(1, 4, 64), expert_bias=torch.zeros(1, 4, 6))
    assert out.shape == (1, 4, 64)
    mb = SharedExpertMoE(_cfg()).bfloat16().eval()
    out, _ = mb(torch.randn(2, 7, 64).bfloat16())
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
