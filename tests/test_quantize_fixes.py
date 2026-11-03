"""Tests for GEN-04: cached and fused dequantization in QuantLinear.

Run with: python -B -m pytest tests/test_quantize.py tests/test_quantize_fixes.py -q -p no:cacheprovider
"""

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inky.quantize as quantize  # noqa: E402
from inky.quantize import QuantLinear, quantize_model  # noqa: E402
from tests.test_quantize import model, tiny_config  # noqa: E402,F401  (fixture reuse)
from inky import InkyModel  # noqa: E402


def _linear(in_f=48, out_f=40, bias=True, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f, bias=bias)


def _reference(ql: QuantLinear, x: torch.Tensor) -> torch.Tensor:
    """The pre-GEN-04 forward: full dequantize, then one F.linear."""
    b = None if ql.bias is None else ql.bias.to(x.dtype)
    return F.linear(x, ql.dequantize(x.dtype), b)


@pytest.fixture
def small_chunks(monkeypatch):
    """Force the fused path to split the weight into many row blocks."""
    monkeypatch.setattr(quantize, "_FUSED_CHUNK_ELEMS", 16 * 48)
    return 16 * 48


def _assert_close(a, b):
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), float((a - b).abs().max())


@pytest.mark.parametrize("bits,in_f", [(8, 48), (4, 48), (4, 15)])
@pytest.mark.parametrize("cache", [False, True])
def test_forward_matches_full_dequantize_path(bits, in_f, cache, small_chunks, monkeypatch):
    lin = _linear(in_f=in_f, out_f=40, bias=True)
    ql = QuantLinear.from_linear(lin, bits=bits, group_size=16, cache_dequantized=cache)
    x = torch.randn(2, 5, in_f)
    _assert_close(ql(x), _reference(ql, x))
    # Also with the default (single-block) chunk size.
    monkeypatch.setattr(quantize, "_FUSED_CHUNK_ELEMS", 1 << 30)
    _assert_close(ql(x), _reference(ql, x))


def test_fused_path_never_materialises_full_weight(small_chunks, monkeypatch):
    ql = QuantLinear.from_linear(_linear(in_f=48, out_f=40), bits=8)
    requested = []
    orig = QuantLinear._dequantize_rows

    def spy(self, start, end, dtype):
        requested.append(end - start)
        return orig(self, start, end, dtype)

    monkeypatch.setattr(QuantLinear, "_dequantize_rows", spy)
    ql(torch.randn(3, 48))
    assert requested and max(requested) <= 16
    assert sum(requested) == 40


def test_cache_off_is_default_and_keeps_no_float_copy():
    ql = QuantLinear.from_linear(_linear(), bits=8)
    assert ql.cache_dequantized is False
    ql(torch.randn(2, 48))
    assert ql._dequant_cache is None


def test_quantize_model_default_is_no_cache(model):
    q = quantize_model(copy.deepcopy(model), bits=8)
    layers = [m for m in q.modules() if isinstance(m, QuantLinear)]
    assert layers and not any(m.cache_dequantized for m in layers)


def test_model_outputs_equal_with_and_without_cache(model):
    ids = torch.randint(0, 512, (2, 10))
    for bits in (8, 4):
        off = quantize_model(copy.deepcopy(model), bits=bits, cache_dequantized=False)
        on = quantize_model(copy.deepcopy(model), bits=bits, cache_dequantized=True)
        with torch.no_grad():
            _assert_close(on(ids).logits, off(ids).logits)


def test_cache_dequantizes_once_across_forwards(monkeypatch):
    ql = QuantLinear.from_linear(_linear(), bits=4, group_size=16, cache_dequantized=True)
    calls = {"n": 0}
    orig = QuantLinear.dequantize

    def spy(self, dtype=torch.float32):
        calls["n"] += 1
        return orig(self, dtype)

    monkeypatch.setattr(QuantLinear, "dequantize", spy)
    xs = [torch.randn(b, 48) for b in (1, 3, 1, 7, 2)]
    outs = [ql(x) for x in xs]
    assert calls["n"] == 1
    _ = ql.weight  # the .weight property also reuses the cache
    assert calls["n"] == 1
    for x, y in zip(xs, outs):  # reference comparisons call dequantize themselves, so check them last
        _assert_close(y, _reference(ql, x))


def test_no_cache_dequantizes_every_forward_on_single_block_path(monkeypatch):
    ql = QuantLinear.from_linear(_linear(), bits=8, cache_dequantized=False)
    calls = {"n": 0}
    orig = QuantLinear.dequantize

    def spy(self, dtype=torch.float32):
        calls["n"] += 1
        return orig(self, dtype)

    monkeypatch.setattr(QuantLinear, "dequantize", spy)
    for _ in range(4):
        ql(torch.randn(2, 48))
    assert calls["n"] == 4


def test_cache_invalidated_by_load_state_dict_and_dtype_move():
    ql = QuantLinear.from_linear(_linear(), bits=8, cache_dequantized=True)
    x = torch.randn(2, 48)
    before = ql(x).clone()

    other = QuantLinear.from_linear(_linear(seed=7), bits=8, cache_dequantized=True)
    ql.load_state_dict(other.state_dict())
    assert ql._dequant_cache is None
    _assert_close(ql(x), _reference(ql, x))
    assert not torch.allclose(ql(x), before)

    ql.to(torch.float64)
    assert ql._dequant_cache is None


def test_weight_property_unchanged_with_and_without_cache():
    lin = _linear()
    off = QuantLinear.from_linear(lin, bits=4, group_size=16, cache_dequantized=False)
    on = QuantLinear.from_linear(lin, bits=4, group_size=16, cache_dequantized=True)
    assert torch.equal(off.weight, on.weight)
    assert torch.equal(off.weight, off.dequantize(torch.float32))


def test_state_dict_keys_unchanged_by_cache(model):
    off = quantize_model(copy.deepcopy(model), bits=8, cache_dequantized=False)
    on = quantize_model(copy.deepcopy(model), bits=8, cache_dequantized=True)
    on(torch.randint(0, 512, (1, 4)))
    assert set(off.state_dict().keys()) == set(on.state_dict().keys())
