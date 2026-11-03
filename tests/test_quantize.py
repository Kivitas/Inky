"""Tests for weight-only quantization. Run with: python -m pytest tests/test_quantize.py -q"""

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.quantize import QuantLinear, model_nbytes, quantize_model  # noqa: E402


def tiny_config() -> InkyConfig:
    return InkyConfig(
        vocab_size=512,
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
        arlo_dim=16,
        arlo_pointer_rank=8,
        arlo_slots=4,
        enable_arlo=True,
        scratch_open_id=508,
        scratch_sep_id=509,
        scratch_close_id=510,
        eos_token_id=1,
        pad_token_id=1,
    )


@pytest.fixture
def model() -> InkyModel:
    torch.manual_seed(0)
    m = InkyModel(tiny_config())
    m.eval()
    return m


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach(), b.detach()
    return float((a - b).norm() / b.norm())


def test_int8_output_close_to_float(model):
    ids = torch.randint(0, 512, (2, 12))
    ref = model(ids).logits
    q = quantize_model(copy.deepcopy(model), bits=8)
    out = q(ids).logits
    assert _rel_err(out, ref) < 0.02


def test_int4_output_close_to_float(model):
    ids = torch.randint(0, 512, (2, 12))
    ref = model(ids).logits
    q = quantize_model(copy.deepcopy(model), bits=4, group_size=64)
    out = q(ids).logits
    assert _rel_err(out, ref) < 0.2


def test_model_bytes_shrink(model):
    base = model_nbytes(model)
    q8 = quantize_model(copy.deepcopy(model), bits=8)
    q4 = quantize_model(copy.deepcopy(model), bits=4, group_size=64)
    assert model_nbytes(q8) * 2 <= base
    assert model_nbytes(q4) * 3 <= base


def test_generate_on_quantized_model(model):
    for bits in (8, 4):
        q = quantize_model(copy.deepcopy(model), bits=bits)
        prompt = torch.randint(0, 512, (1, 6))
        out = q.generate(
            prompt,
            max_new_tokens=5,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        assert out.shape[1] == prompt.shape[1] + 5


def test_skipped_modules_stay_unquantized(model):
    q = quantize_model(copy.deepcopy(model), bits=8)
    assert isinstance(q.embed_tokens, nn.Embedding)
    arlo_linears = [m for m in q.arlo.modules() if isinstance(m, nn.Linear)]
    assert arlo_linears, "expected Linear layers inside arlo"
    assert not any(isinstance(m, QuantLinear) for m in q.arlo.modules())
    assert all(type(m) is nn.Linear for m in arlo_linears)
    assert any(isinstance(m, QuantLinear) for m in q.core.modules())


def test_int4_packing_handles_odd_in_features():
    torch.manual_seed(0)
    lin = nn.Linear(15, 7, bias=True)
    ql = QuantLinear.from_linear(lin, bits=4, group_size=64)
    assert ql.group_size == 15  # largest divisor of 15 that is <= 64
    x = torch.randn(3, 15)
    assert _rel_err(ql(x), lin(x)) < 0.2
