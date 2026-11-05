"""Tests for bytes_per_token, bits_per_byte and evaluate(..., bytes_per_token=...).
Run with: python -m pytest tests/test_bits_per_byte.py -q"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.evaluation import bits_per_byte, bytes_per_token, evaluate  # noqa: E402

VOCAB = 512


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        highway_gain_init=0.1,
        vocab_size=VOCAB,
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
        scratch_open_id=508,
        scratch_sep_id=509,
        scratch_close_id=510,
        eos_token_id=1,
        pad_token_id=1,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def make_batches(n_batches: int = 2, B: int = 2, S: int = 16, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, VOCAB, (B, S), generator=g, dtype=torch.long) for _ in range(n_batches)]


@pytest.fixture
def model() -> InkyModel:
    torch.manual_seed(0)
    m = InkyModel(tiny_config())
    m.eval()
    return m


def whitespace_encode(s: str):
    return list(s.split())


# ---------------------------------------------------------------- (a) bytes_per_token

def test_bytes_per_token_counts_utf8_bytes_not_characters():
    # "héllo wörld": héllo = 1+2+1+1+1 = 6 bytes, space = 1, wörld = 1+2+1+1+1 = 6 -> 13 bytes, 2 tokens.
    # "naïve café":  naïve = 1+1+2+1+1 = 6 bytes, space = 1, café = 1+1+1+2 = 5 -> 12 bytes, 2 tokens.
    # Total 25 bytes over 4 tokens = 6.25 (the character count would be 21 and give 5.25).
    texts = ["héllo wörld", "naïve café"]
    assert bytes_per_token(texts, whitespace_encode) == pytest.approx(25 / 4)


def test_bytes_per_token_accepts_any_iterable():
    texts = (t for t in ["héllo wörld", "naïve café"])
    assert bytes_per_token(texts, whitespace_encode) == pytest.approx(25 / 4)


def test_bytes_per_token_ascii_example():
    # "hi there" = 8 bytes, 2 tokens -> 4.0
    assert bytes_per_token(["hi there"], whitespace_encode) == pytest.approx(4.0)


# ---------------------------------------------------------------- (b) bits_per_byte

def test_bits_per_byte_eight_bits_per_four_bytes_is_two():
    assert bits_per_byte(math.log(2) * 8, 4.0) == pytest.approx(2.0)


def test_bits_per_byte_hand_computed():
    # nll = 1 nat per token, 2 bytes per token: 1 / (2 ln 2) = 0.72134752...
    assert bits_per_byte(1.0, 2.0) == pytest.approx(1.0 / (2.0 * math.log(2.0)), rel=1e-12)
    assert bits_per_byte(1.0, 2.0) == pytest.approx(0.7213475204444817, rel=1e-9)


def test_bits_per_byte_zero_nll_is_zero():
    assert bits_per_byte(0.0, 3.0) == 0.0


# ---------------------------------------------------------------- (c) error cases

def test_bytes_per_token_rejects_empty_texts():
    with pytest.raises(ValueError):
        bytes_per_token([], whitespace_encode)


def test_bytes_per_token_rejects_texts_with_no_tokens():
    with pytest.raises(ValueError):
        bytes_per_token(["abc", "def"], lambda s: [])


@pytest.mark.parametrize("bad_bytes", [0.0, -1.0])
def test_bits_per_byte_rejects_non_positive_bytes(bad_bytes):
    with pytest.raises(ValueError):
        bits_per_byte(1.0, bad_bytes)


def test_bits_per_byte_rejects_negative_nll():
    with pytest.raises(ValueError):
        bits_per_byte(-0.1, 4.0)


@pytest.mark.parametrize(
    "nll, nbytes",
    [(float("nan"), 4.0), (1.0, float("nan")), (float("inf"), 4.0), (1.0, float("inf"))],
)
def test_bits_per_byte_rejects_non_finite(nll, nbytes):
    with pytest.raises(ValueError):
        bits_per_byte(nll, nbytes)


# ---------------------------------------------------------------- (d) model-level evaluate

def test_evaluate_reports_bits_per_byte_when_given(model):
    batches = make_batches()
    result = evaluate(model, batches, bytes_per_token=3.0)
    assert result["bits_per_byte"] == pytest.approx(result["nll"] / (3.0 * math.log(2)))


def test_evaluate_omits_bits_per_byte_by_default(model):
    assert "bits_per_byte" not in evaluate(model, make_batches())
