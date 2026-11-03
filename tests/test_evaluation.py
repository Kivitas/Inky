"""Tests for the held-out evaluation harness. Run with: python -m pytest tests/test_evaluation.py -q"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.evaluation import bits_per_token, evaluate, perplexity, split_heldout  # noqa: E402
from inky.modeling import InkyOutput  # noqa: E402

VOCAB = 512


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
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


@pytest.fixture
def model() -> InkyModel:
    torch.manual_seed(0)
    m = InkyModel(tiny_config())
    m.eval()
    return m


def make_batches(n_batches: int = 3, B: int = 2, S: int = 16, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, VOCAB, (B, S), generator=g, dtype=torch.long) for _ in range(n_batches)]


def test_split_heldout_is_deterministic_and_sized():
    seqs = list(range(100))
    train_a, held_a = split_heldout(seqs, fraction=0.1, seed=0)
    train_b, held_b = split_heldout(seqs, fraction=0.1, seed=0)
    assert held_a == held_b and train_a == train_b
    assert len(held_a) == 10 and len(train_a) == 90
    assert sorted(train_a + held_a) == seqs  # a true partition of the input
    _, held_c = split_heldout(seqs, fraction=0.1, seed=7)
    assert held_c != held_a  # a different seed gives a different split
    # Held-out size is at least one, even for a tiny fraction.
    _, held_small = split_heldout(list(range(5)), fraction=0.01, seed=0)
    assert len(held_small) == 1


def test_perplexity_nll_matches_manual_cross_entropy(model):
    ids = make_batches(n_batches=1)[0]
    with torch.no_grad():
        logits = model(ids).logits
        manual = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1), reduction="mean"
        ).item()
    result = perplexity(model, [ids])
    assert abs(result["nll"] - manual) < 1e-5
    assert abs(result["perplexity"] - math.exp(manual)) < 1e-3
    assert result["tokens"] == ids.shape[0] * (ids.shape[1] - 1)
    assert result["batches"] == 1


def test_perplexity_ignores_padding_targets(model):
    PAD = 3
    ids = make_batches(n_batches=1)[0]
    padded = torch.cat([ids, torch.full((ids.shape[0], 4), PAD, dtype=torch.long)], dim=1)
    with torch.no_grad():
        logits = model(padded).logits
        targets = padded[:, 1:]
        keep = targets != PAD
        manual = F.cross_entropy(logits[:, :-1][keep], targets[keep], reduction="mean").item()
    result = perplexity(model, [padded], ignore_id=PAD)
    assert abs(result["nll"] - manual) < 1e-5
    assert result["tokens"] == int(keep.sum())
    # Without ignore_id the padding is scored, so the token count and NLL change.
    assert perplexity(model, [padded])["tokens"] == padded.shape[0] * (padded.shape[1] - 1)


class UniformStub(torch.nn.Module):
    """Returns all-zero logits, so every target has probability 1/V."""

    def __init__(self, vocab: int = VOCAB):
        super().__init__()
        self.vocab = vocab
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, num_loops=None):
        B, S = input_ids.shape
        return InkyOutput(logits=self.anchor.new_zeros(B, S, self.vocab))


def test_uniform_logits_give_perplexity_exactly_vocab():
    result = perplexity(UniformStub(), make_batches())
    assert abs(result["perplexity"] - VOCAB) < 1e-3
    assert abs(result["nll"] - math.log(VOCAB)) < 1e-6


def test_overfit_model_drops_well_below_vocab():
    # A tiny model trained on one fixed batch must memorise it: perplexity on that batch falls far
    # below the uniform value V. Before training it sits near V.
    torch.manual_seed(0)
    m = InkyModel(tiny_config())
    batch = make_batches(n_batches=1, B=2, S=16, seed=3)
    before = perplexity(m, batch)["perplexity"]
    assert 0.5 * VOCAB < before < 2 * VOCAB
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    m.train()
    for _ in range(80):
        opt.zero_grad()
        m(batch[0], labels=batch[0]).loss.backward()
        opt.step()
    after = perplexity(m, batch)["perplexity"]
    assert after < 0.05 * VOCAB, f"perplexity only fell from {before:.1f} to {after:.1f}"


def test_perplexity_restores_train_mode_and_respects_max_batches():
    torch.manual_seed(0)
    m = InkyModel(tiny_config())
    m.train()
    result = perplexity(m, make_batches(n_batches=5), max_batches=2)
    assert m.training is True
    assert result["batches"] == 2


def test_evaluate_mean_loops_within_bounds(model):
    cfg = model.config
    result = evaluate(model, make_batches())
    assert 1.0 <= result["mean_loops"] <= cfg.max_recurrent_loops
    assert result["batches"] == 3
    assert result["perplexity"] > 0


def test_bits_per_token():
    assert abs(bits_per_token(math.log(2.0)) - 1.0) < 1e-12
    assert bits_per_token(0.0) == 0.0
