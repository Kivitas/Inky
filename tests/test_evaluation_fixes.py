"""Tests for the evaluation fixes: chunked logits, token-weighted loops, loop sweep, early-exit drift,
and padding mask pass-through. Run with: python -m pytest tests/test_evaluation_fixes.py -q"""

import functools
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import evaluation  # noqa: E402
from inky import InkyConfig, InkyModel  # noqa: E402
from inky.evaluation import early_exit_drift, evaluate, perplexity, perplexity_by_loops  # noqa: E402
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


def make_batches(n_batches: int = 2, B: int = 2, S: int = 16, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, VOCAB, (B, S), generator=g, dtype=torch.long) for _ in range(n_batches)]


class LoopStub(nn.Module):
    """Uniform logits; reports a fixed mean loop count per call, taken in order from ``loops``."""

    def __init__(self, loops=None):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.loops = list(loops or [])

    def forward(self, input_ids, num_loops=None):
        B, S = input_ids.shape
        ml = self.loops.pop(0) if self.loops else None
        return InkyOutput(
            logits=self.anchor.new_zeros(B, S, VOCAB),
            mean_loops=None if ml is None else torch.tensor(float(ml)),
        )


class MaskRecorder(nn.Module):
    """Uniform logits; records the attention_mask it was given (if any)."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, input_ids, num_loops=None, attention_mask=None):
        self.seen.append(attention_mask)
        B, S = input_ids.shape
        return InkyOutput(logits=self.anchor.new_zeros(B, S, VOCAB))


# ---------------------------------------------------------------- 1. chunked logits

def test_chunked_nll_matches_full_logit_cross_entropy(model, monkeypatch):
    # A chunk size smaller than the sequence forces several chunks; the result must not change.
    monkeypatch.setattr(evaluation, "_LOGIT_CHUNK", 7)
    ids = make_batches(n_batches=1, S=40)[0]
    with torch.no_grad():
        logits = model(ids).logits
        manual = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)).item()
    assert abs(perplexity(model, [ids])["nll"] - manual) < 1e-5


def test_evaluation_does_not_request_full_logits(model, monkeypatch):
    original = model.forward
    seen = {}

    @functools.wraps(original)
    def spy(input_ids, num_loops=None, return_hidden=False, **kw):
        out = original(input_ids, num_loops=num_loops, return_hidden=return_hidden)
        seen["logits_none"] = out.logits is None
        seen["hidden_set"] = out.hidden is not None
        return out

    monkeypatch.setattr(model, "forward", spy)
    perplexity(model, make_batches(n_batches=1))
    assert seen == {"logits_none": True, "hidden_set": True}


# ---------------------------------------------------------------- 2. token-weighted loops

def test_mean_loops_is_weighted_by_positions_not_by_batch():
    # Batch A has 1x10 = 10 positions at 2.0 loops; batch B has 3x2 = 6 positions at 5.0 loops.
    # Weighted: (2*10 + 5*6) / 16 = 3.125. The old unweighted batch mean would give 3.5.
    stub = LoopStub(loops=[2.0, 5.0])
    a = torch.zeros(1, 10, dtype=torch.long)
    b = torch.zeros(3, 2, dtype=torch.long)
    result = evaluate(stub, [a, b])
    assert abs(result["mean_loops"] - 3.125) < 1e-6
    assert result["batches"] == 2


def test_mean_loops_none_when_model_reports_none():
    result = evaluate(LoopStub(loops=[]), make_batches(n_batches=2))
    assert result["mean_loops"] is None


# ---------------------------------------------------------------- 3. loop sweep and early exit

def test_perplexity_by_loops_matches_single_runs(model):
    batches = make_batches(n_batches=2)
    sweep = perplexity_by_loops(model, iter(batches), [1, 2, 4])
    assert sorted(sweep) == [1, 2, 4]
    for c in (1, 2, 4):
        single = perplexity(model, batches, num_loops=c)
        assert abs(sweep[c]["perplexity"] - single["perplexity"]) < 1e-6
        assert sweep[c]["tokens"] == single["tokens"]


def test_perplexity_by_loops_rejects_bad_counts(model):
    with pytest.raises(ValueError):
        perplexity_by_loops(model, make_batches(), [])
    with pytest.raises(ValueError):
        perplexity_by_loops(model, make_batches(), [0, 2])


def test_early_exit_without_threshold_matches_full_forward(model):
    result = early_exit_drift(model, make_batches(n_batches=2), halt_threshold=None, num_loops=3)
    assert abs(result["drift"]) < 1e-4
    assert abs(result["perplexity_ratio"] - 1.0) < 1e-4
    assert result["loop_fraction"] == pytest.approx(1.0)
    assert result["sequences"] == 4  # 2 batches of 2 rows, each row run on its own


def test_early_exit_with_threshold_reports_fewer_loop_applications(model):
    result = early_exit_drift(model, make_batches(n_batches=1), halt_threshold=0.3, num_loops=4)
    assert math.isfinite(result["early_nll"]) and math.isfinite(result["drift"])
    assert 0.0 < result["loop_fraction"] <= 1.0
    assert 1.0 <= result["mean_loops_early"] <= 4.0


def test_early_exit_restores_train_mode(model):
    model.train()
    early_exit_drift(model, make_batches(n_batches=1), halt_threshold=None)
    assert model.training is True


def test_early_exit_requires_forward_early_exit():
    with pytest.raises(ValueError, match="forward_early_exit"):
        early_exit_drift(LoopStub(), make_batches(n_batches=1))


# ---------------------------------------------------------------- 4. padding mask pass-through

def test_attention_mask_is_passed_when_forward_accepts_it():
    PAD = 3
    ids = torch.tensor([[5, 6, 7, PAD, PAD], [8, 9, PAD, PAD, PAD]])
    rec = MaskRecorder()
    perplexity(rec, [ids], ignore_id=PAD)
    assert len(rec.seen) == 1
    assert torch.equal(rec.seen[0], (ids != PAD).long())


def test_no_attention_mask_without_ignore_id_or_support():
    rec = MaskRecorder()
    perplexity(rec, make_batches(n_batches=1))  # no ignore_id: nothing to derive a mask from
    assert rec.seen == [None]
    # Without ignore_id there is nothing to derive a mask from, so no mask is passed, even though
    # InkyModel.forward now accepts an attention_mask argument.
    assert evaluation._accepts(InkyModel.forward, "attention_mask")
