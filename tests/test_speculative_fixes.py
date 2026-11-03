"""Equivalence, accounting and scaling tests for inky/speculative.py.

Run with: python -m pytest tests/test_speculative_fixes.py -q -p no:cacheprovider
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.speculative import self_speculative_generate  # noqa: E402

PROMPTS = [
    [[5, 17, 42, 99, 7]],
    [[3]],
    [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
]


def tiny_config(initializer_range: float = 0.5, max_position_embeddings: int = 256) -> InkyConfig:
    return InkyConfig(
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
        max_position_embeddings=max_position_embeddings,
        arlo_dim=16,
        arlo_pointer_rank=8,
        arlo_slots=4,
        eos_token_id=1,
        pad_token_id=1,
        initializer_range=initializer_range,
    )


def model_for(seed: int, initializer_range: float = 0.5, max_position_embeddings: int = 256) -> InkyModel:
    torch.manual_seed(seed)
    model = InkyModel(tiny_config(initializer_range, max_position_embeddings))
    model.eval()
    return model


def plain_greedy(model: InkyModel, input_ids: torch.Tensor, n: int, eos_token_id=None) -> torch.Tensor:
    return model.generate(
        input_ids,
        max_new_tokens=n,
        do_sample=False,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_loops=model.config.max_recurrent_loops,
        eos_token_id=eos_token_id,
    )


def varied(tokens) -> bool:
    return len(set(tokens)) >= 4


# --- Equivalence with plain greedy -------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("draft_len", [1, 2, 3, 5])
@pytest.mark.parametrize("draft_loops", [1, 2, 3])
def test_equivalence_varied_output_over_seeds_and_draft_settings(seed, draft_len, draft_loops):
    model = model_for(seed, initializer_range=0.5)
    prompt = torch.tensor(PROMPTS[0])
    expected = plain_greedy(model, prompt, 14)
    assert varied(expected[0, prompt.shape[1]:].tolist()), "reference output is degenerate"
    seq, _ = self_speculative_generate(model, prompt, 14, draft_loops=draft_loops, draft_len=draft_len)
    assert seq.tolist() == expected.tolist()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_equivalence_at_default_init(seed, prompt):
    # Default init (0.02). Output may collapse to one repeated token. Equality must still be exact.
    model = model_for(seed, initializer_range=0.02)
    ids = torch.tensor(prompt)
    expected = plain_greedy(model, ids, 12)
    seq, _ = self_speculative_generate(model, ids, 12, draft_loops=2, draft_len=3)
    assert seq.tolist() == expected.tolist()


def test_equivalence_with_eos_stop():
    model = model_for(0, initializer_range=0.5)
    prompt = torch.tensor(PROMPTS[0])
    free = plain_greedy(model, prompt, 14)[0, prompt.shape[1]:].tolist()
    eos = free[4]  # a token the model really emits, so the stop path is exercised
    expected = plain_greedy(model, prompt, 14, eos_token_id=eos)
    seq, _ = self_speculative_generate(model, prompt, 14, draft_loops=2, draft_len=4, eos_token_id=eos)
    assert seq.tolist() == expected.tolist()
    assert seq[0, -1].item() == eos
    assert seq.shape[1] < prompt.shape[1] + 14


def test_equivalence_at_context_limit():
    model = model_for(0, initializer_range=0.5, max_position_embeddings=64)
    prompt = torch.tensor(PROMPTS[0])
    seq, _ = self_speculative_generate(model, prompt, 200, draft_loops=2, draft_len=4)
    assert seq.shape[1] == 64
    expected = plain_greedy(model, prompt, 200)
    assert expected.shape[1] >= seq.shape[1]
    assert seq.tolist()[0] == expected[0, : seq.shape[1]].tolist()


def test_equivalence_when_drafter_is_always_wrong():
    model = model_for(1, initializer_range=0.5)
    prompt = torch.tensor(PROMPTS[0])
    expected = plain_greedy(model, prompt, 14)
    original_forward = model.forward

    def drafter_wrong(input_ids, *args, num_loops=None, **kwargs):
        out = original_forward(input_ids, *args, num_loops=num_loops, **kwargs)
        if num_loops == 2 and out.logits is not None:
            out.logits = out.logits.roll(1, dims=-1)
        return out

    model.forward = drafter_wrong
    try:
        seq, stats = self_speculative_generate(model, prompt, 14, draft_loops=2, draft_len=4)
    finally:
        model.forward = original_forward
    assert seq.tolist() == expected.tolist()
    assert stats["accepted"] == 0


# --- Accounting --------------------------------------------------------------------------------


def test_accounting_fields_are_consistent():
    model = model_for(0, initializer_range=0.5)
    prompt = torch.tensor(PROMPTS[0])
    n_new = 14
    _, stats = self_speculative_generate(model, prompt, n_new, draft_loops=2, draft_len=3)
    for key in (
        "target_forward_passes",
        "draft_forward_passes",
        "target_tokens_fed",
        "draft_tokens_fed",
        "tokens_fed_total",
    ):
        assert key in stats and stats[key] >= 0, key
    assert stats["tokens_fed_total"] == stats["target_tokens_fed"] + stats["draft_tokens_fed"]
    assert stats["target_forward_passes"] >= stats["verifier_calls"]
    # Each target call feeds at least one token, and the prompt is fed to the target at least once.
    assert stats["target_tokens_fed"] >= prompt.shape[1]
    assert stats["target_tokens_fed"] >= stats["target_forward_passes"]
    assert stats["draft_tokens_fed"] >= 0


# --- Scaling: fed tokens must not be quadratic in generated length -----------------------------


def test_fed_tokens_grow_linearly_not_quadratically():
    model = model_for(0, initializer_range=0.5, max_position_embeddings=256)
    prompt = torch.tensor(PROMPTS[0])
    S = prompt.shape[1]
    draft_len = 3
    totals = {}
    for n_new in (16, 32, 64):
        _, stats = self_speculative_generate(model, prompt, n_new, draft_loops=2, draft_len=draft_len)
        totals[n_new] = stats["tokens_fed_total"]
        # Linear bound: prompt fed to both models once, then per generated token at most
        # (draft_len + 2) tokens across both models.
        assert stats["tokens_fed_total"] <= 2 * S + (draft_len + 2) * n_new, (n_new, stats)
    # A quadratic implementation would roughly quadruple when the length doubles. Linear is ~2.
    assert totals[64] / totals[32] < 2.6, totals
    assert totals[32] / totals[16] < 2.6, totals


def test_fed_tokens_linear_when_drafts_are_all_rejected():
    model = model_for(0, initializer_range=0.5, max_position_embeddings=256)
    prompt = torch.tensor(PROMPTS[0])
    original_forward = model.forward

    def drafter_wrong(input_ids, *args, num_loops=None, **kwargs):
        out = original_forward(input_ids, *args, num_loops=num_loops, **kwargs)
        if num_loops == 2 and out.logits is not None:
            out.logits = out.logits.roll(1, dims=-1)
        return out

    model.forward = drafter_wrong
    totals = {}
    try:
        for n_new in (16, 32, 64):
            _, stats = self_speculative_generate(model, prompt, n_new, draft_loops=2, draft_len=3)
            totals[n_new] = stats["tokens_fed_total"]
    finally:
        model.forward = original_forward
    assert totals[64] / totals[32] < 2.6, totals
    assert totals[32] / totals[16] < 2.6, totals
