"""Tests for self-speculative greedy decoding. Run with: python -m pytest tests/test_speculative.py -q"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.speculative import self_speculative_generate  # noqa: E402


def tiny_config() -> InkyConfig:
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
        max_position_embeddings=256,
        arlo_dim=16,
        arlo_pointer_rank=8,
        arlo_slots=4,
        eos_token_id=1,
        pad_token_id=1,
        # The default 0.02 init collapses greedy output to one repeated token. 0.5 gives varied output.
        initializer_range=0.5,
    )


def seeded_model() -> InkyModel:
    torch.manual_seed(0)
    model = InkyModel(tiny_config())
    model.eval()
    return model


def plain_greedy(model: InkyModel, input_ids: torch.Tensor, n: int) -> torch.Tensor:
    return model.generate(
        input_ids,
        max_new_tokens=n,
        do_sample=False,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_loops=model.config.max_recurrent_loops,
    )


def assert_varied(generated: torch.Tensor) -> None:
    # Guard against a degenerate reference: a greedy output that repeats one token makes every
    # acceptance check trivially pass, so the test would not detect a broken decoder.
    distinct = len(set(generated.tolist()))
    assert distinct >= 4, f"greedy output has only {distinct} distinct tokens: {generated.tolist()}"


def test_matches_plain_greedy_with_two_draft_loops():
    model = seeded_model()
    prompt = torch.tensor([[5, 17, 42, 99, 7]])
    expected = plain_greedy(model, prompt, 12)
    assert_varied(expected[0, prompt.shape[1]:])
    seq, stats = self_speculative_generate(model, prompt, max_new_tokens=12, draft_loops=2, draft_len=3)
    assert seq.tolist() == expected.tolist()


def test_stats_keys_and_range():
    model = seeded_model()
    prompt = torch.tensor([[5, 17, 42, 99, 7]])
    _, stats = self_speculative_generate(model, prompt, max_new_tokens=12, draft_loops=2, draft_len=3)
    for key in ("drafted", "accepted", "acceptance_rate", "verifier_calls"):
        assert key in stats
    assert 0.0 <= stats["acceptance_rate"] <= 1.0
    assert 0 <= stats["accepted"] <= stats["drafted"]
    assert stats["verifier_calls"] >= 1
    print(f"\nacceptance_rate={stats['acceptance_rate']:.3f} stats={stats}")


def test_acceptance_is_below_one_when_drafter_is_perturbed():
    # The drafter's logits get a deterministic perturbation on every second drafter call, large
    # enough to flip some argmaxes. Some drafts must then be rejected, so the acceptance rate is
    # strictly below 1.0, and the output must still equal plain greedy.
    model = seeded_model()
    prompt = torch.tensor([[5, 17, 42, 99, 7]])
    expected = plain_greedy(model, prompt, 12)
    assert_varied(expected[0, prompt.shape[1]:])
    original_forward = model.forward
    draft_loops = 2
    calls = {"drafter": 0}

    def drafter_perturbed(input_ids, *args, num_loops=None, **kwargs):
        out = original_forward(input_ids, *args, num_loops=num_loops, **kwargs)
        if num_loops == draft_loops and out.logits is not None:
            calls["drafter"] += 1
            if calls["drafter"] % 2 == 0:
                idx = torch.arange(out.logits.shape[-1], dtype=out.logits.dtype)
                out.logits = out.logits + 2.0 * torch.sin(idx * 0.37 + calls["drafter"])
        return out

    model.forward = drafter_perturbed
    seq, stats = self_speculative_generate(model, prompt, max_new_tokens=12, draft_loops=draft_loops, draft_len=3)
    model.forward = original_forward
    assert calls["drafter"] > 0
    assert seq.tolist() == expected.tolist()
    assert stats["drafted"] > 0
    assert stats["acceptance_rate"] < 1.0, stats


def test_matches_plain_greedy_with_one_draft_loop_and_longer_draft():
    model = seeded_model()
    prompt = torch.tensor([[5, 17, 42, 99, 7]])
    expected = plain_greedy(model, prompt, 12)
    assert_varied(expected[0, prompt.shape[1]:])
    seq, _ = self_speculative_generate(model, prompt, max_new_tokens=12, draft_loops=1, draft_len=5)
    assert seq.tolist() == expected.tolist()


def test_rejection_path_is_exercised_and_still_matches_greedy():
    # Make every drafter prediction wrong: shift the drafter's logits so its argmax never equals
    # the verifier's. The verifier is untouched, so the output must still equal plain greedy, and
    # no draft may be accepted.
    model = seeded_model()
    prompt = torch.tensor([[5, 17, 42, 99, 7]])
    expected = plain_greedy(model, prompt, 12)
    original_forward = model.forward
    draft_loops = 2

    def drafter_wrong(input_ids, *args, num_loops=None, **kwargs):
        out = original_forward(input_ids, *args, num_loops=num_loops, **kwargs)
        if num_loops == draft_loops and out.logits is not None:
            out.logits = out.logits.roll(1, dims=-1)
        return out

    model.forward = drafter_wrong
    seq, stats = self_speculative_generate(model, prompt, max_new_tokens=12, draft_loops=draft_loops, draft_len=3)
    assert seq.tolist() == expected.tolist()
    assert stats["drafted"] > 0
    assert stats["accepted"] == 0
    assert stats["acceptance_rate"] == 0.0
