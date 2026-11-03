"""Regression tests for the generation and writing-control findings (docs/critique/generation.md) and
CORE-08 (LM head recomputed in backward).

Covered: GEN-01 (scratch notes, stray markers), GEN-05 (top-k/top-p sampling without a full sort),
GEN-06 and GEN-07 (repeat_window), GEN-10 (scratch close rolls the cache back instead of re-prefilling),
GEN-12 (no_repeat_ngram_size=1 rejected), GEN-13 (budget counts prose only), CORE-08 (lm_loss_recompute).

Run with: python -B -m pytest tests/test_generation_fixes.py -q -p no:cacheprovider
"""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel, NgramBanIndex  # noqa: E402
from inky import modeling  # noqa: E402
from inky.modeling import _WindowedNgramIndex, _apply_repetition_penalty  # noqa: E402
from inky.scratch import ScratchTracker, reduce_scratch  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510
MARKERS = [OPEN, SEP, CLOSE]
D = 64


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512,
        scratch_open_id=OPEN,
        scratch_sep_id=SEP,
        scratch_close_id=CLOSE,
        hidden_size=D,
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
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def seeded_model(config: InkyConfig, train: bool = False) -> InkyModel:
    torch.manual_seed(0)
    model = InkyModel(config)
    return model.train() if train else model.eval()


def scripted_sampler(script, log=None):
    """Sampler that returns the scripted tokens in order (ignores the logits)."""
    def sample(logits, *args):
        if log is not None:
            log.append(logits.clone())
        return torch.tensor([script.pop(0)])
    return sample


def schedule_sampler(schedule, calls, log=None):
    """Returns schedule[step] when present, else the best token that is not a scratch marker."""
    def sample(logits, *args):
        step = calls[0]
        calls[0] += 1
        if log is not None:
            log.append(logits.clone())
        if step in schedule:
            return torch.tensor([schedule[step]])
        masked = logits.clone()
        masked[:, MARKERS] = float("-inf")
        return masked.argmax(dim=-1)
    return sample


# ------------------------------------------------------------------ GEN-01: scratch notes and markers

def test_gen01_critique_repro_is_a_well_formed_block():
    # The critique's repro. Under the PENCIL rule, OPEN 10 11 12 SEP 13 CLOSE is a complete block:
    # 10-12 are the notes and 13 is the text. The reduction is correct for this input, so the repro
    # is not a bug by itself (the critique's "sentence the user would keep" is a judgement about
    # intent, not about the grammar).
    assert reduce_scratch([1, 2, OPEN, 10, 11, 12, SEP, 13, CLOSE, 14], OPEN, SEP, CLOSE, start=2) == [1, 2, 13, 14]


def test_gen01_stray_open_followed_by_long_prose_does_not_delete_the_prose():
    # The real hazard: a stray open, a long run of prose with no separator, then a later separator and
    # close. Before the cap, all 300 prose tokens were deleted as "notes".
    prose = list(range(200, 500))
    tokens = [1, 2, OPEN] + prose + [SEP, 13, CLOSE, 14]
    # The stray open is unpaired once its notes pass the cap, so nothing is removed at all.
    assert reduce_scratch(tokens, OPEN, SEP, CLOSE, start=2) == tokens


def test_gen01_notes_at_the_cap_still_reduce_and_one_over_does_not():
    cap = 5
    notes = [30, 31, 32, 33, 34]
    assert reduce_scratch([OPEN] + notes + [SEP, 9, CLOSE], OPEN, SEP, CLOSE, max_note_tokens=cap) == [9]
    notes_over = notes + [35]
    tokens = [OPEN] + notes_over + [SEP, 9, CLOSE]
    # One token over the cap: the block is unpaired and every token is kept as text.
    assert reduce_scratch(tokens, OPEN, SEP, CLOSE, max_note_tokens=cap) == tokens


def test_gen01_stray_open_does_not_stop_a_later_legitimate_block_from_reducing():
    # The outer open never closes; the inner block (3 is its note) is reduced and the outer marker stays.
    tokens = [OPEN, 1, 2, OPEN, 3, SEP, 4, CLOSE, 5]
    assert reduce_scratch(tokens, OPEN, SEP, CLOSE) == [OPEN, 1, 2, 4, 5]


def test_gen01_no_close_means_nothing_is_removed():
    rng = random.Random(0)
    for _ in range(200):
        tokens = [rng.choice([1, 2, 3, OPEN, SEP]) for _ in range(rng.randint(0, 40))]
        assert reduce_scratch(tokens, OPEN, SEP, CLOSE) == tokens


def test_gen01_reduction_only_removes_tokens_and_keeps_their_order():
    rng = random.Random(1)
    for _ in range(300):
        tokens = [rng.choice([1, 2, 3, 4, OPEN, SEP, CLOSE]) for _ in range(rng.randint(0, 60))]
        out = reduce_scratch(tokens, OPEN, SEP, CLOSE, max_note_tokens=rng.choice([0, 3, 256]))
        it = iter(tokens)
        assert all(any(tok == t for t in it) for tok in out), (tokens, out)  # subsequence check


def test_gen01_banned_markers_follow_the_block_state():
    tracker = ScratchTracker(OPEN, SEP, CLOSE)
    assert tracker.banned_markers() == {SEP, CLOSE}          # nothing open: no sep, no close
    tracker.feed(OPEN)
    assert tracker.banned_markers() == {CLOSE}               # open: sep allowed, close not yet
    tracker.feed(5)
    tracker.feed(SEP)
    assert tracker.banned_markers() == {SEP}                 # sep given: close allowed, sep not again
    tracker.feed(CLOSE)
    assert tracker.banned_markers() == {SEP, CLOSE}


def test_gen01_generate_masks_separator_and_close_with_no_open_block():
    model = seeded_model(tiny_config())
    seen = []
    original = modeling._sample
    try:
        modeling._sample = scripted_sampler([OPEN, 20, SEP, 21, CLOSE, 22], log=seen)
        model.generate(torch.tensor([[3, 4]]), max_new_tokens=6, do_sample=True,
                       reduce_scratch_blocks=True, repetition_penalty=1.0, no_repeat_ngram_size=0)
    finally:
        modeling._sample = original
    assert seen[0][0, SEP] == float("-inf") and seen[0][0, CLOSE] == float("-inf")   # nothing open yet
    assert seen[1][0, CLOSE] == float("-inf") and seen[1][0, SEP] != float("-inf")   # block open
    assert seen[3][0, SEP] == float("-inf") and seen[3][0, CLOSE] != float("-inf")   # separator already given


# ------------------------------------------------------------------ GEN-05: sampling without a full sort

def old_full_vocab_probs(logits, temperature, top_k, top_p):
    """The previous sampler's distribution, kept here as the reference."""
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.softmax(logits, dim=-1)


@pytest.mark.parametrize("top_k,top_p", [(50, 0.95), (0, 0.9), (20, 1.0), (5, 0.5), (0, 1.0), (600, 0.9)])
def test_gen05_candidate_distribution_matches_the_old_sampler(top_k, top_p):
    torch.manual_seed(3)
    logits = torch.randn(2, 512) * 3.0
    for temperature in (0.8, 1.3):
        idx, probs = modeling._candidate_distribution(logits, temperature, top_k, top_p)
        new_full = torch.zeros_like(logits).scatter(-1, idx, probs)
        old_full = old_full_vocab_probs(logits, temperature, top_k, top_p)
        assert torch.allclose(new_full, old_full, atol=1e-6), (top_k, top_p)


def test_gen05_default_sampling_does_not_sort_the_full_vocabulary(monkeypatch):
    def no_full_sort(*args, **kwargs):
        raise AssertionError("torch.sort was called on the vocabulary")
    torch.manual_seed(4)
    logits = torch.randn(1, 50003)
    monkeypatch.setattr(torch, "sort", no_full_sort)
    token = modeling._sample(logits, 0.8, 50, 0.95)
    assert token.shape == (1,) and 0 <= int(token) < 50003


def test_gen05_sample_draws_only_from_the_kept_set():
    torch.manual_seed(5)
    logits = torch.randn(1, 512)
    kept = set(modeling._candidate_distribution(logits, 0.8, 10, 0.9)[0][0].tolist())
    for _ in range(200):
        assert int(modeling._sample(logits, 0.8, 10, 0.9)) in kept


# ------------------------------------------------------------------ GEN-06 and GEN-07: repeat_window

def test_gen06_repetition_penalty_window_limits_the_context():
    logits = torch.full((1, 12), 2.0)
    seq = torch.tensor([[7, 8, 9, 7]])
    windowed = _apply_repetition_penalty(logits.clone(), seq, 1.1, window=2)
    assert windowed[0, 8] == 2.0                    # outside the last 2 tokens: untouched
    assert windowed[0, 9] != 2.0 and windowed[0, 7] != 2.0
    assert windowed[0, 0] == 2.0                    # never seen
    whole = _apply_repetition_penalty(logits.clone(), seq, 1.1)
    assert whole[0, 8] != 2.0                       # default: the whole history is penalised


def test_gen07_windowed_no_repeat_matches_brute_force():
    rng = random.Random(2)
    for n in (2, 3, 4):
        for window in (None, 1, n, n + 2, 12):
            seq = [rng.randrange(4) for _ in range(80)]
            index = NgramBanIndex(n) if window is None else _WindowedNgramIndex(n, window=window)
            for L, token in enumerate(seq, start=1):
                index.update(token)
                got = index.banned()
                want = brute_banned(seq[:L], n, window)
                assert got == want, (n, window, L, got, want)


def brute_banned(seq, n, window):
    L = len(seq)
    if n <= 0 or L < n:
        return set()
    tail = tuple(seq[L - n + 1:])
    lo = 0 if window is None else max(0, L - window)
    return {seq[s + n - 1] for s in range(lo, L - n + 1) if tuple(seq[s:s + n - 1]) == tail}


def test_gen07_window_none_is_the_old_index():
    ids = [1, 2, 3, 1, 2, 3, 1]
    assert _WindowedNgramIndex(3, ids).banned() == NgramBanIndex(3, ids).banned()


def test_gen07_ngram_window_lets_old_phrases_return():
    # "1 2 3" occurred long ago. With a 4-token window it is outside the window, so 3 is allowed again.
    ids = [1, 2, 3, 9, 9, 9, 9, 1, 2]
    assert _WindowedNgramIndex(3, ids, window=4).banned() == set()
    assert _WindowedNgramIndex(3, ids, window=100).banned() == {3}


def test_gen07_generate_accepts_repeat_window_and_default_is_unchanged():
    model = seeded_model(tiny_config())
    prompt = torch.tensor([[2, 3, 4]])
    kwargs = dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=3)
    default = model.generate(prompt, **kwargs)
    explicit_none = model.generate(prompt, repeat_window=None, **kwargs)
    assert torch.equal(default, explicit_none)
    windowed = model.generate(prompt, repeat_window=4, **kwargs)
    assert windowed.shape == default.shape


def test_repeat_window_must_be_positive():
    model = seeded_model(tiny_config())
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[2, 3]]), max_new_tokens=1, repeat_window=0)


# ------------------------------------------------------------------ GEN-10: rollback instead of re-prefill

def reference_generate(model, prompt, schedule, n_steps, num_loops):
    """Naive decoding: a full uncached forward pass at every step, reduction on the whole sequence.

    Returns the final sequence and the raw logits used at each step.
    """
    seq = list(prompt)
    plen = len(seq)
    logs = []
    for step in range(n_steps):
        logits = model(torch.tensor([seq]), num_loops=num_loops, last_only=True).logits[0, -1]
        logs.append(logits.clone())
        if step in schedule:
            tok = schedule[step]
        else:
            masked = logits.clone()
            masked[MARKERS] = float("-inf")
            tok = int(masked.argmax())
        seq.append(tok)
        if tok == CLOSE:
            seq = reduce_scratch(seq, OPEN, SEP, CLOSE, start=plen)
    return seq, logs


def run_with_rollback_check(monkeypatch, schedule, n_steps):
    model = seeded_model(tiny_config())
    prompt = [3, 4, 5]
    restores = []
    original_restore = modeling._cache_restore

    def counting_restore(cache, snap):
        restores.append(snap["seq_len"])
        return original_restore(cache, snap)

    fed = []
    original_forward = model.forward

    def recording_forward(input_ids, *args, **kwargs):
        if kwargs.get("past") is not None:
            fed.append(int(input_ids.shape[1]))
        return original_forward(input_ids, *args, **kwargs)

    gen_logs = []
    monkeypatch.setattr(modeling, "_cache_restore", counting_restore)
    monkeypatch.setattr(model, "forward", recording_forward)
    monkeypatch.setattr(modeling, "_sample", schedule_sampler(schedule, [0], gen_logs))
    out = model.generate(torch.tensor([prompt]), max_new_tokens=n_steps, do_sample=True,
                         num_loops=2, reduce_scratch_blocks=True, repetition_penalty=1.0,
                         no_repeat_ngram_size=0)
    ref, ref_logs = reference_generate(seeded_model(tiny_config()), prompt, schedule, n_steps, num_loops=2)
    return out[0].tolist(), ref, gen_logs, ref_logs, restores, fed


def assert_logits_match(gen_logs, ref_logs):
    """Per step, the logits must match the naive full forward pass (markers are masked on both sides)."""
    assert len(gen_logs) == len(ref_logs)
    for step, (g, r) in enumerate(zip(gen_logs, ref_logs)):
        g = g.reshape(-1)
        r = r.reshape(-1)
        keep = torch.isfinite(g)
        assert torch.allclose(g[keep], r[keep], atol=1e-4), (step, float((g[keep] - r[keep]).abs().max()))


def test_gen10_rollback_matches_full_recompute(monkeypatch):
    schedule = {0: OPEN, 1: 20, 2: 21, 3: SEP, 4: 22, 5: CLOSE, 7: OPEN, 8: 23, 9: SEP, 10: 24, 11: CLOSE}
    got, ref, gen_logs, ref_logs, restores, fed = run_with_rollback_check(monkeypatch, schedule, n_steps=16)
    assert got == ref
    assert got[:4] == [3, 4, 5, 22]
    assert len(restores) == 2                   # one rollback per closed block
    assert_logits_match(gen_logs, ref_logs)     # the logits after each close are the real ones, not stale


def test_gen10_only_the_answer_is_recomputed_after_a_close(monkeypatch):
    schedule = {0: OPEN, 1: 20, 2: 21, 3: SEP, 4: 22, 5: CLOSE}
    got, ref, gen_logs, ref_logs, restores, fed = run_with_rollback_check(monkeypatch, schedule, n_steps=8)
    assert got == ref
    assert_logits_match(gen_logs, ref_logs)
    # fed[0] is the prompt prefill (3 tokens). After that, every cached forward is one token: the
    # rollback recomputes only the answer (22), never the prompt again.
    assert fed[0] == 3 and all(s == 1 for s in fed[1:]), fed


def test_gen10_nested_blocks_match_full_recompute(monkeypatch):
    schedule = {0: OPEN, 1: 20, 2: OPEN, 3: 21, 4: SEP, 5: 22, 6: CLOSE, 7: 23, 8: SEP, 9: 24, 10: CLOSE}
    got, ref, gen_logs, ref_logs, restores, fed = run_with_rollback_check(monkeypatch, schedule, n_steps=14)
    assert got == ref
    assert len(restores) == 2
    assert_logits_match(gen_logs, ref_logs)
    assert fed[0] == 3 and all(s == 1 for s in fed[1:]), fed


# ------------------------------------------------------------------ GEN-12: no_repeat_ngram_size=1

def test_gen12_no_repeat_ngram_size_one_is_rejected():
    model = seeded_model(tiny_config())
    with pytest.raises(ValueError, match="no_repeat_ngram_size=1"):
        model.generate(torch.tensor([[2, 3]]), max_new_tokens=1, no_repeat_ngram_size=1)
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[2, 3]]), max_new_tokens=1, no_repeat_ngram_size=-1)


def test_gen12_zero_and_two_are_still_accepted():
    model = seeded_model(tiny_config())
    assert model.generate(torch.tensor([[2, 3]]), max_new_tokens=2, do_sample=False,
                          no_repeat_ngram_size=0, repetition_penalty=1.0).shape == (1, 4)
    assert model.generate(torch.tensor([[2, 3]]), max_new_tokens=2, do_sample=False,
                          no_repeat_ngram_size=2, repetition_penalty=1.0).shape == (1, 4)


# ------------------------------------------------------------------ GEN-13: budget counts prose

def test_gen13_budget_counts_prose_not_notes(monkeypatch):
    script = [OPEN, 20, SEP, 21, CLOSE, 22, 23, 24]
    monkeypatch.setattr(modeling, "_sample", scripted_sampler(list(script)))
    model = seeded_model(tiny_config())
    out = model.generate(torch.tensor([[3, 4]]), max_new_tokens=3, do_sample=True,
                         reduce_scratch_blocks=True, count_prose_only=True,
                         repetition_penalty=1.0, no_repeat_ngram_size=0)
    assert out.tolist() == [[3, 4, 21, 22, 23]]          # three prose tokens, notes not counted


def test_gen13_default_budget_still_counts_every_token(monkeypatch):
    script = [OPEN, 20, SEP, 21, CLOSE, 22, 23, 24]
    monkeypatch.setattr(modeling, "_sample", scripted_sampler(list(script)))
    model = seeded_model(tiny_config())
    out = model.generate(torch.tensor([[3, 4]]), max_new_tokens=3, do_sample=True,
                         reduce_scratch_blocks=True, repetition_penalty=1.0, no_repeat_ngram_size=0)
    assert out.tolist() == [[3, 4, OPEN, 20, SEP]]


def test_gen13_prose_count_excludes_open_notes():
    tracker = ScratchTracker(OPEN, SEP, CLOSE, prefix=[3, 4])
    for tok in [7, OPEN, 20, 21]:
        tracker.feed(tok)
    assert tracker.prose_count(2) == 1                    # 7 is prose; the open block's notes are not
    tracker.feed(SEP)
    tracker.feed(22)
    assert tracker.prose_count(2) == 2                    # 22 is the answer, so it counts as prose
    tracker.feed(CLOSE)
    assert tracker.out == [3, 4, 7, 22] and tracker.prose_count(2) == 2


# ------------------------------------------------------------------ CORE-08: LM head recompute

def head_grads(model, hidden, labels, recompute):
    model.config.lm_loss_recompute = recompute
    model.zero_grad(set_to_none=True)
    h = hidden.clone().requires_grad_(True)
    calls = []
    original = model._project

    def counting(x):
        calls.append(1)
        return original(x)

    model._project = counting
    try:
        ce, z = model._lm_loss(h, labels)
        (ce + 1e-2 * z).backward()
    finally:
        del model._project
    return len(calls), h.grad.clone(), model.embed_tokens.weight.grad.clone(), float(ce.detach())


def test_core08_recompute_flag_default_is_on():
    assert InkyConfig().lm_loss_recompute is True


def test_core08_without_recompute_the_head_runs_once_and_gradients_match():
    model = seeded_model(tiny_config(), train=True)
    torch.manual_seed(7)
    hidden = torch.randn(2, 9, D)
    labels = torch.randint(2, 500, (2, 9))
    n_on, gh_on, gw_on, ce_on = head_grads(model, hidden, labels, recompute=True)
    n_off, gh_off, gw_off, ce_off = head_grads(model, hidden, labels, recompute=False)
    assert n_on == 2 * n_off == 2                         # forward plus the recompute in backward
    assert abs(ce_on - ce_off) < 1e-6
    assert torch.allclose(gh_on, gh_off, atol=1e-6)
    assert torch.allclose(gw_on, gw_off, atol=1e-6)


def test_core08_flag_is_respected_by_the_step_loss():
    model = seeded_model(tiny_config(), train=True)
    torch.manual_seed(8)
    states = [torch.randn(2, 9, D, requires_grad=True) for _ in range(2)]
    weights = torch.softmax(torch.randn(2, 2, 9), dim=0)
    labels = torch.randint(2, 500, (2, 9))
    model.config.lm_loss_recompute = False
    model._weighted_step_ce(states, weights, labels).backward()
    assert all(s.grad is not None for s in states)


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items())
             if name.startswith("test_") and "monkeypatch" not in value.__code__.co_varnames[:value.__code__.co_argcount]]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
