"""Structural tests for Inky. Run with: python -m pytest tests -q   (or: python tests/test_inky.py)"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky import modeling  # noqa: E402
from inky.scratch import reduce_scratch  # noqa: E402
from inky.style import banned_by_bad_words, banned_by_no_repeat_ngram, build_bad_word_ids  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510  # scratch ids inside the tiny vocabulary


def tiny_config(**overrides) -> InkyConfig:
    """Same structure as the default config, at a size that runs in well under a second on CPU."""
    kwargs = dict(
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
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def seeded_model(config: InkyConfig) -> InkyModel:
    torch.manual_seed(0)
    model = InkyModel(config)
    model.eval()
    return model


# Exact counts, kept here in one place (TEST-03). README: 50.22M default, 50.61M with the decision engine.
EXPECTED_DEFAULT_PARAMS = 50_219_201
EXPECTED_ARLO_PARAMS = 50_614_854


def test_default_config_parameter_count_is_exact():
    model = InkyModel(InkyConfig())
    assert model.count_parameters()["total"] == EXPECTED_DEFAULT_PARAMS


def test_arlo_parameter_count_is_exact():
    model = InkyModel(InkyConfig(enable_arlo=True))
    assert model.count_parameters()["total"] == EXPECTED_ARLO_PARAMS


def test_default_context_window_is_8k():
    assert InkyConfig().max_position_embeddings == 8192


def test_forward_returns_finite_loss_and_gradients():
    model = seeded_model(tiny_config())
    model.train()
    ids = torch.randint(0, 512, (2, 24))
    out = model(ids, labels=ids, num_loops=3)
    assert out.logits is None
    assert torch.isfinite(out.loss)
    out.loss.backward()
    # The decision engine is not in the language-model loss, so it has no gradient here by design.
    missing = [name for name, p in model.named_parameters()
               if p.grad is None and not name.startswith("arlo.")]
    assert not missing, missing


def test_causality_future_tokens_do_not_change_past_logits():
    model = seeded_model(tiny_config())
    a = torch.randint(2, 512, (1, 20))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 7) % 510 + 2
    with torch.no_grad():
        logits_a = model(a, num_loops=3).logits
        logits_b = model(b, num_loops=3).logits
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1])


def test_cached_decoding_matches_full_forward():
    model = seeded_model(tiny_config())
    ids = torch.randint(2, 512, (2, 18))
    with torch.no_grad():
        full = model(ids, num_loops=4).logits

        prefix = 10
        out = model(ids[:, :prefix], use_cache=True, num_loops=4)
        cache = out.past
        step_logits = [out.logits]
        for t in range(prefix, ids.shape[1]):
            out = model(ids[:, t:t + 1], past=cache)
            step_logits.append(out.logits)
        incremental = torch.cat(step_logits, dim=1)

    prefill_only = full[:, :prefix]
    assert torch.allclose(prefill_only, incremental[:, :prefix], atol=1e-4)
    assert torch.allclose(full[:, prefix:], incremental[:, prefix:], atol=1e-4)


def test_cache_loop_count_is_fixed_after_prefill():
    model = seeded_model(tiny_config())
    ids = torch.randint(2, 512, (1, 6))
    cache = model(ids, use_cache=True, num_loops=3).past
    try:
        model(ids[:, :1], past=cache, num_loops=4)
    except ValueError:
        return
    raise AssertionError("changing the loop count mid-sequence should raise")


def test_context_limit_is_enforced():
    model = seeded_model(tiny_config(max_position_embeddings=32))
    ids = torch.randint(2, 512, (1, 33))
    try:
        model(ids)
    except ValueError:
        return
    raise AssertionError("sequences longer than max_position_embeddings should raise")


def test_mean_loops_is_within_bounds_and_follows_the_halting_gate():
    # TEST-03: the halting gate must move the loop count, not only stay inside [1, L].
    config = tiny_config()
    model = seeded_model(config)
    ids = torch.randint(2, 512, (2, 12))
    L = config.max_recurrent_loops
    with torch.no_grad():
        default = float(model(ids, num_loops=L).mean_loops)
        assert 1.0 <= default <= L
        model.halting.gate.bias.fill_(50.0)      # stop probability ~1: almost every token stops at loop 1
        stop_early = float(model(ids, num_loops=L).mean_loops)
        model.halting.gate.bias.fill_(-50.0)     # stop probability ~0: almost every token runs all loops
        stop_late = float(model(ids, num_loops=L).mean_loops)
    assert stop_early < 1.05, stop_early
    assert stop_late > L - 0.05, stop_late


def test_generate_appends_tokens_and_stops_at_eos():
    config = tiny_config()
    model = seeded_model(config)
    prompt = torch.randint(2, 512, (1, 5))
    kw = dict(do_sample=False, repetition_penalty=1.0)
    out = model.generate(prompt, max_new_tokens=8, **kw)
    assert out.shape == (1, 13)
    assert out[:, :5].tolist() == prompt.tolist()
    # Value check: the first new token is the argmax of a full forward pass over the prompt.
    assert int(out[0, 5]) == int(model(prompt).logits[0, -1].argmax())
    assert int(out.max()) < config.vocab_size
    # DOC-05: set the EOS to the token the model emits first. Generation must stop right after it.
    eos = int(out[0, 5])
    stopped = model.generate(prompt, max_new_tokens=8, eos_token_id=eos, **kw)
    assert stopped.shape == (1, 6), stopped.shape
    assert int(stopped[0, -1]) == eos


def test_moe_balance_update_moves_bias_towards_uniform_load():
    # TEST-03: the sign of each bias change follows (target load - observed load).
    from inky.layers import SharedExpertMoE

    moe = SharedExpertMoE(tiny_config())
    with torch.no_grad():
        moe.load_counts.copy_(torch.tensor([9.0, 1.0, 0.0, 0.0]))   # expert 0 overloaded, 2 and 3 idle
    moe.update_balance_bias()
    bias = moe.balance_bias
    assert float(bias[0]) < 0.0                  # overloaded: pushed down
    assert float(bias[1]) > 0.0 and float(bias[2]) > 0.0 and float(bias[3]) > 0.0   # under target: pushed up
    assert float(moe.load_counts.sum()) == 0.0   # counts reset after the update

    # Model level: a real training step moves some bias, and every bias stays finite.
    model = seeded_model(tiny_config())
    model.train()
    ids = torch.randint(2, 512, (2, 16))
    model(ids, labels=ids, num_loops=2).loss.backward()
    model.update_moe_balance()
    biases = [m.balance_bias for m in model.modules() if hasattr(m, "update_balance_bias")]
    assert biases and all(torch.isfinite(b).all() for b in biases)
    assert any(float(b.abs().sum()) > 0 for b in biases)


def test_gradient_checkpointing_trains():
    model = seeded_model(tiny_config(gradient_checkpointing=True))
    model.train()
    ids = torch.randint(2, 512, (2, 16))
    loss = model(ids, labels=ids, num_loops=2).loss
    loss.backward()
    # TEST-03: every parameter the loss depends on must have a gradient, and all gradients are finite.
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_save_and_reload_round_trip():
    import tempfile

    model = seeded_model(tiny_config())
    ids = torch.randint(2, 512, (1, 8))
    with torch.no_grad():
        before = model(ids, num_loops=3).logits
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = InkyModel.from_pretrained(tmp).eval()
        with torch.no_grad():
            after = reloaded(ids, num_loops=3).logits
    assert torch.allclose(before, after, atol=1e-6)


def test_reduction_removes_notes_and_keeps_text():
    # C [CALL] T [SEP] A [RETURN]  =>  C A
    assert reduce_scratch([1, OPEN, 5, 6, SEP, 7, 8, CLOSE, 9], OPEN, SEP, CLOSE) == [1, 7, 8, 9]


def test_reduction_handles_nested_blocks():
    # An inner block in the notes goes with the notes; the outer block then reduces.
    tokens = [OPEN, 1, OPEN, 2, SEP, 3, CLOSE, 4, SEP, 5, CLOSE]
    assert reduce_scratch(tokens, OPEN, SEP, CLOSE) == [5]
    # A block that sits inside the text of an outer block is reduced on its own.
    tokens = [OPEN, 1, SEP, 2, OPEN, 3, SEP, 4, CLOSE, 5, CLOSE]
    assert reduce_scratch(tokens, OPEN, SEP, CLOSE) == [2, 4, 5]
    # Text that follows an inner block in the outer text is kept; the outer notes ("1") are removed.
    tokens = [OPEN, 1, SEP, OPEN, 2, SEP, 3, CLOSE, 4, CLOSE]
    assert reduce_scratch(tokens, OPEN, SEP, CLOSE) == [3, 4]


def test_reduction_keeps_prompt_and_unclosed_blocks():
    prompt = [OPEN, 1, SEP, 2, CLOSE]           # markers in the prompt are never edited
    assert reduce_scratch(prompt + [OPEN, 3, SEP, 4], OPEN, SEP, CLOSE, start=len(prompt)) == prompt + [OPEN, 3, SEP, 4]
    assert reduce_scratch(prompt + [OPEN, 3, SEP, 4, CLOSE], OPEN, SEP, CLOSE, start=len(prompt)) == prompt + [4]


def test_no_repeat_ngram_blocks_completed_repeats():
    assert banned_by_no_repeat_ngram([1, 2, 3, 1, 2], 3) == {3}
    assert banned_by_no_repeat_ngram([1, 2, 3], 4) == set()


def test_bad_word_sequences_block_their_last_token_only_after_prefix():
    assert banned_by_bad_words([1, 7], [[7, 8]]) == {8}
    assert banned_by_bad_words([1, 2], [[7, 8]]) == set()
    assert banned_by_bad_words([1, 2], [[9]]) == {9}


def test_build_bad_word_ids_covers_case_and_space_variants():
    seqs = build_bad_word_ids(lambda s: [ord(c) for c in s], phrases=["delve"])
    assert [ord(c) for c in "delve"] in seqs
    assert [ord(c) for c in " Delve"] in seqs


def test_generate_reduces_scratch_and_keeps_prompt(monkeypatch):
    # Script the sampler: notes (5), separator, text (7), close, then one more text token (9).
    # After the close, the notes and markers must be gone and the text (7, 9) must remain.
    script = [OPEN, 5, SEP, 7, CLOSE, 9]
    monkeypatch.setattr(modeling, "_sample", lambda logits, *args: torch.tensor([script.pop(0)]))
    model = seeded_model(tiny_config())
    prompt = torch.tensor([[3, 4]])
    out = model.generate(prompt, max_new_tokens=6, do_sample=True, reduce_scratch_blocks=True,
                         repetition_penalty=1.0, no_repeat_ngram_size=0)
    assert out.tolist() == [[3, 4, 7, 9]]


def test_scratch_reduction_requires_batch_size_one():
    model = seeded_model(tiny_config())
    try:
        model.generate(torch.zeros(2, 3, dtype=torch.long), max_new_tokens=1, reduce_scratch_blocks=True)
    except ValueError:
        return
    raise AssertionError("batched scratch reduction should raise")


def test_scratch_rebuild_keeps_the_loop_count(monkeypatch):
    # The cache is rebuilt after a scratch close; every loop-count decision must still be 2.
    script = [OPEN, 5, SEP, 7, CLOSE, 9]
    monkeypatch.setattr(modeling, "_sample", lambda logits, *args: torch.tensor([script.pop(0)]))
    model = seeded_model(tiny_config())
    chosen = []
    original = model._choose_loops

    def recording(cache, num_loops, *args, **kwargs):
        loops = original(cache, num_loops, *args, **kwargs)
        chosen.append(loops)
        return loops

    monkeypatch.setattr(model, "_choose_loops", recording)
    out = model.generate(torch.tensor([[3, 4]]), max_new_tokens=6, do_sample=True, num_loops=2,
                         reduce_scratch_blocks=True, repetition_penalty=1.0, no_repeat_ngram_size=0)
    assert out.tolist() == [[3, 4, 7, 9]]
    assert len(chosen) >= 2 and chosen == [2] * len(chosen), chosen


def test_bad_words_are_never_generated():
    # Ban the token the model would otherwise pick first. Without the ban it is chosen, so a
    # missing or broken ban shows up as the same first token coming back.
    model = seeded_model(tiny_config(initializer_range=0.5))
    prompt = torch.tensor([[2, 3]])
    free = model.generate(prompt, max_new_tokens=8, do_sample=False,
                          repetition_penalty=1.0, no_repeat_ngram_size=0)
    banned_token = int(free[0, prompt.shape[1]])
    out = model.generate(prompt, max_new_tokens=8, do_sample=False,
                         repetition_penalty=1.0, no_repeat_ngram_size=0,
                         bad_word_ids=[[banned_token]])
    generated = out[0, prompt.shape[1]:].tolist()
    assert generated[0] != banned_token
    assert banned_token not in generated


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
