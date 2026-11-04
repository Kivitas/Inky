"""Round-2 review fixes: ARLO-01 flag survives reload/deepcopy and warns at the caller, and the EOS sentinel.

Run with: python -m pytest tests/test_review_round2.py -q -p no:cacheprovider
"""

import copy
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.arlo import train_arlo  # noqa: E402
from inky.speculative import self_speculative_generate  # noqa: E402


def tiny_config(**overrides) -> InkyConfig:
    kw = dict(
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
        arlo_heads=4,
        enable_arlo=True,
        eos_token_id=1,
        pad_token_id=1,
        initializer_range=0.5,
    )
    kw.update(overrides)
    return InkyConfig(**kw)


def seeded(**overrides) -> InkyModel:
    torch.manual_seed(0)
    model = InkyModel(tiny_config(**overrides))
    model.eval()
    return model


CTX = torch.arange(2, 8).unsqueeze(0)


def tiny_binary_batches():
    return [[{"context": CTX, "gold": 1}, {"context": CTX + 1, "gold": 0}]]


def untrained_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if "decision engine" in str(w.message)]


def decide_all(model):
    model.decide_binary(CTX)
    model.decide_choice(CTX, [torch.tensor([[2, 3]]), torch.tensor([[4]])])


def test_untrained_warning_attributed_to_caller_file():
    model = seeded()
    opts = [torch.tensor([[2]]), torch.tensor([[3]])]
    for call in (lambda: model.decide_binary(CTX), lambda: model.decide_choice(CTX, opts)):
        with pytest.warns(UserWarning, match="decision engine") as rec:
            call()
        assert Path(rec[0].filename).resolve() == Path(__file__).resolve()


def test_arlo_trained_flag_survives_from_pretrained(tmp_path):
    model = seeded()
    model.save_pretrained(tmp_path)
    loaded = InkyModel.from_pretrained(tmp_path)
    loaded.eval()
    assert untrained_warnings(lambda: decide_all(loaded))      # a reloaded model starts untrained
    train_arlo(loaded, tiny_binary_batches(), task="binary", epochs=1)
    assert loaded.arlo_trained
    assert untrained_warnings(lambda: decide_all(loaded)) == []


def test_arlo_trained_flag_survives_deepcopy():
    model = seeded()
    clone = copy.deepcopy(model)
    train_arlo(clone, tiny_binary_batches(), task="binary", epochs=1)
    assert untrained_warnings(lambda: decide_all(clone)) == []
    clone2 = copy.deepcopy(clone)
    assert untrained_warnings(lambda: decide_all(clone2)) == []
    assert untrained_warnings(lambda: decide_all(seeded()))     # the original, untrained, still warns


GREEDY = dict(do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0)


def _first_token(model, prompt):
    return int(model.generate(prompt, max_new_tokens=1, eos_token_id=None, **GREEDY)[0, -1])


def test_generate_eos_sentinel_none_and_int():
    model = seeded()
    prompt = torch.arange(2, 8).unsqueeze(0)
    first = _first_token(model, prompt)
    model.config.eos_token_id = first
    # Default (sentinel): stops at config EOS right after the first token.
    default = model.generate(prompt, max_new_tokens=8, **GREEDY)
    assert default.shape == (1, 7) and int(default[0, -1]) == first
    # Explicit None: never stops at EOS.
    never = model.generate(prompt, max_new_tokens=8, eos_token_id=None, **GREEDY)
    assert never.shape == (1, 14)
    # Explicit int: stops at that id, not at the configured one.
    other = int(never[0, 8])
    if other != first:
        stopped = model.generate(prompt, max_new_tokens=8, eos_token_id=other, **GREEDY)
        assert int(stopped[0, -1]) == other and stopped.shape[1] <= 14
    # An int that is never emitted overrides the config EOS, so generation runs to the budget.
    unused = next(t for t in range(2, 500) if t not in never[0].tolist() and t != first)
    assert model.generate(prompt, max_new_tokens=8, eos_token_id=unused, **GREEDY).shape == (1, 14)


def test_speculative_follows_generate_eos_rule():
    model = seeded()
    prompt = torch.arange(2, 8).unsqueeze(0)
    loops = model.config.max_recurrent_loops
    first = _first_token(model, prompt)
    model.config.eos_token_id = first
    kw = dict(num_loops=loops, **GREEDY)

    ref = model.generate(prompt, max_new_tokens=10, **kw)
    assert ref.shape == (1, 7)                                   # the configured EOS really is emitted
    out, _ = self_speculative_generate(model, prompt, 10)
    assert torch.equal(out, ref)

    ref_none = model.generate(prompt, max_new_tokens=10, eos_token_id=None, **kw)
    out_none, _ = self_speculative_generate(model, prompt, 10, eos_token_id=None)
    assert ref_none.shape == (1, 16) and torch.equal(out_none, ref_none)

    later = int(ref_none[0, 9])
    ref_int = model.generate(prompt, max_new_tokens=10, eos_token_id=later, **kw)
    out_int, _ = self_speculative_generate(model, prompt, 10, eos_token_id=later)
    assert torch.equal(out_int, ref_int)
