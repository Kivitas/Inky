"""Quality tests for looping / reasoning modes and style controls: prefill vs cached-decode equivalence at
many loop counts, deep-loop damping, configurable loop hyperparameters, mode conditioning, and style
penalties that provably change generated tokens."""

import random

import pytest
import torch

from inky import InkyCache, InkyConfig, InkyModel
from inky.generation import _block_tokens
from inky.layers import LoopStepModulation, ModeConditioner, RecurrentCore, _deep_loop_damp
from inky.style import BadWordIndex, banned_by_bad_words, build_style_controls, get_style_profile
from test_reasoning_and_style_modes import _tiny_cfg


def _model(**kw):
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_thought_gate=True, convergence_halting=True, use_mode_conditioning=True,
                    use_additive_step_embedding=True, **kw)
    m = InkyModel(cfg).eval()
    with torch.no_grad():                       # make every loop/mode knob non-trivial
        for p in m.parameters():
            if p.abs().max() == 0:
                p.add_(torch.randn_like(p) * 0.05)
    return m


@pytest.mark.parametrize("loops", [1, 2, 4, 7, 10])
@pytest.mark.parametrize("modes", [(None, None), ("low", "formal"), ("max", "informal")])
def test_prefill_equals_cached_decode(loops, modes):
    m = _model()
    ids = torch.randint(2, 200, (2, 9))
    kw = dict(reasoning_mode=modes[0], style_mode=modes[1])
    with torch.no_grad():
        full = m(ids, num_loops=loops, **kw).logits
        cache = InkyCache(m.config)
        m(ids[:, :5], past=cache, num_loops=loops, **kw)
        steps = [m(ids[:, i:i + 1], past=cache, num_loops=loops, **kw).logits for i in range(5, 9)]
    assert torch.allclose(full[:, 5:], torch.cat(steps, 1), atol=1e-4, rtol=1e-4)


def test_deep_loop_damp_and_flag():
    assert _deep_loop_damp(3, 6) == 1.0 and _deep_loop_damp(6, 6) == 1.0
    assert _deep_loop_damp(12, 6) == 0.5
    assert _deep_loop_damp(12, 6, enabled=False) == 1.0
    mod = LoopStepModulation(4, 2)
    with torch.no_grad():
        mod.step_embedding.fill_(0.5)
    h = torch.ones(1, 1, 4)
    assert torch.allclose(mod(h, 4), h * (1 + 0.25) + 0.25)          # damped 2/4
    off = LoopStepModulation(4, 2, damp_deep=False)
    with torch.no_grad():
        off.step_embedding.fill_(0.5)
    assert torch.allclose(off(h, 4), h * 1.5 + 0.5)


def test_loop_hyperparameters_configurable():
    d = RecurrentCore(_tiny_cfg())
    assert (d.beta_min, d.beta_span) == pytest.approx((0.05, 0.90))
    c = RecurrentCore(_tiny_cfg(loop_beta_min=0.2, loop_beta_max=0.4, deep_loop_damping=False))
    h_in, h_new = torch.zeros(1, 2, 32), torch.ones(1, 2, 32)
    out = c._mix_update(h_in, h_new, step=99)                         # sigmoid(0)=0.5 -> beta=0.3, no damping
    assert torch.allclose(out, torch.full_like(out, 0.3))
    with pytest.raises(ValueError):
        _tiny_cfg(loop_beta_min=0.9, loop_beta_max=0.1)
    cfg = _tiny_cfg(loop_beta_min=0.1, style_profile_overrides={"stock_word_penalty": 5.0})
    assert InkyConfig.from_dict(cfg.to_dict()).style_profile_overrides == {"stock_word_penalty": 5.0}


def test_mode_conditioner_composition_and_errors():
    mc = ModeConditioner(8)
    h = torch.randn(2, 3, 8)
    assert torch.equal(mc(h, "max", "formal"), h)                      # identity at init
    with torch.no_grad():
        mc.reasoning_embed.normal_(); mc.style_embed.normal_()
    seq = h
    rs, rt = mc.reasoning_embed[2].chunk(2); seq = seq * (1 + rs) + rt
    ss, st = mc.style_embed[1].chunk(2); seq = seq * (1 + ss) + st
    assert torch.allclose(mc(h, "max", "informal"), seq, atol=1e-5)
    assert not torch.allclose(mc(h, "low"), mc(h, "max"))
    assert not torch.allclose(mc(h, None, "formal"), mc(h, None, "informal"))
    with pytest.raises(ValueError):
        mc(h, "bogus")
    with pytest.raises(ValueError):
        mc(h, None, "bogus")


def test_modes_behave_differently():
    m = _model()
    ids = torch.randint(2, 200, (1, 10))
    with torch.no_grad():
        lo, hi, mx = (m(ids, reasoning_mode=r) for r in ("low", "high", "max"))
    assert lo.mean_loops == 2 and hi.mean_loops == 5 and mx.mean_loops == 10
    assert not torch.allclose(lo.logits, mx.logits, atol=1e-4)
    assert not torch.allclose(lo.logits, hi.logits, atol=1e-4)
    with torch.no_grad():
        f, i = m(ids, style_mode="formal").logits, m(ids, style_mode="informal").logits
    assert not torch.allclose(f, i, atol=1e-4)


def test_halting_prior_is_mode_aware():
    m = _model()
    h = torch.randn(1, 6, m.config.hidden_size)
    lam = {r: m.halting(h, prior_steps=float(n)) for r, n in (("low", 2), ("high", 5), ("max", 10))}
    assert lam["low"].mean() > lam["high"].mean() > lam["max"].mean()
    # the generate path hands the same mode to the model that the loop count came from
    out = m.generate(ids := torch.randint(2, 200, (1, 4)), max_new_tokens=3, do_sample=False,
                     reasoning_mode="low", eos_token_id=None)
    assert out.shape[1] == ids.shape[1] + 3


def test_bad_word_index_matches_reference():
    rng = random.Random(0)
    seqs = [[rng.randrange(8) for _ in range(rng.randint(1, 4))] for _ in range(40)]
    idx = BadWordIndex(seqs)
    for _ in range(200):
        ctx = [rng.randrange(8) for _ in range(rng.randint(0, 7))]
        tail = ctx[-idx.tail_len:] if idx.tail_len else []
        assert idx.banned(tail) == banned_by_bad_words(ctx, seqs)
    assert BadWordIndex([]).banned([1, 2]) == set()


def test_style_overrides_validated():
    assert get_style_profile("formal", {"stock_word_penalty": 9.0})["stock_word_penalty"] == 9.0
    assert get_style_profile("formal")["stock_word_penalty"] == 1.5
    with pytest.raises(ValueError):
        get_style_profile("formal", {"stock_word_pnalty": 1.0})
    enc = lambda t: [ord(c) % 200 + 3 for c in t][:1] if t.strip() else [3]
    ctrl = build_style_controls("informal", encode=enc, overrides={"temperature": 0.1})
    assert ctrl["temperature"] == 0.1 and ctrl["soft_penalty_ids"]


def test_penalties_change_generated_tokens():
    m = _model()
    ids = torch.randint(2, 200, (1, 6))
    base = m.generate(ids, max_new_tokens=8, do_sample=False, eos_token_id=None,
                      no_repeat_ngram_size=0, repetition_penalty=1.0)
    first = int(base[0, 6])
    pen = m.generate(ids, max_new_tokens=8, do_sample=False, eos_token_id=None, no_repeat_ngram_size=0,
                     repetition_penalty=1.0, stock_word_penalty=1e4, soft_penalty_ids=[first])
    assert int(pen[0, 6]) != first                                       # soft penalty moved the greedy pick
    banned = m.generate(ids, max_new_tokens=8, do_sample=False, eos_token_id=None, no_repeat_ngram_size=0,
                        repetition_penalty=1.0, bad_word_ids=[[first]])
    assert first not in banned[0, 6:].tolist()
    # a bad-word *sequence* only bites after its prefix has been emitted
    seq2 = [int(base[0, 6]), int(base[0, 7])]
    two = m.generate(ids, max_new_tokens=8, do_sample=False, eos_token_id=None, no_repeat_ngram_size=0,
                     repetition_penalty=1.0, bad_word_ids=[seq2])
    assert int(two[0, 6]) == seq2[0] and int(two[0, 7]) != seq2[1]


def test_style_mode_generates_and_overrides_apply():
    m = _model()
    ids = torch.randint(2, 200, (1, 6))
    enc = lambda t: [ord(c) % 200 + 3 for c in t.strip()][:1] or [3]
    torch.manual_seed(1)
    a = m.generate(ids, max_new_tokens=6, style_mode="formal", encode=enc, eos_token_id=None)
    torch.manual_seed(1)
    b = m.generate(ids, max_new_tokens=6, style_mode="formal", encode=enc, eos_token_id=None,
                   style_overrides={"temperature": 1e-4, "top_k": 1})
    assert a.shape == b.shape == (1, 12)
    with pytest.raises(ValueError):
        m.generate(ids, max_new_tokens=2, style_mode="formal", style_overrides={"nope": 1})
    for bad in (dict(top_p=0.0), dict(top_k=-1), dict(max_new_tokens=-1), dict(repetition_penalty=0.0)):
        with pytest.raises(ValueError):
            m.generate(ids, **{"max_new_tokens": 2, **bad})


def test_zero_new_tokens_and_greedy_determinism():
    m = _model()
    ids = torch.randint(2, 200, (2, 5))
    assert torch.equal(m.generate(ids, max_new_tokens=0), ids)
    a = m.generate(ids, max_new_tokens=5, do_sample=False, reasoning_mode="high", eos_token_id=None)
    b = m.generate(ids, max_new_tokens=5, do_sample=False, reasoning_mode="high", eos_token_id=None)
    assert torch.equal(a, b)
