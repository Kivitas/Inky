"""Regression tests for CORE-10 (padding in the model), DATA-05 (blocked special ids in generate) and
TRAIN-04 (embedding and residual dropout).

Run with: python -B -m pytest tests/test_core10_data05_train04_fixes.py -q -p no:cacheprovider"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky import modeling  # noqa: E402
from inky.data import DEFAULT_RESERVED_IDS, FIRST_SPECIAL_ID  # noqa: E402
from inky.layers import pondernet_kl  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510
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


# ------------------------------------------------------------------ CORE-10: padding through the model

def test_left_padded_row_matches_the_unpadded_run_through_the_whole_model():
    model = seeded_model(tiny_config())
    real = torch.randint(2, 500, (1, 6))
    pad = 3
    # The pad ids are random, so the check also shows that pad contents do not reach real positions.
    padded = torch.cat([torch.randint(2, 500, (1, pad)), real], dim=1)
    mask = torch.tensor([[0] * pad + [1] * 6], dtype=torch.bool)
    with torch.no_grad():
        alone = model(real, num_loops=3).logits
        left = model(padded, attention_mask=mask, num_loops=3).logits
    assert torch.allclose(left[:, pad:], alone, atol=1e-4), float((left[:, pad:] - alone).abs().max())


def test_right_padded_row_matches_the_unpadded_run_through_the_whole_model():
    model = seeded_model(tiny_config())
    real = torch.randint(2, 500, (1, 6))
    padded = torch.cat([real, torch.randint(2, 500, (1, 4))], dim=1)
    mask = torch.tensor([[1] * 6 + [0] * 4], dtype=torch.bool)
    with torch.no_grad():
        alone = model(real, num_loops=3).logits
        right = model(padded, attention_mask=mask, num_loops=3).logits
    assert torch.allclose(right[:, :6], alone, atol=1e-4)


def test_mean_loops_and_halting_kl_ignore_padded_positions():
    model = seeded_model(tiny_config())
    real = torch.randint(2, 500, (1, 6))
    pad = 3
    padded = torch.cat([torch.randint(2, 500, (1, pad)), real], dim=1)
    mask = torch.tensor([[0] * pad + [1] * 6], dtype=torch.bool)
    with torch.no_grad():
        alone = model(real, return_hidden=True, num_loops=3)
        left = model(padded, return_hidden=True, attention_mask=mask, num_loops=3)
    assert abs(float(alone.mean_loops) - float(left.mean_loops)) < 1e-4


def test_loss_parts_on_a_padded_batch_match_the_unpadded_loss():
    model = seeded_model(tiny_config())
    real = torch.randint(2, 500, (1, 6))
    pad = 3
    padded = torch.cat([torch.randint(2, 500, (1, pad)), real], dim=1)
    # The last pad position predicts the first real token, so that target is ignored too (-100).
    labels = torch.cat([torch.full((1, pad + 1), -100, dtype=torch.long), real[:, 1:]], dim=1)
    mask = torch.tensor([[0] * pad + [1] * 6], dtype=torch.bool)
    with torch.no_grad():
        alone = model(real, labels=real, num_loops=3).loss_parts
        left = model(padded, labels=labels, attention_mask=mask, num_loops=3).loss_parts
    assert torch.allclose(alone["ce"], left["ce"], atol=1e-4)
    assert torch.allclose(alone["ponder_kl"], left["ponder_kl"], atol=1e-4)


def test_pondernet_kl_mask_averages_only_the_real_tokens():
    torch.manual_seed(4)
    weights = torch.softmax(torch.randn(3, 2, 5), dim=0)          # [L, B, S], sums to 1 over L
    prior = torch.tensor([0.5, 0.3, 0.2])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    per_token = (weights * (weights.clamp(min=1e-12).log() - prior.clamp(min=1e-12).log().view(-1, 1, 1))).sum(0)
    expected = torch.cat([per_token[0, :3], per_token[1]]).mean()
    assert torch.allclose(pondernet_kl(weights, prior, mask=mask), expected, atol=1e-6)
    assert torch.allclose(pondernet_kl(weights, prior), per_token.mean(), atol=1e-6)


# ------------------------------------------------------------------ DATA-05: blocked ids in generate

def test_default_blocked_ids_are_reserved_non_scratch_ids_without_eos():
    cfg = InkyConfig()
    ids = modeling._blocked_ids(cfg, None, None)
    assert cfg.eos_token_id not in ids                       # EOS stays allowed
    assert {49975, 49999} <= set(ids)                        # chat, tool, vision, thought, ... are blocked
    assert not {50000, 50001, 50002} & set(ids)              # scratch ids are not blocked
    assert ids == sorted(set(DEFAULT_RESERVED_IDS) - {50000, 50001, 50002, cfg.eos_token_id})


def test_eos_is_never_blocked_even_when_listed():
    cfg = InkyConfig()
    ids = modeling._blocked_ids(cfg, [49974, 49980], 49980)
    assert 49974 not in ids and 49980 not in ids


def test_generate_default_never_emits_a_reserved_special_id():
    # A vocabulary with the real special range, so the default blocklist is in effect.
    model = seeded_model(tiny_config(
        vocab_size=50003, scratch_open_id=50000, scratch_sep_id=50001, scratch_close_id=50002,
        eos_token_id=FIRST_SPECIAL_ID, pad_token_id=FIRST_SPECIAL_ID, hidden_size=32,
        intermediate_size=64, max_position_embeddings=64,
    ))
    prompt = torch.randint(2, 400, (1, 5))
    out = model.generate(prompt, max_new_tokens=8, do_sample=False, repetition_penalty=1.0,
                         no_repeat_ngram_size=0)
    emitted = set(out[0, 5:].tolist())
    assert not emitted & (set(range(FIRST_SPECIAL_ID + 1, 50000)))


def test_generate_blocked_token_ids_are_never_sampled():
    model = seeded_model(tiny_config())
    prompt = torch.randint(2, 500, (1, 5))
    kw = dict(do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0)
    first = int(model.generate(prompt, max_new_tokens=1, **kw)[0, -1])
    out = model.generate(prompt, max_new_tokens=6, blocked_token_ids=[first], **kw)
    assert first not in out[0, 5:].tolist()


def test_generate_keeps_eos_allowed_even_when_it_is_in_the_blocked_list():
    model = seeded_model(tiny_config())
    prompt = torch.randint(2, 500, (1, 5))
    kw = dict(do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0)
    first = int(model.generate(prompt, max_new_tokens=1, **kw)[0, -1])
    stopped = model.generate(prompt, max_new_tokens=6, eos_token_id=first, blocked_token_ids=[first], **kw)
    assert stopped.shape[1] == 6 and int(stopped[0, -1]) == first


def test_generate_blocked_token_ids_outside_the_vocabulary_are_ignored():
    model = seeded_model(tiny_config())
    prompt = torch.randint(2, 500, (1, 5))
    kw = dict(do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0)
    plain = model.generate(prompt, max_new_tokens=4, **kw)
    with_far = model.generate(prompt, max_new_tokens=4, blocked_token_ids=[10_000, 49990], **kw)
    assert torch.equal(plain, with_far)


# ------------------------------------------------------------------ TRAIN-04: dropout

@pytest.mark.parametrize("field", ["embedding_dropout", "residual_dropout"])
def test_dropout_fields_are_validated_to_zero_inclusive_one_exclusive(field):
    InkyConfig(**{field: 0.0})
    InkyConfig(**{field: 0.5})
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match=field):
            InkyConfig(**{field: bad})


def test_default_dropout_leaves_train_mode_unchanged():
    base = seeded_model(tiny_config(), train=True)
    ids = torch.randint(2, 500, (2, 7))
    with torch.no_grad():
        a = base(ids, num_loops=3).logits
        b = base(ids, num_loops=3).logits
    assert torch.equal(a, b)                       # no random draws at all when the rates are zero
    evaluated = seeded_model(tiny_config())
    with torch.no_grad():
        assert torch.allclose(a, evaluated(ids, num_loops=3).logits, atol=1e-6)


@pytest.mark.parametrize("field", ["embedding_dropout", "residual_dropout"])
def test_dropout_is_active_in_train_mode_only(field):
    cfg = tiny_config(**{field: 0.5})
    model = seeded_model(cfg, train=True)
    ids = torch.randint(2, 500, (2, 7))
    torch.manual_seed(1)
    with torch.no_grad():
        first = model(ids, num_loops=3).logits
    torch.manual_seed(2)
    with torch.no_grad():
        second = model(ids, num_loops=3).logits
    assert not torch.allclose(first, second)       # different masks in two train passes
    model.eval()
    with torch.no_grad():
        e1 = model(ids, num_loops=3).logits
        e2 = model(ids, num_loops=3).logits
    assert torch.equal(e1, e2)                     # eval mode is deterministic
    assert torch.allclose(e1, seeded_model(tiny_config())(ids, num_loops=3).logits, atol=1e-6)


def test_dropout_is_reproducible_under_a_torch_seed():
    model = seeded_model(tiny_config(embedding_dropout=0.2, residual_dropout=0.2), train=True)
    ids = torch.randint(2, 500, (2, 7))
    torch.manual_seed(7)
    with torch.no_grad():
        a = model(ids, num_loops=3).logits
    torch.manual_seed(7)
    with torch.no_grad():
        b = model(ids, num_loops=3).logits
    assert torch.equal(a, b)


def test_training_loss_with_dropout_backpropagates():
    model = seeded_model(tiny_config(embedding_dropout=0.1, residual_dropout=0.1), train=True)
    ids = torch.randint(2, 500, (2, 7))
    out = model(ids, labels=ids, num_loops=3)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert model.embed_tokens.weight.grad is not None


def test_cli_accepts_dropout_flags_and_rejects_out_of_range_values(tmp_path):
    ok = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "train.py"), "--dry-run", "--steps", "1",
         "--embedding-dropout", "0.1", "--residual-dropout", "0.1", "--out-dir", str(tmp_path / "ok")],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT),
    )
    assert ok.returncode == 0, ok.stderr
    bad = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "train.py"), "--dry-run", "--steps", "1",
         "--residual-dropout", "1.0", "--out-dir", str(tmp_path / "bad")],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT),
    )
    assert bad.returncode != 0
    assert "--residual-dropout" in bad.stderr
