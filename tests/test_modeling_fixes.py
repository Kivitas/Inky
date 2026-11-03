"""Regression tests for inky/modeling.py: halting-gate gradients, loop statistics, seeded loop depth,
packing (segment_ids / attention_mask), step-CE token subsetting, MoE aux averaging, loss parts,
incremental generation, and config defaults.

Run with: python -m pytest tests/test_modeling_fixes.py -q -p no:cacheprovider
"""

import math
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inky  # noqa: E402
from inky import InkyConfig, InkyModel, NgramBanIndex, soft_penalty_token_ids  # noqa: E402
from inky import modeling  # noqa: E402
from inky.layers import SharedExpertMoE  # noqa: E402
from inky.modeling import InkyCache  # noqa: E402
from inky.style import banned_by_bad_words, banned_by_no_repeat_ngram  # noqa: E402

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


# ---------------------------------------------------------------- 1. halting gate and step CE

def test_step_ce_does_not_send_gradient_into_the_halting_weights():
    model = seeded_model(tiny_config())
    L, B, S = 3, 2, 9
    torch.manual_seed(1)
    states = [torch.randn(B, S, D, requires_grad=True) for _ in range(L)]
    raw = torch.randn(L, B, S, requires_grad=True)
    weights = torch.softmax(raw, dim=0)
    labels = torch.randint(2, 500, (B, S))
    model._weighted_step_ce(states, weights, labels).backward()
    assert raw.grad is None
    assert all(s.grad is not None and float(s.grad.abs().sum()) > 0 for s in states)


def test_halting_gate_still_trains_from_the_main_loss():
    model = seeded_model(tiny_config(), train=True)
    ids = torch.randint(2, 500, (2, 10))
    model(ids, labels=ids, num_loops=3).loss.backward()
    grad = model.halting.gate.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0


# ---------------------------------------------------------------- 2. loop statistics

def test_mean_loops_is_the_same_quantity_in_forward_and_early_exit():
    model = seeded_model(tiny_config())
    ids = torch.randint(2, 500, (1, 12))
    with torch.no_grad():
        full = model(ids, num_loops=4)
        early = model.forward_early_exit(ids, halt_threshold=None, num_loops=4, return_hidden=True)
    assert abs(float(full.mean_loops) - float(early.mean_loops)) < 1e-5
    assert 1.0 <= float(full.mean_loops) <= 4.0
    # Without a threshold every token runs all loops, so the realised exit loop is L.
    assert float(early.mean_exit_loop) == 4.0


def test_early_exit_reports_expected_and_realised_loops_separately():
    model = seeded_model(tiny_config())
    S = 12
    ids = torch.randint(2, 500, (1, S))
    with torch.no_grad():
        out = model.forward_early_exit(ids, halt_threshold=0.5, num_loops=4, return_hidden=True)
    assert 1.0 <= float(out.mean_loops) <= 4.0
    assert 1.0 <= float(out.mean_exit_loop) <= 4.0
    # The realised exit loops are whole numbers per token, so their sum over tokens is an integer.
    total = float(out.mean_exit_loop) * S
    assert abs(total - round(total)) < 1e-4


# ---------------------------------------------------------------- 3. seeded loop depth

def test_loop_depth_follows_torch_seed_not_python_random():
    model = seeded_model(tiny_config(min_recurrent_loops=1, max_recurrent_loops=6), train=True)
    torch.manual_seed(7)
    first = [model._choose_loops(None, None) for _ in range(30)]
    torch.manual_seed(7)
    second = [model._choose_loops(None, None) for _ in range(30)]
    assert first == second
    assert all(1 <= n <= 6 for n in first) and len(set(first)) > 1

    random.seed(1)
    torch.manual_seed(3)
    a = [model._choose_loops(None, None) for _ in range(10)]
    random.seed(2)
    torch.manual_seed(3)
    b = [model._choose_loops(None, None) for _ in range(10)]
    assert a == b


def test_loop_depth_accepts_an_explicit_generator():
    model = seeded_model(tiny_config(min_recurrent_loops=1, max_recurrent_loops=6), train=True)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    a = [model._choose_loops(None, None, generator=g1) for _ in range(20)]
    b = [model._choose_loops(None, None, generator=g2) for _ in range(20)]
    assert a == b


def test_training_forward_reproduces_under_a_torch_seed():
    model = seeded_model(tiny_config(), train=True)
    ids = torch.randint(2, 500, (2, 10))
    torch.manual_seed(5)
    first = float(model(ids, labels=ids).loss.detach())
    torch.manual_seed(5)
    second = float(model(ids, labels=ids).loss.detach())
    assert first == second


def test_forward_generator_argument_makes_the_run_reproducible():
    model = seeded_model(tiny_config(), train=True)
    ids = torch.randint(2, 500, (2, 10))
    a = float(model(ids, labels=ids, generator=torch.Generator().manual_seed(9)).loss)
    b = float(model(ids, labels=ids, generator=torch.Generator().manual_seed(9)).loss)
    assert a == b


# ---------------------------------------------------------------- 4. packing

def test_packed_documents_match_documents_run_separately():
    model = seeded_model(tiny_config())
    a = torch.randint(2, 500, (1, 5))
    b = torch.randint(2, 500, (1, 4))
    packed = torch.cat([a, b], dim=1)
    segments = torch.tensor([[0] * 5 + [1] * 4])
    with torch.no_grad():
        together = model(packed, segment_ids=segments, num_loops=4).logits
        alone_a = model(a, num_loops=4).logits
        alone_b = model(b, num_loops=4).logits
    assert torch.allclose(together[:, :5], alone_a, atol=1e-4), float((together[:, :5] - alone_a).abs().max())
    assert torch.allclose(together[:, 5:], alone_b, atol=1e-4), float((together[:, 5:] - alone_b).abs().max())


def test_packing_is_per_row_in_a_batch():
    model = seeded_model(tiny_config())
    row0 = torch.randint(2, 500, (1, 3))
    row0b = torch.randint(2, 500, (1, 5))
    row1 = torch.randint(2, 500, (1, 8))
    ids = torch.stack([torch.cat([row0, row0b], dim=1)[0], row1[0]])      # [2, 8]
    segments = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1], [0] * 6 + [1] * 2])
    with torch.no_grad():
        out = model(ids, segment_ids=segments, num_loops=3).logits
        ref_row0 = model(row0, num_loops=3).logits
        ref_row0b = model(row0b, num_loops=3).logits
        ref_row1a = model(row1[:, :6], num_loops=3).logits
        ref_row1b = model(row1[:, 6:], num_loops=3).logits
    assert torch.allclose(out[0:1, :3], ref_row0, atol=1e-4)
    assert torch.allclose(out[0:1, 3:], ref_row0b, atol=1e-4)
    assert torch.allclose(out[1:2, :6], ref_row1a, atol=1e-4)
    assert torch.allclose(out[1:2, 6:], ref_row1b, atol=1e-4)


def test_right_padding_does_not_change_real_positions():
    # Causal attention already hides right padding, so this checks the mask is accepted and harmless.
    model = seeded_model(tiny_config())
    real = torch.randint(2, 500, (1, 6))
    padded = torch.cat([real, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    mask = torch.tensor([[1] * 6 + [0] * 3], dtype=torch.bool)
    with torch.no_grad():
        alone = model(real, num_loops=3).logits
        masked = model(padded, attention_mask=mask, num_loops=3).logits
    assert torch.allclose(masked[:, :6], alone, atol=1e-4)


def test_attention_mask_blocks_padded_keys_inside_the_attention_layer():
    # Pad keys sit in the middle of the row. Changing only their inputs must not change the outputs at
    # real positions once they are masked. (The state highway is not masked; see InkyModel.forward.)
    model = seeded_model(tiny_config())
    attn = model.pre_layers[0].attn
    S = 7
    x = torch.randn(1, S, D)
    x_pad_changed = x.clone()
    x_pad_changed[:, 3] += 5.0
    mask = torch.ones(1, S, dtype=torch.bool)
    mask[:, 3] = False
    keep = mask[:, None, None, :] | torch.eye(S, dtype=torch.bool)[None, None]
    cos, sin = model.rope(0, S)
    with torch.no_grad():
        a, _ = attn(x, cos, sin, None, attn_mask=keep)
        b, _ = attn(x_pad_changed, cos, sin, None, attn_mask=keep)
        a_unmasked, _ = attn(x, cos, sin, None)
        b_unmasked, _ = attn(x_pad_changed, cos, sin, None)
    real = [0, 1, 2, 4, 5, 6]
    assert torch.allclose(a[:, real], b[:, real], atol=1e-5)
    assert not torch.allclose(a_unmasked[:, real], b_unmasked[:, real], atol=1e-5)


def test_left_padding_gives_finite_outputs():
    model = seeded_model(tiny_config())
    ids = torch.randint(2, 500, (1, 5))
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, attention_mask=mask, num_loops=2).logits
    assert torch.isfinite(out).all()


def test_packed_training_loss_is_finite_and_backpropagates():
    model = seeded_model(tiny_config(), train=True)
    ids = torch.randint(2, 500, (1, 9))
    segments = torch.tensor([[0] * 5 + [1] * 4])
    labels = ids.clone()
    labels[0, 5] = -100                     # the first token of the second document is not a target
    out = model(ids, labels=labels, segment_ids=segments, num_loops=3)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.embed_tokens.weight.grad is not None


def test_packing_arguments_are_validated():
    model = seeded_model(tiny_config())
    ids = torch.randint(2, 500, (2, 6))
    with pytest.raises(ValueError, match="segment_ids"):
        model(ids, segment_ids=torch.zeros(1, 6, dtype=torch.long))
    with pytest.raises(ValueError, match="attention_mask"):
        model(ids, attention_mask=torch.ones(2, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match="uncached"):
        model(ids, past=InkyCache(model.config), segment_ids=torch.zeros(2, 6, dtype=torch.long))


def test_ssm_mixer_layers_raise_the_clear_error_with_masks():
    model = seeded_model(tiny_config(ssm_mixer_layers=(0,)))
    ids = torch.randint(2, 500, (1, 6))
    with pytest.raises(NotImplementedError, match="does not support attn_mask or segment_ids"):
        model(ids, attention_mask=torch.ones(1, 6, dtype=torch.bool))
    with pytest.raises(NotImplementedError, match="ssm_mixer_layers"):
        model(ids, segment_ids=torch.zeros(1, 6, dtype=torch.long))


# ---------------------------------------------------------------- 5. step-CE token subset

def _reference_step_ce(model, states, weights, labels, rows=None, scale=1.0):
    """Stateless reference: sum over loops of weighted CE, on the given flat positions."""
    y_all = labels[:, 1:].reshape(-1)
    n_valid = (y_all != -100).sum().clamp(min=1).float()
    total = 0.0
    for n, st in enumerate(states):
        flat = st[:, :-1].reshape(-1, st.shape[-1])
        w = weights[n][:, :-1].reshape(-1)
        y = y_all
        if rows is not None:
            flat, w, y = flat[rows], w[rows], y_all[rows]
        logits = model._project(model.norm(flat))
        ce = F.cross_entropy(logits, y, ignore_index=-100, reduction="none")
        total = total + (ce * w).sum()
    return total * scale / n_valid


def _step_inputs(seed: int = 1, L: int = 3, B: int = 2, S: int = 9):
    torch.manual_seed(seed)
    states = [torch.randn(B, S, D) for _ in range(L)]
    weights = torch.softmax(torch.randn(L, B, S), dim=0)
    labels = torch.randint(2, 500, (B, S))
    labels[0, 3] = -100
    return states, weights, labels


def test_step_ce_fraction_one_is_the_full_position_loss():
    model = seeded_model(tiny_config(step_ce_token_fraction=1.0))
    states, weights, labels = _step_inputs()
    got = model._weighted_step_ce(states, weights, labels)
    ref = _reference_step_ce(model, states, weights, labels)
    assert torch.allclose(got, ref, atol=1e-5)


def test_step_ce_subset_is_shared_by_all_loops_and_rescaled(monkeypatch):
    model = seeded_model(tiny_config(step_ce_token_fraction=0.25))
    states, weights, labels = _step_inputs()
    L, B, S = 3, 2, 9
    N = B * (S - 1)
    k = max(1, round(0.25 * N))
    sel = torch.randperm(N, generator=torch.Generator().manual_seed(4))[:k].sort().values

    rows = []
    original = model._chunk_weighted_ce

    def spy(h, y, w):
        rows.append(h.shape[0])
        return original(h, y, w)

    monkeypatch.setattr(model, "_chunk_weighted_ce", spy)
    got = model._weighted_step_ce(states, weights, labels, generator=torch.Generator().manual_seed(4))
    assert sum(rows) == L * k, rows                      # every loop projects only the subset
    ref = _reference_step_ce(model, states, weights, labels, rows=sel, scale=N / k)
    assert torch.allclose(got, ref, atol=1e-5)


def test_step_ce_fraction_controls_the_projected_positions(monkeypatch):
    states, weights, labels = _step_inputs()
    counts = {}
    for fraction in (1.0, 0.5):
        model = seeded_model(tiny_config(step_ce_token_fraction=fraction))
        seen = []
        original = model._chunk_weighted_ce

        def spy(h, y, w, _seen=seen, _orig=original):
            _seen.append(h.shape[0])
            return _orig(h, y, w)

        monkeypatch.setattr(model, "_chunk_weighted_ce", spy)
        model._weighted_step_ce(states, weights, labels, generator=torch.Generator().manual_seed(0))
        counts[fraction] = sum(seen)
    N = 2 * 8
    assert counts[1.0] == 3 * N
    assert counts[0.5] == 3 * round(0.5 * N)


def test_step_ce_fraction_must_be_in_unit_interval():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="step_ce_token_fraction"):
            InkyConfig(step_ce_token_fraction=bad)
    assert InkyConfig().step_ce_token_fraction == 0.25


# ---------------------------------------------------------------- 6. MoE aux averaged over loops

def test_moe_aux_does_not_grow_with_loop_count():
    cfg = tiny_config()
    model = seeded_model(cfg)
    # With the router weights at zero every router logit is 0, so each MoE layer's z-loss is a
    # constant, and the aux depends only on how many MoE calls are averaged or summed.
    for module in model.modules():
        if isinstance(module, SharedExpertMoE):
            torch.nn.init.zeros_(module.router.weight)
    per_layer = cfg.router_z_loss_coef * math.log(cfg.num_routed_experts) ** 2
    # 1 pre layer + 1 post layer, plus the recurrent layers averaged over the loops run.
    expected = (1 + 1 + cfg.num_recurrent_layers) * per_layer
    ids = torch.randint(2, 500, (1, 8))
    with torch.no_grad():
        for loops in (2, 4):
            aux = float(model(ids, num_loops=loops).moe_aux_loss)
            assert abs(aux - expected) < 1e-6, (loops, aux, expected)


# ---------------------------------------------------------------- 7. loss parts

def test_loss_parts_recombine_to_the_total_loss():
    cfg = tiny_config()
    model = seeded_model(cfg)
    ids = torch.randint(2, 500, (2, 10))
    with torch.no_grad():
        out = model(ids, labels=ids, num_loops=3)
    parts = out.loss_parts
    assert set(parts) == {"ce", "step_ce", "z_loss", "moe_aux", "ponder_kl"}
    assert all(not v.requires_grad for v in parts.values())
    total = (
        parts["ce"]
        + cfg.z_loss_coef * parts["z_loss"]
        + parts["moe_aux"]
        + cfg.ponder_step_loss_coef * parts["step_ce"]
        + cfg.ponder_kl_coef * parts["ponder_kl"]
    )
    assert abs(float(total) - float(out.loss)) < 1e-4


def test_loss_parts_absent_without_labels():
    model = seeded_model(tiny_config())
    ids = torch.randint(2, 500, (1, 6))
    assert model(ids, num_loops=2).loss_parts is None


# ---------------------------------------------------------------- 8. generation

def _reference_generate(model, prompt, max_new_tokens, num_loops, no_repeat_ngram_size, bad_word_ids):
    """The previous generate loop, kept here as the reference: stateless n-gram and bad-word checks
    over the whole sequence, greedy decoding, default repetition penalty."""
    cfg = model.config
    cache = InkyCache(cfg)
    seq = prompt
    out = model(seq, past=cache, num_loops=num_loops, last_only=True)
    for _ in range(max_new_tokens):
        logits = modeling._apply_repetition_penalty(out.logits[:, -1, :], seq, 1.1)
        ids = seq[0].tolist()
        banned = banned_by_no_repeat_ngram(ids, no_repeat_ngram_size)
        if bad_word_ids:
            banned |= banned_by_bad_words(ids, bad_word_ids)
        logits = logits.clone()
        if banned:
            logits[0, sorted(banned)] = float("-inf")
        nxt = logits.argmax(dim=-1)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
        out = model(nxt[:, None], past=cache, last_only=True)
    return seq


def test_generate_matches_the_stateless_reference_loop():
    model = seeded_model(tiny_config(initializer_range=0.5))
    prompt = torch.tensor([[2, 3, 4]])
    plain = _reference_generate(model, prompt, 14, 2, 2, None)
    # Ban the most recent bigram of the reference output and one single token, so both paths are used.
    bigram = plain[0, -2:].tolist()
    single = [int(plain[0, 6])]
    bad = [bigram, single]
    ref = _reference_generate(model, prompt, 14, 2, 2, bad)
    out = model.generate(prompt, max_new_tokens=14, do_sample=False, repetition_penalty=1.1,
                         no_repeat_ngram_size=2, bad_word_ids=bad, num_loops=2)
    assert torch.equal(out, ref)
    out_plain = model.generate(prompt, max_new_tokens=14, do_sample=False, repetition_penalty=1.1,
                               no_repeat_ngram_size=2, num_loops=2)
    assert torch.equal(out_plain, _reference_generate(model, prompt, 14, 2, 2, None))


def test_generate_default_arguments_are_unchanged_on_a_batch():
    model = seeded_model(tiny_config(initializer_range=0.5))
    prompt = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    out = model.generate(prompt, max_new_tokens=10, do_sample=False, repetition_penalty=1.0,
                         no_repeat_ngram_size=3, num_loops=2)
    assert out.shape == (2, 14)
    assert out[:, :4].tolist() == prompt.tolist()


def test_stock_word_penalty_lowers_the_penalised_token():
    model = seeded_model(tiny_config(initializer_range=0.5))
    prompt = torch.tensor([[2, 3]])
    free = model.generate(prompt, max_new_tokens=4, do_sample=False, repetition_penalty=1.0,
                          no_repeat_ngram_size=0)
    first = int(free[0, 2])
    penalised = model.generate(prompt, max_new_tokens=4, do_sample=False, repetition_penalty=1.0,
                               no_repeat_ngram_size=0, stock_word_penalty=1e6, soft_penalty_ids=[first])
    assert int(penalised[0, 2]) != first
    # A zero penalty leaves the output unchanged, even when ids are supplied.
    zero = model.generate(prompt, max_new_tokens=4, do_sample=False, repetition_penalty=1.0,
                          no_repeat_ngram_size=0, stock_word_penalty=0.0, soft_penalty_ids=[first])
    assert torch.equal(zero, free)


def test_stock_word_penalty_argument_errors():
    model = seeded_model(tiny_config())
    prompt = torch.tensor([[2, 3]])
    with pytest.raises(ValueError, match="stock_word_penalty must be"):
        model.generate(prompt, max_new_tokens=1, stock_word_penalty=-1.0, soft_penalty_ids=[5])
    with pytest.raises(ValueError, match="needs soft_penalty_ids"):
        model.generate(prompt, max_new_tokens=1, stock_word_penalty=1.0)
    with pytest.raises(ValueError, match="inside the vocabulary"):
        model.generate(prompt, max_new_tokens=1, stock_word_penalty=1.0, soft_penalty_ids=[10_000])


def test_ngram_index_is_incremental():
    index = NgramBanIndex(3, [1, 2, 3])
    assert index.banned() == set()
    index.update(1)
    index.update(2)
    assert index.banned() == {3}
    assert index.banned() == banned_by_no_repeat_ngram([1, 2, 3, 1, 2], 3)


def test_soft_penalty_token_ids_keeps_single_token_forms_only():
    ids = soft_penalty_token_ids(lambda s: [ord(c) for c in s], words=["x", "ab"])
    assert ord("x") in ids and ord("X") in ids
    assert all(i not in (ord("a"), ord("b")) for i in ids)


# ---------------------------------------------------------------- 9. Arlo default

def test_arlo_is_off_by_default():
    assert InkyConfig().enable_arlo is False
    model = seeded_model(tiny_config())
    assert model.arlo is None
    with pytest.raises(RuntimeError, match="enable_arlo=False"):
        model.decide_binary(torch.randint(2, 500, (1, 4)))
    assert seeded_model(tiny_config(enable_arlo=True)).arlo is not None


# ---------------------------------------------------------------- 10. exports and evaluation

def test_new_public_helpers_are_exported():
    names = [
        "assign_parameter_groups", "format_group_counts", "pack_sequences_with_segments",
        "doc_boundary_mask", "baseline_reports", "compare_to_reference", "perplexity_by_loops",
        "early_exit_drift", "NgramBanIndex", "soft_penalty_token_ids",
    ]
    for name in names:
        assert hasattr(inky, name), name
        assert name in inky.__all__, name


def test_evaluate_with_padding_still_runs():
    from inky.evaluation import evaluate

    model = seeded_model(tiny_config())
    batch = torch.tensor([[2, 3, 4, 5, 1, 1], [6, 7, 8, 9, 10, 11]])
    result = evaluate(model, [batch], num_loops=2, ignore_id=1)
    assert math.isfinite(result["nll"]) and result["tokens"] > 0


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        if test.__code__.co_argcount:
            print(f"skip {test.__name__} (needs pytest fixtures)")
            continue
        test()
        print(f"ok  {test.__name__}")
