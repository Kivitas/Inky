"""Tests for reasoning modes (low=2, high=5, max=10), style modes (formal, informal), and architecture upgrades."""

import pytest
import torch

import scripts.ablate as ablate
import scripts.sample as sample_script
from inky import (
    FORMAL_PHRASES,
    FORMAL_WORDS,
    INFORMAL_PHRASES,
    INFORMAL_WORDS,
    REASONING_MODES,
    STYLE_MODES,
    InkyCache,
    InkyConfig,
    InkyModel,
    build_hybrid_optimizer,
    build_style_controls,
    get_style_profile,
    resolve_reasoning_loops,
)
from inky.generation import _candidate_distribution


def _tiny_cfg(**overrides) -> InkyConfig:
    base = dict(
        vocab_size=256,
        scratch_open_id=253,
        scratch_sep_id=254,
        scratch_close_id=255,
        hidden_size=32,
        num_attention_heads=2,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        intermediate_size=48,
        num_routed_experts=4,
        num_active_routed=2,
        ssm_state_dim=8,
        max_recurrent_loops=4,
        min_recurrent_loops=2,
        loop_dropout_prob=0.0,
        max_position_embeddings=128,
        eos_token_id=1,
        pad_token_id=1,
    )
    base.update(overrides)
    return InkyConfig(**base)


def test_reasoning_mode_constants_and_resolution():
    assert REASONING_MODES == {"low": 2, "high": 5, "max": 10}
    assert resolve_reasoning_loops("low") == 2
    assert resolve_reasoning_loops("high") == 5
    assert resolve_reasoning_loops("max") == 10
    assert resolve_reasoning_loops("max", num_loops=7) == 7
    assert resolve_reasoning_loops(None, num_loops=None) is None
    with pytest.raises(ValueError, match="reasoning_mode"):
        resolve_reasoning_loops("ultra")


def test_reasoning_modes_low_high_max_forward_and_10_loop_extrapolation():
    torch.manual_seed(0)
    # Model configured with max_recurrent_loops=4 still extrapolates cleanly to 5 ('high') and 10 ('max') loops!
    model = InkyModel(_tiny_cfg(max_recurrent_loops=4)).eval()
    ids = torch.tensor([[10, 11, 12, 13, 14]])

    out_low = model(ids, reasoning_mode="low")
    out_high = model(ids, reasoning_mode="high")
    out_max = model(ids, reasoning_mode="max")

    assert float(out_low.mean_loops.detach()) < float(out_high.mean_loops.detach()) < float(out_max.mean_loops.detach())

    # Cached token-by-token decoding at reasoning_mode="max" (10 loops) matches uncached forward
    cache = InkyCache(model.config)
    logits_Prefill = model(ids[:, :-1], past=cache, reasoning_mode="max").logits
    logits_last = model(ids[:, -1:], past=cache, reasoning_mode="max").logits
    assert cache.num_loops == 10
    assert len(cache.core) == 10
    cat_logits = torch.cat([logits_Prefill, logits_last], dim=1)
    torch.testing.assert_close(cat_logits, out_max.logits, atol=1e-4, rtol=1e-4)

    # Early exit without threshold matches forward in 'max' (10 loops) and 'high' (5 loops)
    ee_max = model.forward_early_exit(ids, halt_threshold=None, reasoning_mode="max")
    torch.testing.assert_close(ee_max.logits, out_max.logits, atol=1e-4, rtol=1e-4)


def test_human_reasoning_architecture_upgrades_and_gradients():
    torch.manual_seed(1)
    cfg = _tiny_cfg(
        max_recurrent_loops=10,
        min_recurrent_loops=2,
        ponder_prior_steps=5.0,
        use_thought_gate=True,
        use_mode_conditioning=True,
        convergence_halting=True,
        loop_input_injection=True,
        attention_gate=True,
        qk_norm=True,
    )
    model = InkyModel(cfg)
    assert model.core.thought_gate is not None
    assert model.mode_cond is not None
    assert "mode_conditioning" in model.count_parameters()

    # Optimiser assigns thought_gate, reasoning_embed, style_embed to AdamW
    opt = build_hybrid_optimizer(model)
    assert opt is not None

    ids = torch.tensor([[5, 6, 7, 8]])
    out = model(ids, labels=ids, reasoning_mode="high", style_mode="formal")
    out.loss.backward()

    assert model.core.thought_gate.weight.grad is not None
    assert model.mode_cond.reasoning_embed.grad is not None
    assert model.mode_cond.style_embed.grad is not None
    # Only index 1 ('high') of reasoning_embed and index 0 ('formal') of style_embed received gradient
    assert model.mode_cond.reasoning_embed.grad[1].abs().sum() > 0
    assert model.mode_cond.reasoning_embed.grad[0].abs().sum() == 0
    assert model.mode_cond.style_embed.grad[0].abs().sum() > 0
    assert model.mode_cond.style_embed.grad[1].abs().sum() == 0


def test_convergence_halting_responds_to_state_velocity():
    torch.manual_seed(2)
    cfg = _tiny_cfg(convergence_halting=True)
    model = InkyModel(cfg).eval()
    h = torch.randn(1, 4, cfg.hidden_size)
    # Converged state (prev_h == h, velocity 0) vs rapidly changing state (prev_h far from h)
    lam_converged = model.halting(h, prev_h=h, convergence_weight=1.0)
    lam_changing = model.halting(h, prev_h=h + 2.0 * torch.randn_like(h), convergence_weight=1.0)
    assert bool((lam_converged > lam_changing).all())


def test_formal_and_informal_style_modes_and_min_p():
    assert STYLE_MODES == ("formal", "informal")
    formal_prof = get_style_profile("formal")
    informal_prof = get_style_profile("informal")
    assert formal_prof["temperature"] < informal_prof["temperature"]
    assert formal_prof["repetition_penalty"] > informal_prof["repetition_penalty"]

    vocab = {w: [10 + i] for i, w in enumerate(INFORMAL_WORDS[:3] + FORMAL_WORDS[:3])}
    encode = lambda s: vocab.get(s.strip().lower(), [99, 100])  # noqa: E731

    f_ctrl = build_style_controls("formal", encode=encode)
    i_ctrl = build_style_controls("informal", encode=encode)
    assert vocab[INFORMAL_WORDS[0]][0] in f_ctrl["soft_penalty_ids"]
    assert vocab[FORMAL_WORDS[0]][0] in i_ctrl["soft_penalty_ids"]
    assert len(INFORMAL_PHRASES) > 0 and len(FORMAL_PHRASES) > 0

    # min_p filters low-probability tail tokens relative to top token
    logits = torch.tensor([[10.0, 9.5, 1.0, 0.0]])
    _, probs = _candidate_distribution(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.2)
    assert probs[0, 2].item() == 0.0
    assert probs[0, 3].item() == 0.0
    assert probs[0, 0].item() > 0.0 and probs[0, 1].item() > 0.0


def test_generate_and_sample_cli_with_reasoning_and_style_modes():
    torch.manual_seed(3)
    model = InkyModel(_tiny_cfg()).eval()
    prompt = torch.tensor([[10, 11, 12]])
    codec = sample_script.CharCodec()

    out_formal = model.generate(
        prompt, max_new_tokens=4, reasoning_mode="high", style_mode="formal", encode=codec.encode
    )
    out_informal = model.generate(
        prompt, max_new_tokens=4, reasoning_mode="max", style_mode="informal", encode=codec.encode
    )
    assert out_formal.shape == (1, 7)
    assert out_informal.shape == (1, 7)

    # CLI sample.py smoke test with --reasoning-mode max and --style-mode formal
    txt = sample_script.main([
        "--dry-run", "--max-new-tokens", "4", "--reasoning-mode", "max", "--style-mode", "formal"
    ])
    assert isinstance(txt, str)

    # Preset and ablation variant check
    preset = InkyConfig.human_reasoning()
    assert preset.max_recurrent_loops == 10
    assert "human_reasoning" in ablate.VARIANTS
