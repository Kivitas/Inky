"""Tests for per-token early exit (inference only). Run with: python -m pytest tests -q"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512, scratch_open_id=OPEN, scratch_sep_id=SEP, scratch_close_id=CLOSE,
        hidden_size=64, num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16,
        q_lora_rank=32, kv_lora_rank=24, intermediate_size=96, num_routed_experts=4, num_active_routed=2,
        ssm_state_dim=16, max_recurrent_loops=4, min_recurrent_loops=2, loop_dropout_prob=0.0,
        max_position_embeddings=256, arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4,
        eos_token_id=1, pad_token_id=1,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def seeded_model(config: InkyConfig) -> InkyModel:
    torch.manual_seed(0)
    return InkyModel(config).eval()


def test_without_exit_the_output_matches_the_full_forward():
    cfg = tiny_config()
    model = seeded_model(cfg)
    ids = torch.randint(2, 500, (1, 12))
    with torch.no_grad():
        full = model(ids, return_hidden=True, num_loops=4).hidden
        early = model.forward_early_exit(ids, halt_threshold=None, num_loops=4, return_hidden=True).hidden
    assert torch.allclose(full, early, atol=1e-4), float((full - early).abs().max())


def test_without_exit_every_token_runs_every_loop():
    cfg = tiny_config()
    model = seeded_model(cfg)
    ids = torch.randint(2, 500, (1, 12))
    out = model.forward_early_exit(ids, halt_threshold=None, num_loops=4, return_hidden=True)
    assert out.loop_applications == out.full_loop_applications == 12 * 4


def test_exit_computes_fewer_token_loops():
    cfg = tiny_config()
    model = seeded_model(cfg)
    ids = torch.randint(2, 500, (1, 20))
    out = model.forward_early_exit(ids, halt_threshold=0.5, return_hidden=True)
    assert 0 < out.loop_applications < out.full_loop_applications
    assert torch.isfinite(out.hidden).all()


def test_lower_threshold_computes_no_more_than_higher():
    cfg = tiny_config()
    model = seeded_model(cfg)
    ids = torch.randint(2, 500, (1, 20))
    low = model.forward_early_exit(ids, halt_threshold=0.3, return_hidden=True).loop_applications
    high = model.forward_early_exit(ids, halt_threshold=0.9, return_hidden=True).loop_applications
    full = model.forward_early_exit(ids, halt_threshold=None, return_hidden=True).loop_applications
    assert low <= high <= full


def test_mean_exit_loop_is_within_bounds():
    cfg = tiny_config()
    model = seeded_model(cfg)
    ids = torch.randint(2, 500, (1, 16))
    out = model.forward_early_exit(ids, halt_threshold=0.6, return_hidden=True)
    assert 1.0 <= float(out.mean_loops) <= cfg.max_recurrent_loops


def test_logits_path_returns_vocabulary_scores():
    cfg = tiny_config()
    model = seeded_model(cfg)
    ids = torch.randint(2, 500, (1, 8))
    out = model.forward_early_exit(ids, halt_threshold=0.7, last_only=True)
    assert out.logits.shape == (1, 1, cfg.vocab_size)


def test_batch_larger_than_one_is_refused():
    model = seeded_model(tiny_config())
    with pytest.raises(ValueError):
        model.forward_early_exit(torch.randint(2, 500, (2, 6)), halt_threshold=0.5)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_threshold_is_refused(bad):
    model = seeded_model(tiny_config())
    with pytest.raises(ValueError):
        model.forward_early_exit(torch.randint(2, 500, (1, 6)), halt_threshold=bad)


def test_hybrid_ssm_blocks_are_refused():
    model = seeded_model(tiny_config(ssm_mixer_layers=(0,)))
    with pytest.raises(NotImplementedError):
        model.forward_early_exit(torch.randint(2, 500, (1, 6)), halt_threshold=0.5)


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        if hasattr(test, "pytestmark"):
            continue
        test()
        print(f"ok  {test.__name__}")
    print("done")
