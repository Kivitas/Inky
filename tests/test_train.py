"""Tests for the training loop. Run with: python -m pytest tests/test_train.py -q -p no:cacheprovider"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import HybridOptimizer, InkyConfig, InkyModel  # noqa: E402
from inky.train import loop_curriculum, train, wsd_lr_scale  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510
MIN_LOOPS, MAX_LOOPS = 2, 4


def tiny_model() -> InkyModel:
    torch.manual_seed(0)
    config = InkyConfig(
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
        max_recurrent_loops=MAX_LOOPS,
        min_recurrent_loops=MIN_LOOPS,
        loop_dropout_prob=0.0,
        max_position_embeddings=256,
        arlo_dim=16,
        arlo_pointer_rank=8,
        arlo_slots=4,
        eos_token_id=1,
        pad_token_id=1,
    )
    return InkyModel(config)


def random_batches(n: int = 2):
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, 512, (2, 16), generator=gen) for _ in range(n)]


def test_wsd_lr_scale_warmup_plateau_and_decay():
    total, warmup = 10, 2
    values = [wsd_lr_scale(s, total, warmup, decay_fraction=0.2) for s in range(total)]

    assert values[0] == 0.0
    assert values[warmup] == 1.0
    assert values[total - 1] == 0.0
    assert all(0.0 <= v <= 1.0 for v in values)
    # Flat plateau between warmup and the decay window, then non-increasing decay.
    assert values[warmup:total - 2] == [1.0] * (total - 2 - warmup)
    decay = values[total - 2:]
    assert all(a >= b for a, b in zip(decay, decay[1:]))


def test_wsd_lr_scale_stays_in_unit_interval_for_odd_inputs():
    for s in range(-2, 15):
        v = wsd_lr_scale(s, 7, 3, decay_fraction=0.5)
        assert 0.0 <= v <= 1.0


def test_loop_curriculum_min_at_start_max_at_end_and_monotone():
    total = 20
    values = [loop_curriculum(s, total, MIN_LOOPS, MAX_LOOPS, warmup_fraction=0.5) for s in range(total)]

    assert values[0] == MIN_LOOPS
    assert values[-1] == MAX_LOOPS
    assert all(isinstance(v, int) for v in values)
    assert all(MIN_LOOPS <= v <= MAX_LOOPS for v in values)
    assert all(a <= b for a, b in zip(values, values[1:]))
    # After the first half of training the count stays at the maximum.
    assert all(v == MAX_LOOPS for v in values[total // 2 + 1:])


def test_train_runs_three_steps_and_logs():
    model = tiny_model()
    optimizer = HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02)
    logs = []

    history = train(
        model,
        optimizer,
        random_batches(2),
        steps=3,
        log_every=1,
        log_fn=logs.append,
    )

    train_records = [r for r in history if r.get("kind") == "train"]
    assert len(train_records) == 3
    assert all(math.isfinite(r["loss"]) for r in train_records)
    assert [r["step"] for r in train_records] == [1, 2, 3]
    assert logs, "log_fn was never called"


def test_train_writes_checkpoints_at_interval_and_end(tmp_path):
    model = tiny_model()
    optimizer = HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02)

    train(
        model,
        optimizer,
        random_batches(2),
        steps=3,
        checkpoint_dir=tmp_path,
        checkpoint_every=2,
        log_fn=lambda _msg: None,
    )

    assert (tmp_path / "step_2").is_dir()
    assert (tmp_path / "step_3").is_dir()


def test_train_loop_curriculum_num_loops_within_bounds():
    model = tiny_model()
    optimizer = HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02)

    history = train(
        model,
        optimizer,
        random_batches(2),
        steps=3,
        use_loop_curriculum=True,
        log_fn=lambda _msg: None,
    )

    loops = [r["num_loops"] for r in history if r.get("kind") == "train"]
    assert len(loops) == 3
    assert all(v is not None and MIN_LOOPS <= v <= MAX_LOOPS for v in loops)


def test_train_rejects_zero_steps():
    model = tiny_model()
    optimizer = HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02)
    with pytest.raises(ValueError):
        train(model, optimizer, random_batches(2), steps=0, log_fn=lambda _msg: None)
