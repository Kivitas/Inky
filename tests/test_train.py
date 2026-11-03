"""Tests for the training loop. Run with: python -m pytest tests/test_train.py -q -p no:cacheprovider"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import HybridOptimizer, InkyConfig, InkyModel  # noqa: E402
from inky.train import (  # noqa: E402
    load_training_state,
    loop_curriculum,
    save_training_state,
    train,
    wsd_lr_scale,
)

OPEN, SEP, CLOSE = 508, 509, 510
MIN_LOOPS, MAX_LOOPS = 2, 4


class _NaNOnCall(torch.nn.Module):
    """Wraps a model and makes its loss NaN on one forward call (1-based ``bad_call``)."""

    def __init__(self, inner, bad_call: int):
        super().__init__()
        self.inner = inner
        self.config = inner.config
        self.bad_call = bad_call
        self.calls = 0

    def forward(self, input_ids, labels=None, num_loops=None, **kwargs):
        out = self.inner(input_ids=input_ids, labels=labels, num_loops=num_loops, **kwargs)
        self.calls += 1
        loss = out.loss * float("nan") if self.calls == self.bad_call else out.loss
        return SimpleNamespace(loss=loss, loss_parts=out.loss_parts)

    def update_moe_balance(self):
        self.inner.update_moe_balance()

    def save_pretrained(self, path):
        self.inner.save_pretrained(path)


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
    # The final step still gets a nonzero rate; the decay hits zero just after it.
    assert values[total - 1] == 0.5
    assert wsd_lr_scale(total, total, warmup, decay_fraction=0.2) == 0.0
    assert all(0.0 <= v <= 1.0 for v in values)
    # Flat plateau between warmup and the decay window, then non-increasing decay.
    assert values[warmup:total - 2] == [1.0] * (total - 2 - warmup)
    decay = values[total - 2:]
    assert all(a >= b for a, b in zip(decay, decay[1:]))


def test_default_warmup_is_nonzero_for_a_fresh_run():
    model = tiny_model()
    optimizer = HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02)
    history = train(model, optimizer, random_batches(2), steps=40, log_fn=lambda _msg: None)
    records = [r for r in history if r.get("kind") == "train"]
    # 5% of 40 steps is 2 warmup steps, so the run ramps up instead of starting at full rate.
    assert records[0]["lr_scale"] == 0.0
    assert records[1]["lr_scale"] == 0.5


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


def test_non_finite_loss_is_skipped_recorded_and_not_checkpointed(tmp_path):
    model = _NaNOnCall(tiny_model(), bad_call=2)
    optimizer = HybridOptimizer(model.inner, adamw_lr=3e-3, muon_lr=0.02)

    history = train(
        model,
        optimizer,
        random_batches(2),
        steps=3,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
        log_fn=lambda _msg: None,
    )

    records = [r for r in history if r.get("kind") == "train"]
    assert [r["skipped"] for r in records] == [False, True, False]
    assert not math.isfinite(records[1]["loss"])
    assert (tmp_path / "step_1").is_dir() and (tmp_path / "step_3").is_dir()
    assert not (tmp_path / "step_2").exists()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_too_many_skipped_steps_raises():
    model = _NaNOnCall(tiny_model(), bad_call=1)
    optimizer = HybridOptimizer(model.inner, adamw_lr=3e-3, muon_lr=0.02)
    with pytest.raises(RuntimeError):
        train(
            model, optimizer, random_batches(2), steps=2,
            max_skipped_steps=0, log_fn=lambda _msg: None,
        )


def test_resume_continues_step_numbers_and_schedule(tmp_path):
    model = tiny_model()
    optimizer = HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02)
    first = train(model, optimizer, random_batches(2), steps=2, log_fn=lambda _msg: None)
    save_training_state(tmp_path / "state", model, optimizer, 2, first)

    fresh = tiny_model()
    with torch.no_grad():
        for p in fresh.parameters():
            p.add_(1.0)  # make sure the reload really overwrites the weights
    fresh_opt = HybridOptimizer(fresh, adamw_lr=3e-3, muon_lr=0.02)
    assert load_training_state(tmp_path / "state", fresh, fresh_opt) == 2
    assert torch.equal(fresh.embed_tokens.weight, model.embed_tokens.weight)

    resumed = train(
        fresh,
        fresh_opt,
        random_batches(2),
        steps=3,
        resume_from=tmp_path / "state",
        log_fn=lambda _msg: None,
    )
    records = [r for r in resumed if r.get("kind") == "train"]
    # The returned history is the saved history (steps 1-2) followed by the new step 3.
    assert [r["step"] for r in records] == [1, 2, 3]
    # The schedule is computed from the total step count passed to train(), not from the resumed run.
    assert records[2]["lr_scale"] == wsd_lr_scale(2, 3, 0, 0.2)
