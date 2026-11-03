"""Tests for the training-loop fixes: loss logging, data order, document boundaries, non-finite gradients,
seeding, checkpoints and the scripts/train.py entry point.

Run with: python -B -m pytest tests/test_train_fixes.py -q -p no:cacheprovider
"""

import json
import math
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inky import HybridOptimizer, InkyConfig, InkyModel  # noqa: E402
from inky import train as train_module  # noqa: E402
from inky.optim import format_group_counts  # noqa: E402
from inky.train import (  # noqa: E402
    LOSS_PART_NAMES,
    load_rng_state,
    load_training_state,
    packed_stream,
    read_train_state,
    save_training_state,
    train,
)

OPEN, SEP, CLOSE = 508, 509, 510
MIN_LOOPS, MAX_LOOPS = 2, 4


def quiet(_msg):
    return None


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


def optimizer_for(model) -> HybridOptimizer:
    return HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02)


def random_batches(n: int = 2):
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, 512, (2, 16), generator=gen) for _ in range(n)]


def marked_batches(n: int):
    """Batch i starts with token 20*i, so its identity can be read back from the model inputs."""
    return [torch.arange(16).repeat(2, 1) + 20 * i for i in range(n)]


def train_records(history):
    return [r for r in history if r.get("kind") == "train"]


class _Spy(torch.nn.Module):
    """Wraps the model to record each call's inputs, to make the loss non-finite on chosen calls, and to
    snapshot the parameters before each call. ``nan_loss`` gives a NaN loss. ``nan_grad`` gives a finite
    loss whose backward pass produces NaN gradients (sqrt at zero)."""

    def __init__(self, inner, nan_loss=(), nan_grad=()):
        super().__init__()
        self.inner = inner
        self.config = inner.config
        self.nan_loss = set(nan_loss)
        self.nan_grad = set(nan_grad)
        self.calls = []
        self.params_at_call = []

    def forward(self, input_ids, labels=None, num_loops=None, **kwargs):
        self.calls.append({
            "ids": input_ids.clone(),
            "labels": None if labels is None else labels.clone(),
            "segment_ids": None if kwargs.get("segment_ids") is None else kwargs["segment_ids"].clone(),
        })
        self.params_at_call.append([p.detach().clone() for p in self.inner.parameters()])
        out = self.inner(input_ids=input_ids, labels=labels, num_loops=num_loops, **kwargs)
        n = len(self.calls)
        loss = out.loss
        if n in self.nan_loss:
            loss = loss * float("nan")
        if n in self.nan_grad:
            loss = loss + torch.sqrt((next(self.inner.parameters()) * 0.0).sum())
        return SimpleNamespace(loss=loss, loss_parts=out.loss_parts)

    def update_moe_balance(self):
        self.inner.update_moe_balance()

    def save_pretrained(self, path):
        self.inner.save_pretrained(path)


# ----------------------------------------------------------------------------- 1. loss logging


def test_loss_parts_are_logged_and_total_is_their_weighted_sum():
    model = tiny_model()
    logs = []
    history = train(model, optimizer_for(model), random_batches(2), steps=2, log_every=1, log_fn=logs.append)

    rec = train_records(history)[0]
    for name in LOSS_PART_NAMES:
        assert isinstance(rec[name], float), name
    cfg = model.config
    expected = (
        rec["ce"]
        + cfg.z_loss_coef * rec["z_loss"]
        + rec["moe_aux"]
        + cfg.ponder_step_loss_coef * rec["step_ce"]
        + cfg.ponder_kl_coef * rec["ponder_kl"]
    )
    assert rec["loss"] == pytest.approx(expected, rel=1e-4)
    # The logged line names each component, and says which one is comparable to eval.
    line = next(m for m in logs if m.startswith("step 1/2 loss"))
    for name in ("ce", "step_ce", "z_loss", "moe_aux", "ponder_kl", "eval-comparable"):
        assert name in line
    assert any("loss = ce +" in m and "eval-comparable" in m for m in logs)


# ----------------------------------------------------------------------------- 2. data order


def test_batch_list_is_reshuffled_each_epoch_and_reproducible():
    batches = marked_batches(4)

    def run(seed):
        spy = _Spy(tiny_model())
        train(spy, optimizer_for(spy), batches, steps=8, log_every=0, log_fn=quiet, seed=seed)
        return [int(c["ids"][0, 0]) // 20 for c in spy.calls]

    order = run(5)
    epoch1, epoch2 = order[:4], order[4:]
    assert sorted(epoch1) == [0, 1, 2, 3]
    assert sorted(epoch2) == [0, 1, 2, 3]
    assert epoch1 != epoch2
    assert run(5) == order


def test_window_tensor_is_reshuffled_and_batched():
    windows = torch.arange(16).repeat(10, 1) + torch.arange(10).unsqueeze(1) * 16
    spy = _Spy(tiny_model())
    train(spy, optimizer_for(spy), windows, steps=4, batch_size=4, log_every=0, log_fn=quiet, seed=2)
    sizes = [c["ids"].shape[0] for c in spy.calls]
    # 10 rows in batches of 4 gives 4, 4, 2 per epoch, so the first 4 steps are 4, 4, 2, 4.
    assert sizes == [4, 4, 2, 4]


def test_packed_stream_moves_window_offsets_between_epochs():
    docs = [list(range(10, 10 + 300))]
    source = packed_stream(lambda: iter(docs), seq_len=16, eos_id=1)
    first_a = source(random.Random(11))
    first_b = source(random.Random(12))
    assert first_a.shape[1] == 16 and first_b.shape[1] == 16
    # Different RNGs give different window offsets, so the windows start at different places.
    assert int(first_a[0, 0]) != int(first_b[0, 0])

    spy = _Spy(tiny_model())
    history = train(spy, optimizer_for(spy), source, steps=6, batch_size=8, log_every=0, log_fn=quiet, seed=3)
    assert len(train_records(history)) == 6
    assert all(c["ids"].shape[1] == 16 for c in spy.calls)


def test_window_source_needs_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        train(tiny_model(), optimizer_for(tiny_model()), torch.zeros(4, 16, dtype=torch.long), steps=1, log_fn=quiet)


# ----------------------------------------------------------------------------- 3. document boundaries


def test_document_boundaries_are_masked_and_segments_passed_to_the_model():
    ids = torch.tensor([[5, 6, 7, 1, 8, 9, 10, 1]])
    docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    spy = _Spy(tiny_model())
    train(spy, optimizer_for(spy), [(ids, docs)], steps=1, log_every=0, log_fn=quiet)

    call = spy.calls[0]
    assert torch.equal(call["segment_ids"], docs)
    expected = ids.clone()
    expected[0, 4] = -100  # the target 8 starts document 1, so it is not scored from document 0
    assert torch.equal(call["labels"], expected)


def test_without_segments_labels_are_the_inputs_and_no_segment_ids_are_passed():
    ids = torch.tensor([[5, 6, 7, 1, 8, 9, 10, 1]])
    spy = _Spy(tiny_model())
    train(spy, optimizer_for(spy), [ids], steps=1, log_every=0, log_fn=quiet)
    assert torch.equal(spy.calls[0]["labels"], ids)
    assert spy.calls[0]["segment_ids"] is None


def test_packed_stream_with_segments_returns_doc_ids_and_masks_boundaries():
    docs = [list(range(10, 20)), list(range(30, 45))]
    source = packed_stream(lambda: iter(docs), seq_len=8, eos_id=1, with_segments=True)
    tokens, doc_ids = source(random.Random(0))
    assert tokens.shape == doc_ids.shape
    spy = _Spy(tiny_model())
    train(spy, optimizer_for(spy), source, steps=1, batch_size=tokens.shape[0], log_every=0, log_fn=quiet, seed=0)
    labels = spy.calls[0]["labels"]
    assert torch.equal(spy.calls[0]["segment_ids"], doc_ids)
    assert (labels[:, 1:] == -100).sum() >= 1  # at least one document boundary was masked


# ----------------------------------------------------------------------------- 4. non-finite gradients


def test_nonfinite_grad_skips_the_step_and_leaves_parameters_unchanged():
    spy = _Spy(tiny_model(), nan_grad=[2])
    # warmup_steps=0 so step 1 runs at full learning rate and visibly changes the weights.
    history = train(spy, optimizer_for(spy), random_batches(2), steps=3, warmup_steps=0, log_every=0, log_fn=quiet)

    records = train_records(history)
    assert [r["skipped"] for r in records] == [False, True, False]
    assert records[1]["skip_reason"] == "nonfinite_grad"
    assert not math.isfinite(records[1]["grad_norm"])
    # Call 3 sees the parameters after step 1 only: the skipped step 2 changed nothing.
    assert any(not torch.equal(a, b) for a, b in zip(spy.params_at_call[0], spy.params_at_call[1]))
    for a, b in zip(spy.params_at_call[1], spy.params_at_call[2]):
        assert torch.equal(a, b)
    assert all(torch.isfinite(p).all() for p in spy.inner.parameters())


def test_consecutive_skips_abort_but_scattered_skips_do_not():
    with pytest.raises(RuntimeError, match="consecutive"):
        spy = _Spy(tiny_model(), nan_loss=[1, 2, 3])
        train(spy, optimizer_for(spy), random_batches(2), steps=5, max_skipped_steps=2, log_fn=quiet)

    spy = _Spy(tiny_model(), nan_loss=[1, 3, 5])
    history = train(spy, optimizer_for(spy), random_batches(2), steps=5, max_skipped_steps=1, log_fn=quiet)
    assert sum(r["skipped"] for r in train_records(history)) == 3


# ----------------------------------------------------------------------------- 5. seeding and resume


def test_same_seed_gives_the_same_losses():
    def run(seed):
        model = tiny_model()
        history = train(
            model, optimizer_for(model), random_batches(2), steps=3, seed=seed,
            use_loop_curriculum=True, log_fn=quiet,
        )
        return [r["loss"] for r in train_records(history)]

    first = run(7)
    assert run(7) == pytest.approx(first, rel=1e-6)


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    batches = random_batches(3)
    model_a = tiny_model()
    history_a = train(
        model_a, optimizer_for(model_a), batches, steps=4, seed=11, use_loop_curriculum=True,
        checkpoint_dir=tmp_path / "run", checkpoint_every=2, log_fn=quiet,
    )

    model_b = tiny_model()
    history_b = train(
        model_b, optimizer_for(model_b), batches, steps=4, seed=11, use_loop_curriculum=True,
        resume_from=tmp_path / "run" / "step_2", log_fn=quiet,
    )

    records_a, records_b = train_records(history_a), train_records(history_b)
    assert [r["step"] for r in records_b] == [1, 2, 3, 4]
    assert [r["num_loops"] for r in records_b[2:]] == [r["num_loops"] for r in records_a[2:]]
    assert [r["loss"] for r in records_b[2:]] == pytest.approx([r["loss"] for r in records_a[2:]], rel=1e-4)


# ----------------------------------------------------------------------------- 6. checkpoints


def test_keep_last_prunes_old_step_directories(tmp_path):
    model = tiny_model()
    train(model, optimizer_for(model), random_batches(2), steps=5, checkpoint_dir=tmp_path,
          checkpoint_every=1, keep_last=2, log_fn=quiet)
    assert sorted(p.name for p in tmp_path.glob("step_*")) == ["step_4", "step_5"]


def test_final_checkpoint_is_always_written(tmp_path):
    model = tiny_model()
    train(model, optimizer_for(model), random_batches(2), steps=5, checkpoint_dir=tmp_path / "a",
          checkpoint_every=3, keep_last=0, log_fn=quiet)
    assert sorted(p.name for p in (tmp_path / "a").glob("step_*")) == ["step_3", "step_5"]

    train(model, optimizer_for(model), random_batches(2), steps=5, checkpoint_dir=tmp_path / "b",
          checkpoint_every=0, log_fn=quiet)
    assert sorted(p.name for p in (tmp_path / "b").glob("step_*")) == ["step_5"]


def test_checkpoint_has_rng_state_config_and_no_temp_leftovers(tmp_path):
    model = tiny_model()
    train(model, optimizer_for(model), random_batches(2), steps=2, seed=4, checkpoint_dir=tmp_path,
          checkpoint_every=1, log_fn=quiet)
    step_dir = tmp_path / "step_2"
    assert (step_dir / "rng.pt").exists()
    assert load_rng_state(step_dir) is not None
    cfg = json.loads((step_dir / "train_config.json").read_text())
    assert cfg["seed"] == 4 and cfg["steps"] == 2 and "model_config" in cfg
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())


def test_failed_save_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    model = tiny_model()
    optimizer = optimizer_for(model)
    save_training_state(tmp_path / "state", model, optimizer, 1, [])

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(train_module.torch, "save", boom)
    with pytest.raises(OSError):
        save_training_state(tmp_path / "state", model, optimizer, 2, [])
    monkeypatch.undo()

    assert read_train_state(tmp_path / "state")["step"] == 1
    fresh = tiny_model()
    assert load_training_state(tmp_path / "state", fresh, optimizer_for(fresh)) == 1
    assert not (tmp_path / ".state.tmp").exists()


def test_best_checkpoint_follows_the_lowest_eval_nll(tmp_path):
    model = tiny_model()
    history = train(
        model, optimizer_for(model), random_batches(2), steps=4, eval_batches=random_batches(1),
        eval_every=1, checkpoint_dir=tmp_path, log_fn=quiet,
    )
    evals = [r for r in history if r["kind"] == "eval"]
    best = min(evals, key=lambda r: r["eval_nll"])
    marked = [r["step"] for r in evals if r["is_best"]]
    assert marked and marked[-1] == best["step"]

    saved = torch.load(tmp_path / "best.pt", weights_only=False)
    assert saved["step"] == best["step"]
    assert saved["eval_nll"] == pytest.approx(best["eval_nll"])
    assert set(saved["model"]) == set(model.state_dict())


def test_group_counts_are_logged_once_at_start():
    model = tiny_model()
    optimizer = optimizer_for(model)
    logs = []
    train(model, optimizer, random_batches(2), steps=2, log_every=0, log_fn=logs.append)
    expected = format_group_counts(optimizer.group_counts())
    assert sum(1 for m in logs if m == "optimizer groups: " + expected) == 1


# ----------------------------------------------------------------------------- 7. scripts/train.py


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "train.py"), *args],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT),
    )


def test_cli_dry_run_smoke(tmp_path):
    proc = _run_cli("--dry-run", "--steps", "3", "--out-dir", str(tmp_path / "out"))
    assert proc.returncode == 0, proc.stderr
    assert "step 3/3" in proc.stdout
    assert (tmp_path / "out" / "checkpoints" / "step_3").is_dir()
    assert (tmp_path / "out" / "checkpoints" / "best.pt").exists()


def test_cli_real_run_without_tokenizer_fails_with_a_clear_message(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("some text " * 40, encoding="utf-8")
    proc = _run_cli("--corpus", str(corpus), "--out-dir", str(tmp_path / "out"), "--steps", "1")
    assert proc.returncode != 0
    assert "tokenizer" in proc.stderr.lower()
