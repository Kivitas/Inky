"""Tests for the open training items: gradient accumulation (TRAIN-12), mixed precision (TRAIN-12) and the
per-depth eval option (TRAIN-08).

Run with: python -B -m pytest tests/test_train_open_fixes.py -q -p no:cacheprovider
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inky import HybridOptimizer, InkyConfig, InkyModel  # noqa: E402
from inky.train import train  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


def quiet(_msg):
    return None


def make_model(min_loops: int = 2, max_loops: int = 4, zero_aux: bool = True) -> InkyModel:
    """A tiny model. ``zero_aux`` removes the auxiliary loss terms, so the loss is a plain per-token mean."""
    torch.manual_seed(0)
    extra = dict(z_loss_coef=0.0, router_z_loss_coef=0.0, ponder_step_loss_coef=0.0, ponder_kl_coef=0.0) if zero_aux else {}
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
        max_recurrent_loops=max_loops,
        min_recurrent_loops=min_loops,
        loop_dropout_prob=0.0,
        max_position_embeddings=256,
        arlo_dim=16,
        arlo_pointer_rank=8,
        arlo_slots=4,
        eos_token_id=1,
        pad_token_id=1,
        **extra,
    )
    return InkyModel(config)


def optimizer_for(model) -> HybridOptimizer:
    return HybridOptimizer(model, adamw_lr=3e-3, muon_lr=0.02)


def random_batches(n: int, rows: int = 2, seq: int = 16, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 512, (rows, seq), generator=gen) for _ in range(n)]


def train_records(history):
    return [r for r in history if r.get("kind") == "train"]


# ----------------------------------------------------------------------------- gradient accumulation


def test_grad_accum_equals_one_big_batch():
    base = make_model(min_loops=2, max_loops=2)  # fixed depth, so both runs see the same loop count
    big = torch.randint(0, 512, (4, 16), generator=torch.Generator().manual_seed(3))
    before = [p.detach().clone() for p in base.parameters()]

    model_big = copy.deepcopy(base)
    hist_big = train(
        model_big, optimizer_for(model_big), [big], steps=1, warmup_steps=0,
        max_grad_norm=None, log_fn=quiet,
    )

    model_acc = copy.deepcopy(base)
    hist_acc = train(
        model_acc, optimizer_for(model_acc), [big[:2], big[2:]], steps=1, warmup_steps=0,
        max_grad_norm=None, grad_accum_steps=2, log_fn=quiet,
    )

    loss_big = train_records(hist_big)[0]["loss"]
    loss_acc = train_records(hist_acc)[0]["loss"]
    assert abs(loss_big - loss_acc) < 1e-5

    moved = max(
        (p.detach() - b).abs().max().item() for p, b in zip(model_big.parameters(), before)
    )
    assert moved > 1e-4, "the step should change the weights, otherwise the comparison proves nothing"
    for pa, pb in zip(model_big.parameters(), model_acc.parameters()):
        assert torch.allclose(pa.detach(), pb.detach(), atol=1e-5, rtol=1e-4)


def test_grad_accum_runs_one_record_per_optimiser_step():
    model = make_model()
    history = train(model, optimizer_for(model), random_batches(3), steps=3, grad_accum_steps=2, log_fn=quiet)
    records = train_records(history)
    assert [r["step"] for r in records] == [1, 2, 3]
    assert all(not r["skipped"] and r["grad_norm"] is not None for r in records)


def test_resume_with_grad_accum_matches_the_uninterrupted_run(tmp_path):
    base = make_model(min_loops=2, max_loops=4)
    batches = random_batches(3)  # 4 steps x 2 micro-batches = 8 draws, so the run crosses epoch boundaries
    ckpt = tmp_path / "ckpt"

    full = copy.deepcopy(base)
    hist_full = train(
        full, optimizer_for(full), batches, steps=4, grad_accum_steps=2, warmup_steps=1,
        checkpoint_dir=ckpt, checkpoint_every=2, keep_last=0, log_fn=quiet,
    )

    resumed = copy.deepcopy(base)
    hist_resumed = train(
        resumed, optimizer_for(resumed), batches, steps=4, grad_accum_steps=2, warmup_steps=1,
        resume_from=ckpt / "step_2", log_fn=quiet,
    )

    for pa, pb in zip(full.parameters(), resumed.parameters()):
        assert torch.allclose(pa.detach(), pb.detach(), atol=1e-5, rtol=1e-4)
    tail_full = [r["loss"] for r in train_records(hist_full) if r["step"] > 2]
    tail_resumed = [r["loss"] for r in train_records(hist_resumed) if r["step"] > 2]
    assert len(tail_full) == len(tail_resumed) == 2
    assert all(abs(a - b) < 1e-5 for a, b in zip(tail_full, tail_resumed))


def test_grad_accum_is_recorded_in_the_checkpoint_config(tmp_path):
    model = make_model()
    train(model, optimizer_for(model), random_batches(2), steps=2, grad_accum_steps=2,
          checkpoint_dir=tmp_path / "ckpt", checkpoint_every=2, log_fn=quiet)
    cfg = json.loads((tmp_path / "ckpt" / "step_2" / "train_config.json").read_text())
    assert cfg["grad_accum_steps"] == 2
    assert cfg["amp_dtype"] is None
    assert cfg["eval_by_depth"] is False


def test_invalid_grad_accum_is_rejected():
    model = make_model()
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match="grad_accum_steps"):
            train(model, optimizer_for(model), random_batches(2), steps=1, grad_accum_steps=bad, log_fn=quiet)


# ----------------------------------------------------------------------------- mixed precision


def test_bf16_autocast_on_cpu_runs_and_the_loss_is_finite():
    model = make_model(zero_aux=False)  # the full objective, so every loss term goes through autocast
    history = train(
        model, optimizer_for(model), random_batches(2), steps=3, amp_dtype="bf16",
        warmup_steps=1, log_fn=quiet,
    )
    records = train_records(history)
    assert len(records) == 3
    assert all(not r["skipped"] for r in records)
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in records)
    assert all(torch.isfinite(p.detach()).all() for p in model.parameters())


def test_bf16_loss_is_close_to_fp32_loss_on_the_first_step():
    base = make_model(min_loops=2, max_loops=2)
    batches = random_batches(1)
    fp32 = copy.deepcopy(base)
    loss_fp32 = train_records(train(fp32, optimizer_for(fp32), batches, steps=1, warmup_steps=0,
                                    log_fn=quiet))[0]["loss"]
    bf16 = copy.deepcopy(base)
    loss_bf16 = train_records(train(bf16, optimizer_for(bf16), batches, steps=1, warmup_steps=0,
                                    amp_dtype="bf16", log_fn=quiet))[0]["loss"]
    assert abs(loss_fp32 - loss_bf16) < 0.05 * max(1.0, abs(loss_fp32))


def test_unknown_amp_dtype_is_rejected():
    model = make_model()
    with pytest.raises(ValueError, match="amp_dtype"):
        train(model, optimizer_for(model), random_batches(2), steps=1, amp_dtype="fp8", log_fn=quiet)


def test_fp16_is_rejected_on_cpu():
    model = make_model()
    with pytest.raises(ValueError, match="CUDA"):
        train(model, optimizer_for(model), random_batches(2), steps=1, amp_dtype="fp16", log_fn=quiet)


# ----------------------------------------------------------------------------- per-depth eval (TRAIN-08)


def test_eval_by_depth_logs_every_training_depth_and_max_matches_eval_nll():
    model = make_model(min_loops=2, max_loops=4)
    logs = []
    history = train(
        model, optimizer_for(model), random_batches(3), steps=2, warmup_steps=1,
        eval_batches=random_batches(2, seed=7), eval_every=1, eval_by_depth=True, log_fn=logs.append,
    )
    evals = [r for r in history if r.get("kind") == "eval"]
    assert len(evals) == 2
    for e in evals:
        assert set(e["per_depth"]) == {"2", "3", "4"}
        assert abs(e["per_depth"]["4"]["nll"] - e["eval_nll"]) < 1e-5
        for depth in ("2", "3", "4"):
            assert e["per_depth"][depth]["perplexity"] > 0
    text = "\n".join(logs)
    for depth in (2, 3, 4):
        assert f"eval depth {depth} loops" in text


def test_eval_by_depth_is_off_by_default():
    model = make_model(min_loops=2, max_loops=4)
    logs = []
    history = train(
        model, optimizer_for(model), random_batches(3), steps=2, warmup_steps=1,
        eval_batches=random_batches(2, seed=7), eval_every=1, log_fn=logs.append,
    )
    evals = [r for r in history if r.get("kind") == "eval"]
    assert evals and all("per_depth" not in e for e in evals)
    assert not any("eval depth" in line for line in logs)


# ----------------------------------------------------------------------------- CLI


def test_cli_dry_run_with_grad_accum_bf16_and_depth_eval(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "train.py"), "--dry-run", "--steps", "3",
         "--grad-accum", "2", "--amp", "bf16", "--eval-by-depth", "--out-dir", str(out)],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "step 3/3" in proc.stdout
    assert "eval depth" in proc.stdout
    cfg = json.loads((out / "checkpoints" / "step_3" / "train_config.json").read_text())
    assert cfg["grad_accum_steps"] == 2
    assert cfg["amp_dtype"] == "bf16"
    assert cfg["eval_by_depth"] is True


def test_cli_rejects_zero_grad_accum(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "train.py"), "--dry-run", "--steps", "1",
         "--grad-accum", "0", "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT),
    )
    assert proc.returncode != 0
    assert "grad-accum" in proc.stderr
