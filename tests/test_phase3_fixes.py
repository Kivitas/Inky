"""Regression tests for Phase 3 and Phase 4 fixes."""

from pathlib import Path
from types import SimpleNamespace

import tomllib
import torch
import torch.nn as nn

import scripts.ablate as ablate
import scripts.train as train_script
from inky.config import InkyConfig
from inky.distill import chunked_distillation_loss, distillation_loss
from inky.layers import SelectiveSSMMixer


def test_ablate_unrolled_param_match_and_mqa_variants():
    for name in ("dense_unrolled", "dense_param_match", "mqa"):
        assert name in ablate.VARIANTS

    unrolled = ablate.resolve_variant("dense_unrolled", {})
    assert unrolled["use_moe"] is False
    assert unrolled["num_recurrent_layers"] == 12
    assert unrolled["min_recurrent_loops"] == 1
    assert unrolled["max_recurrent_loops"] == 1
    cfg_unrolled = InkyConfig(**unrolled)
    assert cfg_unrolled.num_recurrent_layers == 12

    param_match = ablate.resolve_variant("dense_param_match", {})
    assert param_match["use_moe"] is False
    assert param_match["dense_intermediate_size"] == 6336
    cfg_pm = InkyConfig(**param_match)
    assert cfg_pm.dense_ffn_size == 6336

    mqa = ablate.resolve_variant("mqa", {})
    assert mqa["attention_type"] == "gqa"
    assert mqa["num_kv_heads"] == 1
    cfg_mqa = InkyConfig(**mqa)
    assert cfg_mqa.num_kv_heads == 1


def test_selective_ssm_fast_decode_matches_chunked_scan_and_fp32_carry():
    torch.manual_seed(0)
    cfg = InkyConfig(hidden_size=16, num_attention_heads=2, num_kv_heads=2, ssm_state_dim=8)
    ssm = SelectiveSSMMixer(cfg)
    ssm.scan_chunk = 4
    ssm.eval()

    x_prefill = torch.randn(2, 5, 16)
    x_step = torch.randn(2, 1, 16)
    with torch.no_grad():
        _, past = ssm(x_prefill)
        assert past[0].dtype == torch.float32

        # Fast path: S == 1, seg=None
        y_fast, state_fast = ssm(x_step, past=past)
        assert state_fast[0].dtype == torch.float32

        # Reference: _chunked_scan directly
        u = ssm.in_proj(x_step)
        log_a = torch.nn.functional.logsigmoid(ssm.decay_proj(x_step))
        g = torch.sigmoid(ssm.gate_proj(x_step))
        y_scan, state_scan = ssm._chunked_scan(log_a, g, u, past[0], seg=None)
        y_scan = ssm.out_proj(y_scan + ssm.skip * u)

        torch.testing.assert_close(y_fast, y_scan, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(state_fast[0], state_scan, atol=1e-6, rtol=1e-5)


def test_selective_ssm_bfloat16_keeps_fp32_carry():
    torch.manual_seed(1)
    cfg = InkyConfig(hidden_size=16, num_attention_heads=2, num_kv_heads=2, ssm_state_dim=8)
    ssm = SelectiveSSMMixer(cfg).to(torch.bfloat16)
    ssm.scan_chunk = 3
    x_seq = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    x_step = torch.randn(1, 1, 16, dtype=torch.bfloat16)
    with torch.no_grad():
        y_seq, past = ssm(x_seq)
        assert y_seq.dtype == torch.bfloat16
        assert past[0].dtype == torch.float32
        y_step, state_step = ssm(x_step, past=past)
        assert y_step.dtype == torch.bfloat16
        assert state_step[0].dtype == torch.float32


def test_chunked_distillation_loss_matches_distillation_loss():
    torch.manual_seed(42)
    B, S, D, V = 2, 10, 16, 32
    proj_ref = nn.Linear(D, V, bias=False)
    proj_chk = nn.Linear(D, V, bias=False)
    proj_chk.load_state_dict(proj_ref.state_dict())

    h_ref = torch.randn(B, S, D, requires_grad=True)
    h_chk = h_ref.detach().clone().requires_grad_(True)
    teacher_logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    targets[0, :3] = -100

    loss_ref = distillation_loss(proj_ref(h_ref), teacher_logits, targets=targets, temperature=2.0, alpha=0.5)
    loss_chk = chunked_distillation_loss(
        h_chk, teacher_logits, proj_chk, targets=targets, chunk_size=3, temperature=2.0, alpha=0.5
    )
    torch.testing.assert_close(loss_chk, loss_ref, atol=1e-5, rtol=1e-5)

    loss_ref.backward()
    loss_chk.backward()
    torch.testing.assert_close(h_chk.grad, h_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(proj_chk.weight.grad, proj_ref.weight.grad, atol=1e-5, rtol=1e-5)

    # Soft-only (targets=None)
    soft_ref = distillation_loss(proj_ref(h_ref.detach()), teacher_logits, targets=None, temperature=1.5)
    soft_chk = chunked_distillation_loss(
        h_chk.detach(), teacher_logits, proj_chk, targets=None, chunk_size=4, temperature=1.5
    )
    torch.testing.assert_close(soft_chk, soft_ref, atol=1e-5, rtol=1e-5)

    # All ignored targets returns zero connected to graph
    all_ignored = torch.full((B, S), -100, dtype=torch.long)
    zero_loss = chunked_distillation_loss(h_chk, teacher_logits, proj_chk, targets=all_ignored)
    assert zero_loss.item() == 0.0
    assert zero_loss.requires_grad


def test_train_cli_gradient_checkpointing_and_pyproject_scripts():
    args = SimpleNamespace(
        embedding_dropout=0.0,
        residual_dropout=0.0,
        gradient_checkpointing=True,
        config_overrides={},
    )
    assert train_script._config_kwargs(args)["gradient_checkpointing"] is True

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts_tbl = pyproject["project"]["scripts"]
    for cmd in ("inky-train", "inky-sample", "inky-ablate", "inky-diagnose", "inky-bench"):
        assert cmd in scripts_tbl
