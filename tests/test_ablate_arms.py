"""Tests for the architecture-question variants added to scripts/ablate.py. Everything runs on CPU with tiny models."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from inky import InkyConfig, InkyModel
from inky.layers import GroupedQueryAttention, MultiLatentAttention

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_inky_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ablate = _load("ablate")

NEW_VARIANTS = ["gqa", "attn_gate", "last_loop", "inject", "big_experts", "core3x4", "core4x3",
                "v16k_core3x4", "v16k_core4x3"]

EXPECTED_EMPTY_BASE = {
    "gqa": {"attention_type": "gqa"},
    "attn_gate": {"attention_gate": True},
    "last_loop": {"routing_mode": "last"},
    "inject": {"loop_input_injection": True},
    # ff = 704, routed 8 -> 4 of width 1408, active 2 -> 1
    "big_experts": dict(shared_intermediate_size=704, intermediate_size=1408, num_routed_experts=4,
                        num_active_routed=1),
    # 2 layers x 6 loops = 12 applications; 3 layers -> round(12 / 3) = 4 loops; 4 layers -> 3 loops
    "core3x4": dict(num_recurrent_layers=3, max_recurrent_loops=4, min_recurrent_loops=2, ponder_prior_steps=3.0),
    "core4x3": dict(num_recurrent_layers=4, max_recurrent_loops=3, min_recurrent_loops=2, ponder_prior_steps=3.0),
    "v16k_core3x4": dict(vocab_size=16003, scratch_open_id=16000, scratch_sep_id=16001, scratch_close_id=16002,
                         eos_token_id=15974, pad_token_id=15974, num_recurrent_layers=3, max_recurrent_loops=4,
                         min_recurrent_loops=2, ponder_prior_steps=3.0),
    "v16k_core4x3": dict(vocab_size=16003, scratch_open_id=16000, scratch_sep_id=16001, scratch_close_id=16002,
                         eos_token_id=15974, pad_token_id=15974, num_recurrent_layers=4, max_recurrent_loops=3,
                         min_recurrent_loops=2, ponder_prior_steps=3.0),
}

# TINY: 2 recurrent layers x max 4 loops = 8 applications; intermediate 96, 4 routed experts, 2 active.
EXPECTED_TINY_BASE = {
    "gqa": {"attention_type": "gqa"},
    "attn_gate": {"attention_gate": True},
    "last_loop": {"routing_mode": "last"},
    "inject": {"loop_input_injection": True},
    "big_experts": dict(shared_intermediate_size=96, intermediate_size=192, num_routed_experts=2,
                        num_active_routed=1),
    "core3x4": dict(num_recurrent_layers=3, max_recurrent_loops=3, min_recurrent_loops=2, ponder_prior_steps=3.0),
    "core4x3": dict(num_recurrent_layers=4, max_recurrent_loops=2, min_recurrent_loops=2, ponder_prior_steps=2.0),
    "v16k_core3x4": dict(vocab_size=16003, scratch_open_id=16000, scratch_sep_id=16001, scratch_close_id=16002,
                         eos_token_id=15974, pad_token_id=15974, num_recurrent_layers=3, max_recurrent_loops=3,
                         min_recurrent_loops=2, ponder_prior_steps=3.0),
    "v16k_core4x3": dict(vocab_size=16003, scratch_open_id=16000, scratch_sep_id=16001, scratch_close_id=16002,
                         eos_token_id=15974, pad_token_id=15974, num_recurrent_layers=4, max_recurrent_loops=2,
                         min_recurrent_loops=2, ponder_prior_steps=2.0),
}


def _tiny_model(name: str) -> InkyModel:
    cfg = InkyConfig(**{**ablate.TINY_KWARGS, **ablate.resolve_variant(name, ablate.TINY_KWARGS)})
    torch.manual_seed(0)
    return InkyModel(cfg)


def _expert_numel(model: InkyModel) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if ".shared_experts." in n or ".routed_experts." in n)


# ----------------------------------------------------------------------------- registry and resolution

def test_new_variants_are_registered():
    for name in NEW_VARIANTS:
        assert name in ablate.VARIANTS, name


def test_new_variants_resolve_to_the_described_overrides():
    for name in NEW_VARIANTS:
        assert ablate.resolve_variant(name, {}) == EXPECTED_EMPTY_BASE[name], name
        assert ablate.resolve_variant(name, ablate.TINY_KWARGS) == EXPECTED_TINY_BASE[name], name


def test_core_shape_keeps_applications_about_constant():
    for name in ("core3x4", "core4x3"):
        for base in ({}, ablate.TINY_KWARGS):
            spec = ablate.resolve_variant(name, base)
            base_apps = base.get("num_recurrent_layers", 2) * base.get("max_recurrent_loops", 6)
            # rounding the loop count can move the application count by at most half a layer's loops
            assert abs(spec["num_recurrent_layers"] * spec["max_recurrent_loops"] - base_apps) <= (
                spec["num_recurrent_layers"] / 2), name


def test_compose_later_keys_win():
    composed = ablate._compose({"a": 1, "b": 1}, lambda base: {"b": 2, "c": base.get("x", 0)})
    assert composed({"x": 5}) == {"a": 1, "b": 2, "c": 5}


# ----------------------------------------------------------------------------- models

@pytest.mark.parametrize("name", NEW_VARIANTS)
def test_new_variant_builds_and_has_finite_loss(name):
    model = _tiny_model(name)
    torch.manual_seed(1)
    ids = torch.randint(2, 500, (2, 16))
    out = model(ids, labels=ids)
    assert torch.isfinite(out.loss), name


def test_big_experts_keeps_expert_parameters_and_dense_width():
    base_cfg = InkyConfig(**ablate.TINY_KWARGS)
    big_cfg = InkyConfig(**{**ablate.TINY_KWARGS, **ablate.resolve_variant("big_experts", ablate.TINY_KWARGS)})
    assert big_cfg.dense_ffn_size == base_cfg.dense_ffn_size
    base_model, big_model = InkyModel(base_cfg), InkyModel(big_cfg)
    assert _expert_numel(big_model) == _expert_numel(base_model) > 0


def test_gqa_variant_uses_grouped_query_attention_and_baseline_uses_mla():
    assert isinstance(_tiny_model("gqa").pre_layers[0].attn, GroupedQueryAttention)
    assert isinstance(_tiny_model("baseline").pre_layers[0].attn, MultiLatentAttention)


# ----------------------------------------------------------------------------- CLI

def test_new_variants_smoke_run(tmp_path):
    names = "gqa,attn_gate,last_loop,inject,big_experts,core3x4"
    result = ablate.main(["--synthetic", "--tiny", "--steps", "2", "--variants", names, "--out-dir", str(tmp_path)])
    rows = result["variants"]
    assert [r["variant"] for r in rows] == names.split(",")
    for row in rows:
        assert row["status"] == "ok", f"{row['variant']} skipped: {row.get('reason')}"
        assert row["final_loss"] is not None and math.isfinite(row["final_loss"]), row["variant"]
    assert (tmp_path / "ablation.json").is_file() and (tmp_path / "ablation.md").is_file()


def test_list_variants_prints_new_names(capsys):
    assert ablate.main(["--list-variants"]) is None
    printed = capsys.readouterr().out.split()
    for name in NEW_VARIANTS:
        assert name in printed, name


# ----------------------------------------------------------------------------- local window and hybrid SSM


def test_local_window_variant_follows_the_context_and_the_loop_count():
    assert ablate.resolve_variant("local_window", {}) == dict(local_window=1024, local_window_loops=(1, 2, 3, 4, 5))
    tiny = ablate.resolve_variant("local_window", ablate.TINY_KWARGS)
    assert tiny == dict(local_window=32, local_window_loops=(1, 2, 3))          # 256 // 8, loops 1 .. max - 1


def test_local_window_and_hybrid_ssm_build_and_give_a_finite_loss():
    import torch

    from inky import InkyConfig, InkyModel
    from inky.layers import SelectiveSSMMixer

    ids = torch.randint(2, 500, (2, 40))
    for name in ("local_window", "hybrid_ssm"):
        cfg = InkyConfig(**{**ablate.TINY_KWARGS, **ablate.resolve_variant(name, ablate.TINY_KWARGS)})
        model = InkyModel(cfg).train()
        assert torch.isfinite(model(ids, labels=ids).loss), name
    hybrid = InkyModel(InkyConfig(**{**ablate.TINY_KWARGS, **ablate.resolve_variant("hybrid_ssm", {})}))
    assert isinstance(hybrid.core.layers[1].attn, SelectiveSSMMixer)
    assert not isinstance(hybrid.core.layers[0].attn, SelectiveSSMMixer)
