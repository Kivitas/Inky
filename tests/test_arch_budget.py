"""Tests for scripts/arch_budget.py. Everything runs on CPU with tiny models."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from inky import InkyConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_inky_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


budget = _load("arch_budget")

SMALL = ["--tiny", "--variants", "baseline,dense", "--prefill-len", "32", "--contexts", "16,32,64",
         "--extrapolate-to", "256"]


@pytest.fixture(scope="module")
def small_budget():
    return budget.main(SMALL)


def test_rows_are_finite_and_dense_is_smaller_with_matched_body_flops(small_budget):
    result = small_budget
    rows = {r["name"]: r for r in result["variants"]}
    assert list(rows) == ["baseline", "dense"]
    for row in rows.values():
        for key in ("params", "body_flops_per_token", "head_flops_per_token", "total_flops_per_token",
                    "decode_flops_largest", "decode_flops_extrapolated", "cache_bytes_per_token",
                    "cache_mib_bf16_extrapolated"):
            assert math.isfinite(row[key]) and row[key] > 0, key
        assert 0 < row["embedding_share"] < 1
        assert row["total_flops_per_token"] == pytest.approx(row["body_flops_per_token"] + row["head_flops_per_token"])
    assert rows["dense"]["params"] < rows["baseline"]["params"]
    # Dense is built to match the MoE's active width. Measured on the tiny config the gap is well under 1%.
    body_base, body_dense = rows["baseline"]["body_flops_per_token"], rows["dense"]["body_flops_per_token"]
    assert abs(body_dense - body_base) / body_base < 0.05, (body_base, body_dense)


def test_decode_flops_grow_with_context(small_budget):
    row = small_budget["variants"][0]
    by_ctx = row["decode_flops_by_context"]
    assert by_ctx[64] > by_ctx[16]
    assert row["decode_flops_extrapolated"] >= by_ctx[64]
    assert row["decode_attention_flops_per_context_token"] > 0
    assert row["measured_contexts"] == [16, 32, 64] and row["largest_context"] == 64


def test_cache_bytes_per_token_matches_the_analytic_value():
    row = budget.main(["--tiny", "--variants", "baseline", "--prefill-len", "32", "--contexts", "16,32",
                       "--extrapolate-to", "256"])["variants"][0]
    cfg = InkyConfig(**budget.TINY_KWARGS)
    entries = cfg.num_pre_layers + cfg.max_recurrent_loops * cfg.num_recurrent_layers + cfg.num_post_layers
    assert row["cache_bytes_per_token"] == entries * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 4
    assert row["cache_mib_bf16_extrapolated"] == pytest.approx(row["cache_bytes_per_token"] * 256 / 2**20 / 2)


def test_contexts_past_the_position_limit_are_skipped():
    row = budget.main(["--tiny", "--variants", "baseline", "--prefill-len", "16", "--contexts", "16,300",
                       "--extrapolate-to", "256"])["variants"][0]
    assert row["skipped_contexts"] == [300] and row["measured_contexts"] == [16]
    assert row["decode_flops_extrapolated"] is None      # one point is not a line


def test_unknown_variant_exits_through_parser_error():
    with pytest.raises(SystemExit):
        budget.main(["--tiny", "--variants", "no_such_variant"])


def test_bad_set_value_exits_through_parser_error():
    with pytest.raises(SystemExit):
        budget.main(["--tiny", "--set", "definitely_not_a_key=1"])


def test_build_error_is_reported_and_the_other_variants_still_run(monkeypatch, capsys):
    monkeypatch.setitem(budget.VARIANTS, "broken", {"attention_type": "nope"})
    result = budget.main(["--tiny", "--variants", "broken,baseline", "--prefill-len", "16", "--contexts", "16,32",
                          "--extrapolate-to", "64"])
    broken, ok = result["variants"]
    assert broken["name"] == "broken" and "attention_type" in broken["error"]
    assert ok["name"] == "baseline" and "error" not in ok and ok["params"] > 0
    assert "Errored variants" in capsys.readouterr().out


def test_json_output_matches_the_returned_rows(tmp_path):
    out = tmp_path / "budget.json"
    md = tmp_path / "budget.md"
    result = budget.main([*SMALL, "--json", str(out), "--markdown", str(md)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [r["name"] for r in saved["variants"]] == [r["name"] for r in result["variants"]] == ["baseline", "dense"]
    assert saved["variants"][0]["params"] == result["variants"][0]["params"]
    assert "| baseline |" in md.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- SDPA attention is counted


def test_the_flop_counter_counts_sdpa_attention_with_the_standard_formula():
    import torch
    import torch.nn.functional as F

    q, k, v = torch.randn(1, 6, 1, 64), torch.randn(1, 2, 500, 64), torch.randn(1, 2, 500, 48)
    with budget.flop_counter() as fc:
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert fc.get_total_flops() == 2 * 1 * 6 * 1 * 500 * 64 + 2 * 1 * 6 * 1 * 500 * 48


def test_gqa_decode_attention_cost_matches_the_analytic_slope():
    rows = budget.main(["--tiny", "--variants", "gqa", "--prefill-len", "32", "--contexts", "16,64,128",
                        "--extrapolate-to", "256"])["variants"]
    row = rows[0]
    cfg = InkyConfig(**budget.TINY_KWARGS)
    applications = cfg.num_pre_layers + cfg.max_recurrent_loops * cfg.num_recurrent_layers + cfg.num_post_layers
    per_cached_token = applications * 2 * cfg.num_attention_heads * (cfg.qk_head_dim + cfg.v_head_dim)
    assert row["decode_attention_flops_per_context_token"] == pytest.approx(per_cached_token, rel=0.01)
    assert row["decode_flops_by_context"][128] > row["decode_flops_by_context"][16]
