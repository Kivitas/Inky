"""Regression tests for Phase 2 optimizer hygiene, speculative cache truncation, and training FLOP accounting."""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inky import HybridOptimizer, InkyConfig, InkyModel  # noqa: E402
from inky.optim import GROUP_ADAMW_NO_DECAY, GROUP_MUON, assign_parameter_groups  # noqa: E402
from inky.speculative import self_speculative_generate  # noqa: E402


def _tiny_config(**overrides) -> InkyConfig:
    defaults = dict(
        vocab_size=64,
        scratch_open_id=61,
        scratch_sep_id=62,
        scratch_close_id=63,
        eos_token_id=1,
        pad_token_id=0,
        hidden_size=32,
        num_attention_heads=2,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        num_pre_layers=1,
        num_recurrent_layers=1,
        num_post_layers=1,
        min_recurrent_loops=2,
        max_recurrent_loops=4,
        loop_dropout_prob=0.0,
        intermediate_size=32,
        num_shared_experts=1,
        num_routed_experts=2,
        num_active_routed=1,
        ssm_state_dim=8,
        max_position_embeddings=64,
    )
    defaults.update(overrides)
    return InkyConfig(**defaults)


def test_optimizer_keeps_a_gate_and_readout_out_of_muon():
    cfg = _tiny_config(ssm_mixer_layers=(0,))
    model = InkyModel(cfg)
    groups = dict((n, g) for n, g, _ in assign_parameter_groups(model))
    a_gates = {n: g for n, g in groups.items() if "a_gate" in n}
    readouts = {n: g for n, g in groups.items() if "readout" in n}
    assert a_gates and all(g == GROUP_ADAMW_NO_DECAY for g in a_gates.values()), a_gates
    assert readouts and all(g != GROUP_MUON for g in readouts.values()), readouts


def test_decoupled_grad_clipping_does_not_throttle_adamw_on_muon_spike():
    cfg = _tiny_config()
    model = InkyModel(cfg)
    opt = HybridOptimizer(model, adamw_lr=1e-2, muon_lr=0.02)

    # Set a tiny grad on an AdamW parameter and a huge spike grad on a Muon parameter.
    adamw_p = opt.adamw.param_groups[0]["params"][0]
    muon_p = opt.muon.param_groups[0]["params"][0]
    adamw_p.grad = torch.full_like(adamw_p, 1e-3)
    muon_p.grad = torch.full_like(muon_p, 1e4)

    adamw_before = adamw_p.grad.clone()
    opt.step(max_grad_norm=1.0)
    # AdamW grad norm was well below 1.0, so it must NOT be scaled down by Muon's 1e4 spike.
    assert torch.allclose(adamw_p.grad, adamw_before)


def test_speculative_cache_truncation_matches_greedy_and_saves_target_tokens():
    torch.manual_seed(0)
    cfg = _tiny_config(use_highway=False)
    model = InkyModel(cfg).eval()
    prompt = torch.randint(2, 50, (1, 6))
    greedy = model.generate(
        prompt, max_new_tokens=10, do_sample=False, no_repeat_ngram_size=0, eos_token_id=None, blocked_token_ids=[]
    )
    spec, stats = self_speculative_generate(
        model, prompt, max_new_tokens=10, draft_loops=2, draft_len=3, eos_token_id=None
    )
    assert torch.equal(spec, greedy)
    assert stats["target_tokens_fed"] >= prompt.shape[1]


def test_arch_budget_reports_train_flops_above_forward_flops():
    spec = importlib.util.spec_from_file_location("_inky_budget", ROOT / "scripts" / "arch_budget.py")
    budget = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = budget
    spec.loader.exec_module(budget)

    res = budget.main(
        ["--tiny", "--variants", "baseline", "--prefill-len", "16", "--contexts", "16", "--extrapolate-to", "32"]
    )
    row = res["variants"][0]
    assert row["train_flops_per_token"] > 3.0 * row["total_flops_per_token"]
