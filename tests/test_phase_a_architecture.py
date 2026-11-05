"""Tests for the Phase A architecture fixes: the highway's learned output gain, the scale-preserving loop mixture,
the sign MoE balance rule (with eval and skipped-step counters kept out of it), and the diagnostics.

Run with: python -B -m pytest tests/test_phase_a_architecture.py -q -p no:cacheprovider
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from inky import HybridOptimizer, InkyConfig, InkyModel, assign_parameter_groups, loop_diagnostics
from inky.diagnostics import format_diagnostics, format_diagnostics_line
from inky.layers import SharedExpertMoE, StateSpaceMemoryHighway
from inky.train import train

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
OPEN, SEP, CLOSE = 508, 509, 510


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512, scratch_open_id=OPEN, scratch_sep_id=SEP, scratch_close_id=CLOSE, hidden_size=64,
        num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16, q_lora_rank=32,
        kv_lora_rank=24, intermediate_size=96, num_routed_experts=4, num_active_routed=2, ssm_state_dim=16,
        max_recurrent_loops=4, min_recurrent_loops=2, loop_dropout_prob=0.0, max_position_embeddings=256,
        arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4, eos_token_id=1, pad_token_id=1,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def make_model(**overrides) -> InkyModel:
    torch.manual_seed(0)
    return InkyModel(tiny_config(**overrides))


def ids(rows=2, seq=24, seed=1):
    return torch.randint(2, 500, (rows, seq), generator=torch.Generator().manual_seed(seed))


def quiet(_msg):
    return None


# ----------------------------------------------------------------------------- A1: highway gain


def test_highway_gain_defaults_to_zero_and_is_one_parameter():
    model = make_model()
    gain = model.core.ssm.out_gain
    assert tuple(gain.shape) == (1,) and float(gain) == 0.0
    no_highway = make_model(use_highway=False)
    assert no_highway.core.ssm is None
    assert model.count_parameters()["total"] - no_highway.count_parameters()["total"] > 1


def test_highway_with_zero_gain_leaves_the_stream_unchanged():
    torch.manual_seed(0)
    hw = StateSpaceMemoryHighway(32, 8, 1e-6)           # gain_init defaults to 0.0
    h = torch.randn(2, 10, 32)
    out, mem, prefix = hw(h, None, None, 0)
    assert torch.equal(out, h)
    assert mem.shape == (2, 10, 8) and prefix is not None      # the memory is still computed and carried


def test_highway_gain_init_reproduces_the_old_fixed_scale():
    torch.manual_seed(0)
    hw = StateSpaceMemoryHighway(32, 8, 1e-6, gain_init=0.1)
    h = torch.randn(2, 10, 32)
    out, mem, _ = hw(h, None, None, 0)
    legacy = h + 0.1 * hw.norm(hw.out_proj(mem))
    assert torch.allclose(out, legacy, atol=1e-7)


def test_zero_gain_gets_a_gradient_and_the_rest_of_the_highway_waits():
    model = make_model()
    model.train()
    x = ids()
    model(x, labels=x, num_loops=3).loss.backward()
    ssm = model.core.ssm
    assert ssm.out_gain.grad is not None and float(ssm.out_gain.grad.abs()) > 0.0
    for name in ("in_proj", "a_gate", "out_proj"):
        for p in getattr(ssm, name).parameters():
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, name


def test_gain_is_a_no_decay_adamw_parameter():
    model = make_model()
    groups = {name: group for name, group, _ in assign_parameter_groups(model)}
    assert groups["core.ssm.out_gain"] == "adamw_no_decay"


def test_one_optimizer_step_opens_the_gate_without_nans():
    model = make_model()
    opt = HybridOptimizer(model, adamw_lr=1e-2, muon_lr=0.02)
    x = ids()
    for _ in range(3):
        loss = model(x, labels=x, num_loops=3).loss
        loss.backward()
        opt.step(max_grad_norm=1.0)
        opt.zero_grad()
    assert float(model.core.ssm.out_gain) != 0.0
    assert all(torch.isfinite(p).all() for p in model.parameters())


# ----------------------------------------------------------------------------- A2: loop mixture


def test_mix_equals_the_plain_sum_when_all_loop_states_have_the_same_scale():
    model = make_model()
    torch.manual_seed(2)
    states = torch.randn(4, 2, 5, 64)
    states = states / states.pow(2).mean(-1, keepdim=True).sqrt() * 0.7
    w = torch.softmax(torch.randn(4, 2, 5), 0)
    plain = (w.unsqueeze(-1) * states).sum(0)
    assert torch.allclose(model._mix_loop_states(states, w), plain, atol=1e-5)


def test_mix_gives_each_loop_its_halting_share_whatever_its_scale():
    u1 = torch.zeros(64)
    u1[0] = 1.0
    u2 = torch.zeros(64)
    u2[1] = 1.0
    # loop 1 is ten times smaller than loop 2, both weighted 0.5, and the two point in orthogonal directions
    states = torch.stack([0.1 * u1 * 8.0, 1.0 * u2 * 8.0]).view(2, 1, 1, 64)
    w = torch.full((2, 1, 1), 0.5)

    plain = make_model(loop_mix_normalize=False)._mix_loop_states(states, w)[0, 0]
    assert float(plain[1]) / float(plain[0]) == pytest.approx(10.0, rel=1e-3)      # loop 2 dominates 10:1

    mixed = make_model(loop_mix_normalize=True)._mix_loop_states(states, w)[0, 0]
    assert float(mixed[1]) / float(mixed[0]) == pytest.approx(1.0, rel=1e-2)       # shares follow p_n


def test_mix_output_scale_is_the_weighted_mean_rms_times_the_direction():
    model = make_model()
    torch.manual_seed(3)
    states = torch.randn(3, 1, 4, 64) * torch.tensor([0.2, 0.5, 1.0]).view(3, 1, 1, 1)
    w = torch.softmax(torch.randn(3, 1, 4), 0)
    rms = model._state_rms(states)
    expected = (w.unsqueeze(-1) * states / rms).sum(0) * (w.unsqueeze(-1) * rms).sum(0)
    assert torch.allclose(model._mix_loop_states(states, w), expected, atol=1e-6)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("gain", [0.0, 0.3])
def test_early_exit_without_threshold_matches_forward(normalize, gain):
    model = make_model(loop_mix_normalize=normalize).eval()
    with torch.no_grad():
        for p in model.halting.parameters():
            p.add_(torch.randn_like(p) * 0.3)
        model.core.ssm.out_gain.fill_(gain)
    x = ids(rows=1, seq=20)
    with torch.no_grad():
        full = model(x, return_hidden=True).hidden
        early = model.forward_early_exit(x, halt_threshold=None, return_hidden=True).hidden
    assert torch.allclose(full, early, atol=1e-5)


@pytest.mark.parametrize("normalize", [True, False])
def test_cached_decode_matches_full_forward(normalize):
    model = make_model(loop_mix_normalize=normalize).eval()
    with torch.no_grad():
        model.core.ssm.out_gain.fill_(0.3)
    x = ids(rows=1, seq=16)
    with torch.no_grad():
        full = model(x, return_hidden=True).hidden
        first = model(x[:, :6], use_cache=True, return_hidden=True)
        parts = [first.hidden]
        for t in range(6, 16):
            parts.append(model(x[:, t:t + 1], past=first.past, return_hidden=True).hidden)
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-5)


def test_normalized_mix_keeps_the_gradient_finite():
    model = make_model().train()
    x = ids()
    out = model(x, labels=x, num_loops=4)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


# ----------------------------------------------------------------------------- A3: balance rule


def moe_with_counts(counts, **overrides):
    moe = SharedExpertMoE(tiny_config(num_routed_experts=4, balance_rate=0.01, **overrides))
    moe.load_counts.copy_(torch.tensor(counts, dtype=torch.float32))
    return moe


def test_sign_rule_moves_every_bias_by_exactly_the_rate():
    moe = moe_with_counts([9.0, 1.0, 0.0, 0.0])           # uniform is 0.25: expert 0 over, 1 to 3 under
    assert moe.balance_update == "sign"
    moe.update_balance_bias()
    assert torch.allclose(moe.balance_bias, torch.tensor([-0.01, 0.01, 0.01, 0.01]), atol=1e-8)


def test_sign_rule_does_not_move_a_bias_at_exactly_uniform_load():
    moe = moe_with_counts([5.0, 5.0, 5.0, 5.0])
    moe.update_balance_bias()
    assert torch.equal(moe.balance_bias, torch.zeros(4))


def test_proportional_rule_is_still_available_and_matches_the_old_formula():
    moe = moe_with_counts([9.0, 1.0, 0.0, 0.0], balance_update="proportional")
    load = torch.tensor([0.9, 0.1, 0.0, 0.0])
    moe.update_balance_bias()
    assert torch.allclose(moe.balance_bias, 0.01 * (0.25 - load), atol=1e-8)


def test_unknown_balance_update_is_rejected():
    with pytest.raises(ValueError, match="balance_update"):
        InkyConfig(balance_update="momentum")


def test_reset_moe_load_clears_every_moe_counter():
    model = make_model().eval()
    with torch.no_grad():
        model(ids())
    moes = [m for m in model.modules() if isinstance(m, SharedExpertMoE)]
    assert moes and all(float(m.load_counts.sum()) > 0 for m in moes)
    model.reset_moe_load()
    assert all(float(m.load_counts.sum()) == 0.0 for m in moes)


def _counts_at_each_update(monkeypatch, **train_kwargs):
    """Total routed assignments each balance update sees, for one tiny training run."""
    model = make_model()
    seen = []
    original = SharedExpertMoE.update_balance_bias

    def spy(self):
        if self is first_moe:
            seen.append(float(self.load_counts.sum()))
        return original(self)

    first_moe = next(m for m in model.modules() if isinstance(m, SharedExpertMoE))
    monkeypatch.setattr(SharedExpertMoE, "update_balance_bias", spy)
    batches = [ids(seed=s) for s in range(3)]
    train(model, HybridOptimizer(model, adamw_lr=1e-3), batches, steps=3, warmup_steps=0, log_fn=quiet,
          max_grad_norm=None, **train_kwargs)
    return seen


def test_eval_passes_do_not_leak_into_the_balance_update(monkeypatch):
    without_eval = _counts_at_each_update(monkeypatch)
    with_eval = _counts_at_each_update(monkeypatch, eval_batches=[ids(seed=9)], eval_every=1)
    assert len(without_eval) == len(with_eval) == 3
    assert with_eval == without_eval


def test_a_skipped_step_does_not_leak_into_the_next_balance_update(monkeypatch):
    model = make_model()
    first_moe = next(m for m in model.modules() if isinstance(m, SharedExpertMoE))     # the pre layer: B*S*top_k
    seen = []
    original = SharedExpertMoE.update_balance_bias

    def spy(self):
        if self is first_moe:
            seen.append(float(self.load_counts.sum()))
        return original(self)

    monkeypatch.setattr(SharedExpertMoE, "update_balance_bias", spy)
    real_forward = model.forward
    calls = {"n": 0}

    def forward(*args, **kwargs):
        out = real_forward(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:                                   # the second training step gets a NaN loss
            out.loss = out.loss * float("nan")
        return out

    model.forward = forward
    history = train(model, HybridOptimizer(model, adamw_lr=1e-3), [ids(seed=s) for s in range(3)], steps=3,
                    warmup_steps=0, log_fn=quiet, max_grad_norm=None)
    skipped = [r["step"] for r in history if r["kind"] == "train" and r["skipped"]]
    assert skipped == [2]
    assert seen == [2 * 24 * 2, 2 * 24 * 2]                   # steps 1 and 3 count only their own tokens


# ----------------------------------------------------------------------------- A4: diagnostics


def test_diagnostics_report_shapes_and_values():
    model = make_model()
    d = loop_diagnostics(model, ids(), num_loops=3)
    assert [r["loop"] for r in d["loops"]] == [1, 2, 3]
    assert d["embedding_rms"] > 0 and d["pre_rms"] > 0 and d["mixed_rms"] > 0
    assert d["highway_gain"] == 0.0
    assert all(r["highway_share"] == 0.0 for r in d["loops"])                     # the gate starts closed
    assert sum(r["weight"] for r in d["loops"]) == pytest.approx(1.0, abs=1e-5)
    assert sum(r["scale_share"] for r in d["loops"]) == pytest.approx(1.0, abs=1e-5)
    assert all(r["rel_change"] >= 0 for r in d["loops"])
    assert d["loops"][1]["in_rms"] == pytest.approx(d["loops"][0]["out_rms"], rel=1e-5)
    assert {e["name"] for e in d["expert_load"]} == {
        "pre_layers.0.mlp", "core.layers.0.mlp", "core.layers.1.mlp", "post_layers.0.mlp"}
    for e in d["expert_load"]:
        assert sum(e["load"]) == pytest.approx(1.0, abs=1e-5) and e["min"] <= e["max"]
    json.dumps(d)                                                                   # plain numbers only


def test_diagnostics_show_the_highway_share_once_the_gain_is_open():
    model = make_model(highway_gain_init=0.1)
    d = loop_diagnostics(model, ids())
    assert d["highway_gain"] == pytest.approx(0.1)
    assert all(r["highway_share"] > 0 for r in d["loops"])


def test_diagnostics_do_not_disturb_the_model():
    model = make_model().train()
    model(ids())                                                                    # fill the training counters
    counts = [m.load_counts.clone() for m in model.modules() if isinstance(m, SharedExpertMoE)]
    loads = [m.expert_load.clone() for m in model.modules() if isinstance(m, SharedExpertMoE)]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    loop_diagnostics(model, ids(seed=5))
    assert model.training
    moes = [m for m in model.modules() if isinstance(m, SharedExpertMoE)]
    for m, kept_counts, kept_load in zip(moes, counts, loads):
        assert torch.equal(m.load_counts, kept_counts) and torch.equal(m.expert_load, kept_load)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    for m in model.modules():                                                       # every hook was removed
        assert not m._forward_hooks and not m._forward_pre_hooks


def test_diagnostics_without_highway_dense_or_per_loop_pass():
    d = loop_diagnostics(make_model(use_highway=False), ids())
    assert d["highway_gain"] is None and all(r["highway_share"] is None for r in d["loops"])
    assert loop_diagnostics(make_model(use_moe=False), ids())["expert_load"] == []
    mor = loop_diagnostics(make_model(routing_mode="mor"), ids())
    assert mor["loops"] == [] and mor["routing_mode"] == "mor"
    assert "no per-loop numbers" in format_diagnostics(mor)


def test_diagnostic_text_formats():
    d = loop_diagnostics(make_model(), ids(), num_loops=3)
    text = format_diagnostics(d)
    assert "rel_change" in text and "experts core.layers.0.mlp" in text
    line = format_diagnostics_line(7, d)
    assert line.startswith("step 7 diag:") and "out_rms [" in line and "expert_load min" in line


def test_train_diagnose_every_records_without_changing_the_run():
    batches = [ids(seed=s) for s in range(4)]

    def run(**kw):
        model = make_model()
        logs = []
        history = train(model, HybridOptimizer(model, adamw_lr=1e-3), batches, steps=4, warmup_steps=0,
                        log_fn=logs.append, max_grad_norm=None, **kw)
        return history, logs, model

    plain, _, m_plain = run()
    diag, logs, m_diag = run(diagnose_every=2)
    records = [r for r in diag if r["kind"] == "diagnostics"]
    assert [r["step"] for r in records] == [2, 4]
    assert records[0]["loops"] and any("diag:" in line for line in logs)
    assert [r["loss"] for r in plain if r["kind"] == "train"] == [r["loss"] for r in diag if r["kind"] == "train"]
    assert all(torch.equal(a, b) for a, b in zip(m_plain.state_dict().values(), m_diag.state_dict().values()))


def test_train_rejects_a_bad_diagnose_every():
    model = make_model()
    with pytest.raises(ValueError, match="diagnose_every"):
        train(model, HybridOptimizer(model), [ids()], steps=1, diagnose_every=-1, log_fn=quiet)


# ----------------------------------------------------------------------------- scripts/diagnose.py


def _load_diagnose():
    spec = importlib.util.spec_from_file_location("_inky_script_diagnose", SCRIPTS / "diagnose.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnose_script_tiny_prints_and_writes_json(tmp_path, capsys):
    diagnose = _load_diagnose()
    out = tmp_path / "diag.json"
    result = diagnose.main(["--tiny", "--seq-len", "16", "--num-loops", "3", "--json", str(out)])
    printed = capsys.readouterr().out
    assert "rel_change" in printed and "highway gain" in printed
    assert json.loads(out.read_text())["loops"] == result["loops"] and len(result["loops"]) == 3


def test_diagnose_script_set_overrides_restore_the_earlier_behaviour():
    diagnose = _load_diagnose()
    new = diagnose.main(["--tiny", "--seq-len", "16"])
    legacy = diagnose.main(["--tiny", "--seq-len", "16", "--set", "highway_gain_init=0.1",
                            "--set", "loop_mix_normalize=False"])
    assert new["highway_gain"] == 0.0 and legacy["highway_gain"] == pytest.approx(0.1)
    assert new["loops"][0]["highway_share"] == 0.0
    assert legacy["loops"][0]["highway_share"] > 0.0


def test_diagnose_script_rejects_conflicting_and_oversized_arguments():
    diagnose = _load_diagnose()
    with pytest.raises(SystemExit):
        diagnose.main(["--checkpoint", "nowhere", "--tiny"])
    with pytest.raises(SystemExit):
        diagnose.main(["--tiny", "--seq-len", "100000"])
