"""Regression tests for optimiser parameter routing. Run with: python -m pytest tests/test_optim_fixes.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inky import HybridOptimizer, split_parameters  # noqa: E402
from inky.optim import (  # noqa: E402
    GROUP_ADAMW_DECAY,
    GROUP_ADAMW_NO_DECAY,
    GROUP_MUON,
    MIN_MUON_DIM,
    assign_parameter_groups,
    format_group_counts,
)
from test_optim import tiny_model  # noqa: E402


def _group_of(model, name_fragment: str):
    """Return {name: group} for every trainable parameter whose name contains the fragment."""
    return {
        name: group
        for name, group, _ in assign_parameter_groups(model)
        if name_fragment in name
    }


def test_router_lands_in_adamw():
    model = tiny_model()
    routers = _group_of(model, "router")
    assert routers, "tiny model should contain MoE router weights"
    assert all(g in (GROUP_ADAMW_DECAY, GROUP_ADAMW_NO_DECAY) for g in routers.values()), routers


def test_halting_gate_lands_in_adamw():
    model = tiny_model()
    gates = _group_of(model, "halting")
    assert "halting.gate.weight" in gates, gates
    assert gates["halting.gate.weight"] in (GROUP_ADAMW_DECAY, GROUP_ADAMW_NO_DECAY)
    assert gates["halting.gate.bias"] == GROUP_ADAMW_NO_DECAY


def test_routing_by_name_applies_even_when_the_matrix_is_large():
    # The name rule must not depend on the shape: a wide router-named matrix still goes to AdamW.
    class Wide(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(64, 64, bias=False)
            self.halting = nn.Module()
            self.halting.gate = nn.Linear(64, 64, bias=False)

    groups = {name: g for name, g, _ in assign_parameter_groups(Wide())}
    assert groups["router.weight"] != GROUP_MUON
    assert groups["halting.gate.weight"] != GROUP_MUON


def test_name_segment_match_does_not_catch_gate_proj():
    # SwiGLU and state-space gates are large hidden matrices and must stay in Muon.
    model = tiny_model()
    gate_proj = _group_of(model, "gate_proj")
    assert gate_proj, "tiny model should contain SwiGLU gate_proj weights"
    assert all(g == GROUP_MUON for n, g in gate_proj.items() if n.endswith(".weight")), gate_proj


def test_large_hidden_matrices_land_in_muon():
    model = tiny_model()
    for fragment in ("q_b_proj.weight", "kv_b_proj.weight", "o_proj.weight", "up_proj.weight",
                     "down_proj.weight", "in_proj.weight", "out_proj.weight"):
        groups = _group_of(model, fragment)
        assert groups, f"no parameter matching {fragment}"
        assert all(g == GROUP_MUON for g in groups.values()), (fragment, groups)


def test_small_2d_matrices_go_to_adamw_by_shape():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.tiny = nn.Linear(MIN_MUON_DIM - 1, 384, bias=False)  # min dim below threshold
            self.big = nn.Linear(MIN_MUON_DIM, 384, bias=False)       # at threshold: Muon

    groups = {name: g for name, g, _ in assign_parameter_groups(Net())}
    assert groups["tiny.weight"] == GROUP_ADAMW_DECAY
    assert groups["big.weight"] == GROUP_MUON


def test_min_muon_dim_is_a_knob_and_is_validated():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 384, bias=False)

    net = Net()
    assert len(split_parameters(net)[0]) == 0                      # default 16: min dim 8 is tiny
    assert len(split_parameters(net, min_muon_dim=8)[0]) == 1      # min dim 8 is not below 8
    assert len(split_parameters(net, min_muon_dim=9)[0]) == 0      # now below threshold
    try:
        split_parameters(net, min_muon_dim=0)
    except ValueError:
        pass
    else:
        raise AssertionError("min_muon_dim=0 should raise")


def test_arlo_binary_head_is_routed_to_adamw():
    # Arlo's binary head is a (1, hidden) readout, the same shape class as the halting gate.
    groups = _group_of(tiny_model(), "binary_head.weight")
    assert groups and all(g != GROUP_MUON for g in groups.values()), groups


def test_assignment_is_deterministic_across_builds():
    first = [(n, g) for n, g, _ in assign_parameter_groups(tiny_model())]
    second = [(n, g) for n, g, _ in assign_parameter_groups(tiny_model())]
    assert first == second
    assert first, "expected at least one parameter"


def test_optimizer_group_order_matches_split_and_is_stable():
    opt_a = HybridOptimizer(tiny_model())
    opt_b = HybridOptimizer(tiny_model())
    names_a = [n for n, _, _ in assign_parameter_groups(tiny_model())]
    names_b = [n for n, _, _ in assign_parameter_groups(tiny_model())]
    assert names_a == names_b
    assert [p.shape for p in opt_a.muon.param_groups[0]["params"]] == \
        [p.shape for p in opt_b.muon.param_groups[0]["params"]]


def test_group_counts_cover_every_trainable_parameter():
    model = tiny_model()
    opt = HybridOptimizer(model)
    counts = opt.group_counts()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert sum(c["params"] for c in counts.values()) == total
    assert counts[GROUP_MUON]["tensors"] == opt.stats["muon_tensors"]
    assert counts[GROUP_MUON]["params"] == opt.stats["muon_params"]
    assert counts[GROUP_ADAMW_DECAY]["tensors"] + counts[GROUP_ADAMW_NO_DECAY]["tensors"] == \
        opt.stats["adamw_tensors"]


def test_group_counts_survive_state_dict_round_trip():
    opt = HybridOptimizer(tiny_model())
    fresh = HybridOptimizer(tiny_model())
    fresh.load_state_dict(opt.state_dict())
    assert fresh.group_counts() == opt.group_counts()


def test_format_group_counts_mentions_every_group():
    text = format_group_counts(HybridOptimizer(tiny_model()).group_counts())
    for group in (GROUP_MUON, GROUP_ADAMW_DECAY, GROUP_ADAMW_NO_DECAY):
        assert group in text


def test_muon_lr_default_is_unchanged():
    # The default is unswept (documented), not changed by this fix.
    assert HybridOptimizer(tiny_model()).muon.param_groups[0]["lr"] == 0.02


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
