import torch

from inky.train import _clip_per_group


def test_per_group_clip_matches_hybrid_semantics():
    class Opt:  # minimal stand-in with the two optimiser groups
        def __init__(self, a, m):
            self.adamw = torch.optim.SGD([a], lr=0.1)
            self.muon = torch.optim.SGD([m], lr=0.1)

    a, m = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    a.grad, m.grad = torch.full((4,), 0.1), torch.full((4,), 10.0)
    n = _clip_per_group(Opt(a, m), 1.0)
    assert abs(float(a.grad.norm()) - 0.2) < 1e-6  # small group untouched by the big group's norm
    assert abs(float(m.grad.norm()) - 1.0) < 1e-4
    assert n > 20
