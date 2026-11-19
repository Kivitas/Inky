import warnings

import pytest
import torch
import torch.nn as nn

from arlo import Arlo, ArloConfig, ArloDecider, FnEncoder, evaluate, train_arlo

V, D = 20, 24


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.rnn = nn.GRU(D, D, batch_first=True)

    def forward(self, ids):
        return self.rnn(self.emb(ids))[0]


def make(**kw):
    torch.manual_seed(0)
    return ArloDecider(Toy(), hidden_size=D, slots=3, dim=16, heads=2, pointer_rank=8, **kw)


def test_config_validation():
    with pytest.raises(ValueError):
        ArloConfig(hidden_size=0)
    with pytest.raises(ValueError):
        ArloDecider(Toy())


def test_engine_shapes_and_option_order_invariance():
    cfg = ArloConfig(hidden_size=D, slots=3, dim=16, heads=2, pointer_rank=8)
    a = Arlo(cfg).eval()
    q, o, s = torch.randn(2, D), torch.randn(2, 4, D), torch.randn(2, 5, D)
    sc = a.choice_scores(q, o, s)
    assert sc.shape == (2, 4)
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(a.choice_scores(q, o[:, perm], s), sc[:, perm], atol=1e-5)
    assert a.binary_logit(q, s).shape == (2,)


def test_mask_ignores_masked_states():
    a = Arlo(ArloConfig(hidden_size=D, slots=3, dim=16, heads=2, pointer_rank=8)).eval()
    q, s = torch.randn(1, D), torch.randn(1, 6, D)
    mask = torch.tensor([[True] * 4 + [False] * 2])
    s2 = s.clone()
    s2[:, 4:] = 99.0
    assert torch.allclose(a.binary_logit(q, s, mask), a.binary_logit(q, s2, mask), atol=1e-5)


def test_choice_and_yes_no_outputs():
    dec = make()
    with pytest.warns(UserWarning):
        p = dec.choice([1, 2, 3], [[4], [5, 6], [7]])
    assert p.shape == (3,) and abs(float(p.sum()) - 1) < 1e-5
    with pytest.warns(UserWarning):
        y = dec.yes_no([1, 2, 3], question=[4])
    assert 0 <= float(y) <= 1


def test_empty_option_and_wrong_width_rejected():
    dec = make()
    dec.mark_trained()
    with pytest.raises(ValueError):
        dec.choice([1, 2], [[3], []])
    with pytest.raises(ValueError):
        dec.choice([1, 2], [])
    bad = ArloDecider(lambda ids: torch.zeros(*ids.shape, D + 1), hidden_size=D)
    bad.mark_trained()
    with pytest.raises(ValueError, match="hidden_size"):
        bad.yes_no([1])


def test_padding_does_not_change_scores():
    dec = make()
    dec.mark_trained()
    dec.eval()
    a = dec.choice([1, 2, 3], [[4], [5, 6, 7]])
    b = dec.choice([1, 2, 3], [[4]] + [[5, 6, 7]])
    assert torch.allclose(a, b)
    solo = dec.choice([1, 2, 3], [[4], [9, 9, 9, 9, 9]])  # longer sibling => more padding on option 0
    assert torch.allclose(dec.features([1, 2, 3], "choice", [[4], [5]])[2][0, 0],
                          dec.features([1, 2, 3], "choice", [[4], [9, 9, 9, 9]])[2][0, 0], atol=1e-5)
    assert solo.shape == (2,)


def test_training_updates_only_arlo_and_restores_modes():
    dec = make()
    lm = dec.encoder.fn
    lm.train()
    before = {k: v.clone() for k, v in lm.state_dict().items()}
    data = [{"context": [1, 2, 3, i % 5 + 2], "options": [[2], [3]], "gold": i % 2} for i in range(8)]
    losses = train_arlo(dec, [data[:4], data[4:]], "choice", epochs=2)
    assert len(losses) == 4 and dec.trained
    assert lm.training
    assert all(torch.equal(before[k], v) for k, v in lm.state_dict().items())
    assert all(p.grad is None for p in lm.parameters())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dec.choice([1, 2], [[3], [4]])        # trained: no warning


def test_learns_a_simple_binary_task():
    torch.manual_seed(1)
    dec = ArloDecider(Toy(), hidden_size=D, slots=4, dim=16, heads=2, pointer_rank=8)
    data = [{"context": [1, 5, 5, 5] if i % 2 else [1, 9, 9, 9], "gold": i % 2} for i in range(32)]
    train_arlo(dec, [data[i:i + 8] for i in range(0, 32, 8)], "binary", lr=1e-2, epochs=40)
    assert evaluate(dec, data, "binary")["accuracy"] == 1.0


def test_bad_gold_rejected():
    dec = make()
    with pytest.raises(ValueError):
        train_arlo(dec, [[{"context": [1], "options": [[2], [3]], "gold": 5}]], "choice")
    with pytest.raises(ValueError):
        train_arlo(dec, [[{"context": [1], "gold": 2}]], "binary")
    with pytest.raises(ValueError):
        train_arlo(dec, [[]], "choice")


def test_save_load_roundtrip(tmp_path):
    dec = make()
    dec.mark_trained()
    path = tmp_path / "arlo.pt"
    dec.save(path)
    new = ArloDecider.load(path, dec.encoder)
    assert new.trained and new.config == dec.config
    opts = [[4], [5, 6]]
    assert torch.allclose(dec.choice([1, 2, 3], opts), new.choice([1, 2, 3], opts))
    assert not any(k.startswith("emb") or k.startswith("rnn") for k in dec.state_dict())


def test_encoder_adapters():
    lm = Toy()

    class Out:
        def __init__(self, h):
            self.hidden = h

    class Wrapped(nn.Module):
        def forward(self, ids, return_hidden=False):
            return Out(lm(ids))

    from arlo import AttrEncoder
    dec = ArloDecider(AttrEncoder(Wrapped(), return_hidden=True), hidden_size=D)
    dec.mark_trained()
    assert dec.yes_no([1, 2]).ndim == 0
    dec.set_encoder(FnEncoder(lm))
    assert dec.pick([1, 2], [[3], [4]]) in (0, 1)
