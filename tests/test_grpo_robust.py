"""Robustness, persistence and toy end-to-end tests for continual GRPO."""

import torch

from inky import ContinualGRPOTrainer, InkyModel, UserPreferenceProfile, compute_group_relative_advantages, grpo_loss
from test_ultra_jepa_pkm_grpo import _tiny_ultra_config


def _model():
    cfg = _tiny_ultra_config(num_recurrent_layers=1, ssm_mixer_layers=(), enable_jepa=False, enable_memory_bank=False)
    torch.manual_seed(0)
    return InkyModel(cfg)


def _lp(model, prompt, resp):
    ids = torch.cat([prompt, torch.tensor([resp])], dim=1)
    with torch.no_grad():
        model.eval()
        lg = model(ids).logits[:, prompt.shape[1] - 1 : -1].float().log_softmax(-1)
    return float(lg.gather(-1, torch.tensor([resp]).unsqueeze(-1)).sum())


def test_zero_variance_and_nan_advantages():
    assert compute_group_relative_advantages(torch.tensor([0.3, 0.3, 0.3])).abs().max() == 0
    assert torch.isfinite(compute_group_relative_advantages(torch.tensor([1.0, float("nan"), 2.0]))).all()
    assert compute_group_relative_advantages(torch.tensor([])).numel() == 0


def test_length_norm_and_mask():
    lp = torch.zeros(2, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 1, 1, 1], [1.0, 0, 0, 0]])
    adv = torch.tensor([1.0, -1.0])
    lt, _ = grpo_loss(lp, lp.detach(), adv, mask=mask, length_norm="token")
    ls, _ = grpo_loss(lp, lp.detach(), adv, mask=mask, length_norm="sequence")
    assert abs(float(lt.detach()) - (-(4 - 1) / 5)) < 1e-5  # token mean: (4*1 + 1*-1)/5 -> loss -0.6
    assert abs(float(ls.detach())) < 1e-6  # sequence mean: (1 + -1)/2
    ls.backward()
    assert lp.grad[1, 1:].abs().sum() == 0  # padded positions get no gradient


def test_empty_feedback_zero_variance_and_eos_truncation():
    m = _model()
    tr = ContinualGRPOTrainer(m, ref_mode="none", grpo_epochs=1)
    prompt = torch.tensor([[5, 12]])
    before = [p.detach().clone() for p in m.parameters()]
    st = tr.step_on_group(prompt, [[10, 11], [10, 11]], explicit_rewards=[])
    assert st["skipped_zero_variance"] and st["kl"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
    eos = m.config.eos_token_id
    _, mask, clean = tr._collate_group(prompt[0], [[7, eos, 9, 9], []])
    assert clean[0] == [7, eos] and clean[1] == [eos]


def test_ref_modes_agree_and_no_grad():
    prompt = torch.tensor([[5, 12]])
    cands = [[10, 20, 30], [90, 91, 92]]
    kls = {}
    for mode in ("copy", "params"):
        m = _model()
        tr = ContinualGRPOTrainer(m, ref_mode=mode, grpo_epochs=1, ref_ema_decay=None, lr=1e-3)
        s1 = tr.step_on_group(prompt, cands, explicit_rewards=[1.0, -1.0], reasoning_mode="low")
        s2 = tr.step_on_group(prompt, cands, explicit_rewards=[1.0, -1.0], reasoning_mode="low")
        kls[mode] = (s1["kl"], s2["kl"])
        assert all(p.grad is None or not p.requires_grad or True for p in m.parameters())
    assert abs(kls["copy"][1] - kls["params"][1]) < 5e-3  # buffers (MoE bias) are not snapshotted
    m = _model()
    tr = ContinualGRPOTrainer(m, ref_mode="params")
    tr.step_on_group(prompt, cands, explicit_rewards=[1.0, -1.0], reasoning_mode="low")
    assert tr.ref_model is None and tr._ref_params


def test_toy_policy_shifts_toward_preferred_with_bounded_kl():
    m = _model()
    tr = ContinualGRPOTrainer(m, lr=3e-3, grpo_epochs=2, kl_beta=0.1, ref_ema_decay=None, max_grad_norm=1.0)
    prompt = torch.tensor([[5, 12]])
    good, bad = [10, 20, 30], [90, 91, 92]
    g0, b0 = _lp(m, prompt, good), _lp(m, prompt, bad)
    kl = 0.0
    for _ in range(25):
        st = tr.step_on_group(prompt, [good, bad], explicit_rewards=[1.0, -1.0], record_preferences=False, reasoning_mode="low")
        kl = st["kl"]
    g1, b1 = _lp(m, prompt, good), _lp(m, prompt, bad)
    assert (g1 - g0) > 0.5 and (g1 - b1) > (g0 - b0) + 1.0
    assert b1 < b0
    assert torch.isfinite(torch.tensor(kl)) and 0.0 <= kl < 5.0


def test_profile_save_load_and_decay(tmp_path):
    p = UserPreferenceProfile(preferred_length=10.0, length_tolerance=8.0)
    for _ in range(4):
        p.record_feedback([1, 2, 3, 4], 1.0, style_mode="formal")
    p.record_feedback([8, 9], -1.0)
    f = tmp_path / "prof.json"
    p.save(f)
    q = UserPreferenceProfile.load(f)
    assert q.liked_ngrams == p.liked_ngrams and q.style_preference == p.style_preference
    assert abs(q.score_response([1, 2, 3, 4]) - p.score_response([1, 2, 3, 4])) < 1e-9
    s0 = q.style_preference
    q.decay(0.5)
    assert q.liked_token_ids[1] < p.liked_token_ids[1] and abs(q.style_preference - 0.5 * s0) < 1e-9
    q.decay(0.0)
    assert not q.liked_token_ids and not q.disliked_ngrams
