"""Tests for inky/distill.py. Run with: python -m pytest tests/test_distill.py -q"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky.distill import distillation_loss, shift_for_next_token  # noqa: E402


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_identical_student_and_teacher_soft_loss_is_zero():
    logits = _rand(2, 5, 11)
    loss = distillation_loss(logits, logits.clone(), targets=None)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_non_negative():
    for seed in range(5):
        s = _rand(3, 4, 9, seed=seed)
        t = _rand(3, 4, 9, seed=seed + 100)
        assert distillation_loss(s, t).item() >= -1e-7
        targets = torch.randint(0, 9, (3, 4), generator=torch.Generator().manual_seed(seed))
        assert distillation_loss(s, t, targets).item() >= -1e-7


def test_teacher_receives_no_gradient():
    student = _rand(2, 3, 7).requires_grad_(True)
    teacher = _rand(2, 3, 7, seed=1).requires_grad_(True)
    targets = torch.tensor([[1, 2, 3], [4, -100, 6]])
    loss = distillation_loss(student, teacher, targets)
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_ignored_positions_do_not_change_loss():
    student = _rand(2, 4, 8)
    teacher = _rand(2, 4, 8, seed=3)
    targets = torch.tensor([[1, -100, 3, 4], [-100, 6, -100, 0]])
    base = distillation_loss(student, teacher, targets)

    perturbed = student.clone()
    perturbed[0, 1] += 50.0
    perturbed[1, 0] -= 30.0
    perturbed[1, 2] += 10.0
    other = distillation_loss(perturbed, teacher, targets)
    assert torch.allclose(base, other, atol=1e-6)


def test_alpha_one_is_soft_only_and_alpha_zero_is_hard_only():
    student = _rand(2, 4, 8)
    teacher = _rand(2, 4, 8, seed=5)
    targets = torch.tensor([[1, -100, 3, 4], [0, 6, -100, 2]])
    T = 2.0

    keep = targets.reshape(-1) != -100
    s = student.reshape(-1, 8)[keep]
    t = teacher.reshape(-1, 8)[keep]
    y = targets.reshape(-1)[keep]

    # alpha=1 must equal the soft loss over the same non-ignored rows.
    soft_on_kept = distillation_loss(s, t, targets=None, temperature=T)
    alpha1 = distillation_loss(student, teacher, targets, temperature=T, alpha=1.0)
    assert torch.allclose(alpha1, soft_on_kept, atol=1e-6)

    # alpha=0 must equal plain cross-entropy over the same non-ignored rows.
    hard_expected = F.cross_entropy(s, y)
    alpha0 = distillation_loss(student, teacher, targets, temperature=T, alpha=0.0)
    assert torch.allclose(alpha0, hard_expected, atol=1e-6)


def test_shapes_3d_and_2d_match():
    B, S, V = 2, 5, 10
    student3 = _rand(B, S, V)
    teacher3 = _rand(B, S, V, seed=7)
    targets3 = torch.randint(0, V, (B, S), generator=torch.Generator().manual_seed(2))
    targets3[0, 0] = -100

    loss3 = distillation_loss(student3, teacher3, targets3)
    loss2 = distillation_loss(
        student3.reshape(-1, V), teacher3.reshape(-1, V), targets3.reshape(-1)
    )
    assert loss3.dim() == 0 and loss2.dim() == 0
    assert torch.allclose(loss3, loss2, atol=1e-6)
    assert loss3.dtype == torch.float32

    soft3 = distillation_loss(student3, teacher3)
    soft2 = distillation_loss(student3.reshape(-1, V), teacher3.reshape(-1, V))
    assert torch.allclose(soft3, soft2, atol=1e-6)


def test_shift_for_next_token_aligns_positions():
    B, S, V = 2, 6, 5
    s = _rand(B, S, V)
    t = _rand(B, S, V, seed=9)
    y = torch.arange(B * S).reshape(B, S)
    s2, t2, y2 = shift_for_next_token(s, t, y)
    assert s2.shape == (B, S - 1, V)
    assert t2.shape == (B, S - 1, V)
    assert y2.shape == (B, S - 1)
    assert torch.equal(s2, s[:, :-1]) and torch.equal(t2, t[:, :-1])
    assert torch.equal(y2, y[:, 1:])
