import torch

from inky.distill import chunked_distillation_loss, distillation_loss


def test_chunked_matches_full_value_and_grad():
    torch.manual_seed(0)
    h = torch.randn(2, 37, 16, requires_grad=True)
    w = torch.randn(50, 16)
    teacher = torch.randn(2, 37, 50)
    y = torch.randint(0, 50, (2, 37))
    y[0, :5] = -100
    full = distillation_loss(h @ w.T, teacher, targets=y)
    g_full, = torch.autograd.grad(full, h)
    for chunk in (1, 8, 1000):
        c = chunked_distillation_loss(h, teacher, lambda x: x @ w.T, targets=y, chunk_size=chunk)
        g_c, = torch.autograd.grad(c, h)
        assert torch.allclose(full, c, atol=1e-5) and torch.allclose(g_full, g_c, atol=1e-5)


def test_chunked_all_ignored_is_zero_and_connected():
    h = torch.randn(1, 4, 8, requires_grad=True)
    out = chunked_distillation_loss(h, torch.randn(1, 4, 9), lambda x: x[..., :9], targets=torch.full((1, 4), -100))
    assert out == 0 and out.requires_grad
