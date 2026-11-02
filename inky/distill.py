"""Distillation loss for training Inky from a larger teacher (Phase 3 preparation).

Logits are [B, S, V] or [N, V]. The teacher is treated as a fixed target: its logits are
detached inside ``distillation_loss`` so no gradient reaches it.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def shift_for_next_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align logits and targets so position t predicts token t+1.

    Matches ``InkyModel._lm_loss``: drop the last logit position and the first target position.
    Inputs are [B, S, V] logits and [B, S] targets. Returns (student, teacher, targets) shifted.
    """
    return student_logits[:, :-1], teacher_logits[:, :-1], targets[:, 1:]


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Knowledge-distillation loss.

    soft = KL(teacher || student) on temperature-scaled distributions, times temperature**2.
    If ``targets`` is given, only positions whose target is not ``ignore_index`` count, and the
    result is ``alpha * soft + (1 - alpha) * hard`` where ``hard`` is cross-entropy against the
    targets. If ``targets`` is None, the soft loss alone is returned.

    Computation is in float32. Teacher logits are detached.
    """
    vocab = student_logits.shape[-1]
    s = student_logits.reshape(-1, vocab).float()
    t = teacher_logits.reshape(-1, vocab).float().detach()
    y = None
    if targets is not None:
        y = targets.reshape(-1)
        keep = y != ignore_index
        s, t, y = s[keep], t[keep], y[keep]

    if s.shape[0] == 0:
        # No valid positions: zero loss that is still connected to the student graph.
        zero = student_logits.sum() * 0.0
        return zero.float()

    scale = float(temperature) ** 2
    log_s = F.log_softmax(s / temperature, dim=-1)
    log_t = F.log_softmax(t / temperature, dim=-1)
    soft = F.kl_div(log_s, log_t, reduction="batchmean", log_target=True) * scale

    if y is None:
        return soft

    hard = F.cross_entropy(s, y, reduction="mean")
    return alpha * soft + (1.0 - alpha) * hard
