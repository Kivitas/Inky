"""Distillation loss for training Inky from a larger teacher (Phase 3 preparation).

Logits are [B, S, V] or [N, V]. The teacher is treated as a fixed target: its logits are
detached inside ``distillation_loss`` so no gradient reaches it.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


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


def chunked_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    targets: Optional[torch.Tensor] = None,
    chunk_size: int = 1024,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Memory-bounded knowledge-distillation loss over hidden states.

    Projects ``student_hidden`` through ``project_fn`` in chunks of ``chunk_size`` tokens so
    long-sequence (e.g. 8k) distillation never materializes full ``[B, S, V]`` float32 logits.
    Mathematically equivalent to ``distillation_loss(project_fn(student_hidden), ...)``.
    """
    h_flat = student_hidden.reshape(-1, student_hidden.shape[-1])
    t_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    y_flat = None
    if targets is not None:
        y_flat = targets.reshape(-1)
        keep = y_flat != ignore_index
        h_flat, t_flat, y_flat = h_flat[keep], t_flat[keep], y_flat[keep]

    total_tokens = h_flat.shape[0]
    if total_tokens == 0:
        return (student_hidden.sum() * 0.0).float()

    chunk = max(1, int(chunk_size))
    scale = float(temperature) ** 2
    soft_sum = student_hidden.new_zeros((), dtype=torch.float32)
    hard_sum = student_hidden.new_zeros((), dtype=torch.float32)
    def _chunk_terms(h_c, t_c, y_c):
        s_c = project_fn(h_c).float()
        log_s = F.log_softmax(s_c / temperature, dim=-1)
        log_t = F.log_softmax(t_c.float().detach() / temperature, dim=-1)
        soft_c = F.kl_div(log_s, log_t, reduction="sum", log_target=True) * scale
        hard_c = F.cross_entropy(s_c, y_c, reduction="sum") if y_c is not None else soft_c.new_zeros(())
        return soft_c, hard_c

    use_ckpt = torch.is_grad_enabled() and h_flat.requires_grad
    for start in range(0, total_tokens, chunk):
        end = min(start + chunk, total_tokens)
        y_c = y_flat[start:end] if y_flat is not None else None
        if use_ckpt:
            # Recompute each chunk's [chunk, V] logits in backward so they are not all kept alive.
            soft_c, hard_c = torch.utils.checkpoint.checkpoint(
                _chunk_terms, h_flat[start:end], t_flat[start:end], y_c, use_reentrant=False
            )
        else:
            soft_c, hard_c = _chunk_terms(h_flat[start:end], t_flat[start:end], y_c)
        soft_sum = soft_sum + soft_c
        hard_sum = hard_sum + hard_c

    soft = soft_sum / total_tokens
    if y_flat is None:
        return soft
    return alpha * soft + (1.0 - alpha) * (hard_sum / total_tokens)
