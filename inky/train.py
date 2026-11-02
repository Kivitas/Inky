"""Training loop for Inky: warmup-stable-decay learning rate, loop curriculum, periodic eval and checkpoints.

  - wsd_lr_scale:      linear warmup, flat plateau, linear decay to zero (a multiplier on the base LRs).
  - loop_curriculum:   recurrent loop count ramps linearly from min to max over the first part of training.
  - train:             the optimisation loop. Steps are 1-based in the history and checkpoint names.
"""

import itertools
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch

from .evaluation import evaluate


def wsd_lr_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_fraction: float = 0.2,
) -> float:
    """Warmup-stable-decay multiplier in [0, 1] for a 0-based ``step`` in ``[0, total_steps - 1]``.

    Linear warmup from 0 to 1 over ``warmup_steps`` (step 0 gives 0), flat at 1, then a linear decay
    over the last ``decay_fraction * total_steps`` steps that reaches 0 at the final step.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if not 0.0 <= decay_fraction <= 1.0:
        raise ValueError(f"decay_fraction must be in [0, 1], got {decay_fraction}")

    step = max(0, step)
    warm = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0

    decay_len = int(decay_fraction * total_steps)
    if decay_len > 0:
        remaining = max(0, total_steps - 1 - step)
        decay = min(1.0, remaining / decay_len)
    else:
        decay = 1.0

    return float(min(1.0, max(0.0, min(warm, decay))))


def loop_curriculum(
    step: int,
    total_steps: int,
    min_loops: int,
    max_loops: int,
    warmup_fraction: float = 0.5,
) -> int:
    """Recurrent loop count for a 0-based ``step``.

    Rises linearly from ``min_loops`` (step 0) to ``max_loops`` over the first ``warmup_fraction``
    of training, then stays at ``max_loops``. Always an int in ``[min_loops, max_loops]``.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if not 1 <= min_loops <= max_loops:
        raise ValueError(f"need 1 <= min_loops <= max_loops, got {min_loops}, {max_loops}")
    if not 0.0 < warmup_fraction <= 1.0:
        raise ValueError(f"warmup_fraction must be in (0, 1], got {warmup_fraction}")

    ramp = max(1.0, (total_steps - 1) * warmup_fraction)
    progress = min(1.0, max(0.0, step) / ramp)
    loops = min_loops + round(progress * (max_loops - min_loops))
    return int(min(max_loops, max(min_loops, loops)))


def train(
    model,
    optimizer,
    train_batches: Iterable[torch.Tensor],
    steps: int,
    eval_batches: Optional[Iterable[torch.Tensor]] = None,
    eval_every: int = 0,
    warmup_steps: int = 0,
    decay_fraction: float = 0.2,
    max_grad_norm: Optional[float] = 1.0,
    use_loop_curriculum: bool = False,
    log_every: int = 10,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_every: int = 0,
    log_fn: Callable[[str], Any] = print,
) -> List[Dict[str, Any]]:
    """Run ``steps`` optimiser steps and return the history.

    ``train_batches`` is materialised once and cycled when exhausted. Each training record is
    ``{"kind": "train", "step", "loss", "lr_scale", "num_loops", "grad_norm"}``; ``num_loops`` is
    None unless the loop curriculum is on. Eval records are ``{"kind": "eval", "step", "eval_nll",
    "eval_perplexity"}``. Eval runs every ``eval_every`` steps and at the last step. Checkpoints go to
    ``checkpoint_dir / f"step_{step}"`` every ``checkpoint_every`` steps and at the last step.
    The model's previous train/eval mode is restored afterwards.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    batches = list(train_batches)
    if not batches:
        raise ValueError("train_batches is empty")
    cycle = itertools.cycle(batches)

    eval_list = None
    if eval_batches is not None and eval_every > 0:
        eval_list = list(eval_batches)

    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_root is not None and checkpoint_every > 0:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    cfg = model.config
    device = next(model.parameters()).device
    history: List[Dict[str, Any]] = []

    was_training = model.training
    model.train()
    try:
        for step in range(1, steps + 1):
            idx = step - 1
            lr_scale = wsd_lr_scale(idx, steps, warmup_steps, decay_fraction)
            optimizer.set_lr_scale(lr_scale)

            num_loops = None
            if use_loop_curriculum:
                num_loops = loop_curriculum(idx, steps, cfg.min_recurrent_loops, cfg.max_recurrent_loops)

            ids = next(cycle).to(device)
            out = model(input_ids=ids, labels=ids, num_loops=num_loops)
            loss = out.loss
            loss.backward()
            grad_norm = optimizer.step(max_grad_norm=max_grad_norm)
            optimizer.zero_grad()
            model.update_moe_balance()

            record = {
                "kind": "train",
                "step": step,
                "loss": float(loss.detach()),
                "lr_scale": lr_scale,
                "num_loops": num_loops,
                "grad_norm": grad_norm,
            }
            history.append(record)
            is_last = step == steps
            if log_every > 0 and (step % log_every == 0 or is_last):
                log_fn(
                    f"step {step}/{steps} loss {record['loss']:.4f} lr_scale {lr_scale:.3f} "
                    f"num_loops {num_loops} grad_norm {grad_norm}"
                )

            if eval_list is not None and (step % eval_every == 0 or is_last):
                ev = evaluate(model, eval_list, num_loops=None)
                eval_record = {
                    "kind": "eval",
                    "step": step,
                    "eval_nll": ev["nll"],
                    "eval_perplexity": ev["perplexity"],
                }
                history.append(eval_record)
                log_fn(f"step {step} eval nll {ev['nll']:.4f} perplexity {ev['perplexity']:.2f}")
                model.train()

            if ckpt_root is not None and checkpoint_every > 0 and (step % checkpoint_every == 0 or is_last):
                model.save_pretrained(ckpt_root / f"step_{step}")
    finally:
        model.train(was_training)

    return history
