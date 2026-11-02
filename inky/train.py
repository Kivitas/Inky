"""Training loop for Inky: warmup-stable-decay learning rate, loop curriculum, periodic eval and checkpoints.

  - wsd_lr_scale:         linear warmup, flat plateau, linear decay to zero (a multiplier on the base LRs).
  - loop_curriculum:      recurrent loop count ramps linearly from min to max over the first part of training.
  - save/load_training_state: model weights, optimiser state and the step counter, for resuming a run.
  - train:                the optimisation loop. Steps are 1-based in the history and checkpoint names.
"""

import itertools
import json
import math
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


TRAIN_STATE_FILE = "train_state.json"
OPTIMIZER_FILE = "optimizer.pt"
MODEL_DIR = "model"


def save_training_state(
    path: Union[str, Path],
    model,
    optimizer,
    step: int,
    history: List[Dict[str, Any]],
) -> None:
    """Write everything needed to resume: ``path/model`` (weights), ``path/optimizer.pt`` and ``path/train_state.json``."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(root / MODEL_DIR)
    torch.save(optimizer.state_dict(), root / OPTIMIZER_FILE)
    (root / TRAIN_STATE_FILE).write_text(json.dumps({"step": int(step), "history": history}))


def load_training_state(path: Union[str, Path], model, optimizer) -> int:
    """Restore weights and optimiser state saved by ``save_training_state``. Returns the saved step."""
    root = Path(path)
    state = json.loads((root / TRAIN_STATE_FILE).read_text())
    fresh = type(model).from_pretrained(root / MODEL_DIR)
    model.load_state_dict(fresh.state_dict())
    optimizer.load_state_dict(torch.load(root / OPTIMIZER_FILE, map_location="cpu", weights_only=False))
    return int(state["step"])


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
    max_skipped_steps: int = 50,
    resume_from: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Run optimiser steps up to the total ``steps`` and return the history of the steps run.

    ``steps`` is the total run length: schedules (LR, curriculum) are always computed from it. With
    ``resume_from`` set to a directory written by ``save_training_state``, weights and optimiser state
    are restored and the run continues from the step after the saved one, so the returned history only
    holds the new steps.

    ``train_batches`` is materialised once and cycled when exhausted. Each training record is
    ``{"kind": "train", "step", "loss", "lr_scale", "num_loops", "grad_norm", "skipped"}``; ``num_loops``
    is None unless the loop curriculum is on. A step whose loss is not finite skips the backward and the
    optimiser step, is recorded with ``skipped=True`` and is not checkpointed. A non-finite gradient norm
    is recorded as ``grad_norm`` (the step is still taken). More than ``max_skipped_steps`` skipped steps
    raises RuntimeError. Eval records are ``{"kind": "eval", "step", "eval_nll", "eval_perplexity"}``.
    Eval runs every ``eval_every`` steps and at the last step. Checkpoints go to
    ``checkpoint_dir / f"step_{step}"`` every ``checkpoint_every`` steps and at the last step, as full
    training state (see ``save_training_state``). The model's previous train/eval mode is restored afterwards.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if max_skipped_steps < 0:
        raise ValueError(f"max_skipped_steps must be >= 0, got {max_skipped_steps}")

    batches = list(train_batches)
    if not batches:
        raise ValueError("train_batches is empty")

    start_step = 1
    if resume_from is not None:
        start_step = load_training_state(resume_from, model, optimizer) + 1
        if start_step > steps:
            raise ValueError(f"resume_from is at step {start_step - 1}, which is not before steps={steps}")
    # Skip the batches already consumed before the saved step so the data order continues.
    cycle = itertools.islice(itertools.cycle(batches), start_step - 1, None)

    eval_list = None
    if eval_batches is not None and eval_every > 0:
        eval_list = list(eval_batches)

    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_root is not None and checkpoint_every > 0:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    cfg = model.config
    device = next(model.parameters()).device
    history: List[Dict[str, Any]] = []
    skipped_count = 0

    was_training = model.training
    model.train()
    try:
        for step in range(start_step, steps + 1):
            idx = step - 1
            lr_scale = wsd_lr_scale(idx, steps, warmup_steps, decay_fraction)
            optimizer.set_lr_scale(lr_scale)

            num_loops = None
            if use_loop_curriculum:
                num_loops = loop_curriculum(idx, steps, cfg.min_recurrent_loops, cfg.max_recurrent_loops)

            ids = next(cycle).to(device)
            out = model(input_ids=ids, labels=ids, num_loops=num_loops)
            loss = out.loss
            loss_value = float(loss.detach())

            skipped = not math.isfinite(loss_value)
            if skipped:
                optimizer.zero_grad()
                skipped_count += 1
                record = {
                    "kind": "train",
                    "step": step,
                    "loss": loss_value,
                    "lr_scale": lr_scale,
                    "num_loops": num_loops,
                    "grad_norm": None,
                    "skipped": True,
                }
                history.append(record)
                log_fn(f"step {step}/{steps} non-finite loss {loss_value}, step skipped ({skipped_count} so far)")
                if skipped_count > max_skipped_steps:
                    raise RuntimeError(
                        f"{skipped_count} steps skipped for non-finite loss, more than max_skipped_steps="
                        f"{max_skipped_steps}; stopping at step {step}"
                    )
            else:
                loss.backward()
                grad_norm = optimizer.step(max_grad_norm=max_grad_norm)
                optimizer.zero_grad()
                model.update_moe_balance()

                record = {
                    "kind": "train",
                    "step": step,
                    "loss": loss_value,
                    "lr_scale": lr_scale,
                    "num_loops": num_loops,
                    "grad_norm": grad_norm,
                    "skipped": False,
                }
                if grad_norm is not None and not math.isfinite(grad_norm):
                    record["nonfinite_grad"] = True
                history.append(record)

                is_last = step == steps
                if log_every > 0 and (step % log_every == 0 or is_last):
                    log_fn(
                        f"step {step}/{steps} loss {record['loss']:.4f} lr_scale {lr_scale:.3f} "
                        f"num_loops {num_loops} grad_norm {grad_norm}"
                    )

            is_last = step == steps
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

            if (
                ckpt_root is not None
                and not skipped
                and checkpoint_every > 0
                and (step % checkpoint_every == 0 or is_last)
            ):
                save_training_state(ckpt_root / f"step_{step}", model, optimizer, step, history)
    finally:
        model.train(was_training)

    return history
