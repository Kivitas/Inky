"""Train only the Arlo head on labelled decision examples, with the encoder frozen."""

from typing import Iterable

import torch

from .decider import ArloDecider


def train_arlo(
    decider: ArloDecider,
    batches: Iterable,
    task: str = "choice",
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 1,
) -> list:
    """Fit the Arlo head. Returns the mean loss of each step.

    ``batches`` is an iterable of batches; a batch is a list of example dicts:
      * task "choice": {"context": ids, "options": [ids, ...], "gold": index, "question": ids (optional)}
      * task "binary": {"context": ids, "gold": 0 or 1, "question": ids (optional)}

    The encoder runs under ``torch.no_grad`` and, if it is an ``nn.Module`` reachable as ``.model`` or
    itself, is put in eval mode for the run and restored afterwards. Only ``decider.arlo`` changes.
    """
    if task not in ("choice", "binary"):
        raise ValueError(f"unknown task {task!r}; expected 'choice' or 'binary'")
    batches = [list(b) for b in batches]
    if not any(batches):
        raise ValueError("train_arlo needs at least one non-empty batch")

    enc = decider.encoder
    modules = [m for m in (getattr(enc, "model", None), getattr(enc, "fn", None), enc)
               if isinstance(m, torch.nn.Module)]
    prev = [(m, m.training) for m in modules]
    was_training = decider.arlo.training
    optimiser = torch.optim.AdamW(decider.arlo.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    try:
        for m in modules:
            m.eval()
        decider.arlo.train()
        for _ in range(epochs):
            for batch in batches:
                if not batch:
                    continue
                terms = []
                for ex in batch:
                    with torch.no_grad():
                        query, states, options = decider.features(
                            ex["context"], task, ex.get("options"), ex.get("question"))
                    gold = int(ex["gold"])
                    dev = query.device
                    if task == "choice":
                        n = options.shape[1]
                        if not 0 <= gold < n:
                            raise ValueError(f"choice gold {gold} is outside the {n} options")
                        terms.append(decider.arlo.training_loss(
                            query, options, states, torch.tensor([gold], device=dev), "choice"))
                    else:
                        if gold not in (0, 1):
                            raise ValueError(f"binary gold must be 0 or 1, got {gold}")
                        terms.append(decider.arlo.training_loss(
                            query, None, states, torch.tensor([float(gold)], device=dev), "binary"))
                loss = torch.stack(terms).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                decider.mark_trained()
                losses.append(float(loss.detach()))
    finally:
        for m, was in prev:
            m.train(was)
        decider.arlo.train(was_training)
    return losses
