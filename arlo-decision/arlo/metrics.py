"""Tiny evaluation helpers for decision examples (same format as ``train_arlo``)."""

from typing import Iterable

from .decider import ArloDecider


def evaluate(decider: ArloDecider, examples: Iterable, task: str = "choice", threshold: float = 0.5) -> dict:
    """Accuracy over labelled examples. Returns {"items": n, "accuracy": float}."""
    import warnings

    n = correct = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for ex in examples:
            if task == "choice":
                pred = decider.pick(ex["context"], ex["options"], ex.get("question"))
            elif task == "binary":
                pred = int(decider.decide(ex["context"], ex.get("question"), threshold))
            else:
                raise ValueError(f"unknown task {task!r}; expected 'choice' or 'binary'")
            n += 1
            correct += int(pred == int(ex["gold"]))
    return {"items": n, "accuracy": correct / n if n else None}
