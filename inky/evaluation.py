"""Evaluation harness for a held-out language-model split.

  - split_heldout:  deterministic shuffle and split of a list of sequences.
  - perplexity:     mean next-token negative log likelihood and perplexity over batches.
  - evaluate:       perplexity plus the mean PonderNet loop count, from the same forward pass.
  - bits_per_token: convert a nat-based NLL to bits.

Losses are computed here from full logits, not from the model's own loss, so that every token
of every batch is weighted equally and the normalisation is explicit.
"""

import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import torch
import torch.nn.functional as F

from .modeling import InkyModel

T = TypeVar("T")


def split_heldout(
    sequences: Sequence[T],
    fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[T], List[T]]:
    """Shuffle deterministically with ``seed`` and split off a held-out part.

    The held-out size is ``max(1, round(fraction * n))``. Returns ``(train, heldout)``.
    """
    n = len(sequences)
    if n == 0:
        raise ValueError("cannot split an empty sequence list")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    k = max(1, round(fraction * n))
    heldout = [sequences[i] for i in perm[:k]]
    train = [sequences[i] for i in perm[k:]]
    return train, heldout


def bits_per_token(nll: float) -> float:
    """Convert a mean negative log likelihood in nats to bits per token."""
    return nll / math.log(2.0)


def _run(
    model: InkyModel,
    batches: Iterable[torch.Tensor],
    max_batches: Optional[int],
    num_loops: Optional[int],
    ignore_id: Optional[int] = None,
) -> Dict[str, float]:
    """Single pass over the batches. Returns summed NLL, token count, batch count and loop stats.

    Targets equal to ``ignore_id`` (padding) are left out of both the NLL sum and the token count.
    """
    ignore_index = -100 if ignore_id is None else ignore_id
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    tokens = 0
    count = 0
    loop_sum = 0.0
    loop_count = 0
    source = batches if max_batches is None else itertools.islice(batches, max_batches)
    try:
        with torch.no_grad():
            for ids in source:
                ids = ids.to(device)
                out = model(ids, num_loops=num_loops)
                logits = out.logits[:, :-1]
                targets = ids[:, 1:]
                vocab = logits.shape[-1]
                nll_sum = F.cross_entropy(
                    logits.reshape(-1, vocab).float(),
                    targets.reshape(-1),
                    reduction="sum",
                    ignore_index=ignore_index,
                )
                total_nll += float(nll_sum)
                tokens += int((targets != ignore_index).sum())
                count += 1
                if out.mean_loops is not None:
                    loop_sum += float(out.mean_loops)
                    loop_count += 1
    finally:
        model.train(was_training)
    if tokens == 0:
        raise ValueError("no tokens to evaluate: batches need at least one non-padding target token")
    return {
        "nll": total_nll / tokens,
        "tokens": tokens,
        "batches": count,
        "mean_loops": (loop_sum / loop_count) if loop_count else None,
    }


def perplexity(
    model: InkyModel,
    batches: Iterable[torch.Tensor],
    max_batches: Optional[int] = None,
    num_loops: Optional[int] = None,
    ignore_id: Optional[int] = None,
) -> Dict[str, float]:
    """Perplexity of ``model`` over ``batches`` (LongTensors of shape [B, S]).

    Returns ``{"nll", "perplexity", "tokens", "batches"}``. ``nll`` is the mean per-token
    negative log likelihood in nats. Target positions equal to ``ignore_id`` are not scored; pass the
    padding id if batches are padded. The default ``None`` scores every target. The model is run in
    eval mode and its previous mode is restored afterwards.
    """
    r = _run(model, batches, max_batches, num_loops, ignore_id)
    return {
        "nll": r["nll"],
        "perplexity": math.exp(r["nll"]),
        "tokens": r["tokens"],
        "batches": r["batches"],
    }


def evaluate(
    model: InkyModel,
    batches: Iterable[torch.Tensor],
    num_loops: Optional[int] = None,
    ignore_id: Optional[int] = None,
) -> Dict[str, float]:
    """``perplexity`` plus ``mean_loops``, the mean expected PonderNet loop count over batches.

    Both numbers come from the same forward pass per batch. ``ignore_id`` works as in ``perplexity``.
    """
    r = _run(model, batches, None, num_loops, ignore_id)
    return {
        "nll": r["nll"],
        "perplexity": math.exp(r["nll"]),
        "tokens": r["tokens"],
        "batches": r["batches"],
        "mean_loops": r["mean_loops"],
    }
