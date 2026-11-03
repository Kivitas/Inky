"""Evaluation harness for a held-out language-model split.

  - split_heldout:        deterministic shuffle and split of a list of sequences.
  - perplexity:           mean next-token negative log likelihood and perplexity over batches.
  - evaluate:             perplexity plus the token-weighted mean PonderNet loop count, from the same pass.
  - perplexity_by_loops:  perplexity at several recurrent loop counts over the same batches.
  - early_exit_drift:     NLL of forward_early_exit against the full forward, one sequence at a time.
  - bits_per_token:       convert a nat-based NLL to bits.

Losses are computed here from the model's outputs, not from the model's own loss, so that every
scored token is weighted equally and the normalisation is explicit.

Memory: when the model returns hidden states (``return_hidden``), the vocabulary projection runs in
chunks of ``_LOGIT_CHUNK`` sequence positions, so full ``[B, S, V]`` fp32 logits are never built. A
model that returns only logits is still scored correctly, but its logits already exist in full, so
no memory is saved in that case.

Padding: ``ignore_id`` only removes padded *targets* from the score. Pad tokens still enter the
context. InkyModel.forward has no padding mask, so they are attended to. If a model's forward accepts
an ``attention_mask`` argument, ``ignore_id`` also passes ``attention_mask = (ids != ignore_id)``.
"""

import inspect
import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import torch
import torch.nn.functional as F

from .modeling import InkyModel

T = TypeVar("T")

# Sequence positions projected onto the vocabulary at once.
_LOGIT_CHUNK = 128


def split_heldout(
    sequences: Sequence[T],
    fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[T], List[T]]:
    """Shuffle deterministically with ``seed`` and split off a held-out part.

    The held-out size is ``max(1, round(fraction * n))``. Returns ``(train, heldout)``.
    Raises ValueError when the split would leave the train part empty (MET-07), for example one
    sequence, or two sequences with a fraction that rounds up to both.
    """
    n = len(sequences)
    if n == 0:
        raise ValueError("cannot split an empty sequence list")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    k = max(1, round(fraction * n))
    if k >= n:
        raise ValueError(
            f"split_heldout needs at least 2 sequences to leave a non-empty train split; got {n} "
            f"with fraction={fraction} (held-out size {k})"
        )
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    heldout = [sequences[i] for i in perm[:k]]
    train = [sequences[i] for i in perm[k:]]
    return train, heldout


def bits_per_token(nll: float) -> float:
    """Convert a mean negative log likelihood in nats to bits per token."""
    return nll / math.log(2.0)


def _accepts(fn, name: str) -> bool:
    """True if ``fn`` has a named parameter ``name`` (``**kwargs`` does not count)."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _nll_sum(model: Any, out: Any, ids: torch.Tensor, ignore_index: int) -> Tuple[torch.Tensor, int]:
    """Summed next-token NLL (nats) and scored-token count for ``ids`` given the outputs ``out``.

    Uses ``out.hidden`` with the model's own vocabulary projection, chunk by chunk, when present;
    otherwise slices ``out.logits``.
    """
    targets = ids[:, 1:]
    if out.hidden is not None:
        project = model._project  # same projection (incl. soft-capping) as the logits path
        source = out.hidden[:, :-1]
    elif out.logits is not None:
        project = _identity
        source = out.logits[:, :-1]
    else:
        raise ValueError("model output has neither hidden states nor logits")
    nll = source.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.shape[1], _LOGIT_CHUNK):
        stop = start + _LOGIT_CHUNK
        logits = project(source[:, start:stop]).float()
        tgt = targets[:, start:stop]
        nll = nll + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
            reduction="sum",
            ignore_index=ignore_index,
        )
    tokens = int((targets != ignore_index).sum())
    return nll, tokens


def _run(
    model: InkyModel,
    batches: Iterable[torch.Tensor],
    max_batches: Optional[int],
    num_loops: Optional[int],
    ignore_id: Optional[int] = None,
) -> Dict[str, float]:
    """Single pass over the batches. Returns summed NLL, token count, batch count and loop stats.

    Targets equal to ``ignore_id`` (padding) are left out of both the NLL sum and the token count.
    ``mean_loops`` is weighted by the number of positions each batch's loop mean was taken over
    (``B * S``), so it is the mean over all positions evaluated.
    """
    ignore_index = -100 if ignore_id is None else ignore_id
    use_hidden = _accepts(model.forward, "return_hidden")
    use_mask = ignore_id is not None and _accepts(model.forward, "attention_mask")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    tokens = 0
    count = 0
    loop_sum = 0.0
    loop_weight = 0
    source = batches if max_batches is None else itertools.islice(batches, max_batches)
    try:
        with torch.no_grad():
            for ids in source:
                ids = ids.to(device)
                kwargs: Dict[str, Any] = {"num_loops": num_loops}
                if use_hidden:
                    kwargs["return_hidden"] = True
                if use_mask:
                    kwargs["attention_mask"] = (ids != ignore_id).long()
                out = model(ids, **kwargs)
                nll_sum, n = _nll_sum(model, out, ids, ignore_index)
                total_nll += float(nll_sum)
                tokens += n
                count += 1
                if out.mean_loops is not None:
                    positions = ids.shape[0] * ids.shape[1]
                    loop_sum += float(out.mean_loops) * positions
                    loop_weight += positions
    finally:
        model.train(was_training)
    if tokens == 0:
        raise ValueError("no tokens to evaluate: batches need at least one non-padding target token")
    return {
        "nll": total_nll / tokens,
        "tokens": tokens,
        "batches": count,
        "mean_loops": (loop_sum / loop_weight) if loop_weight else None,
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
    padding id if batches are padded. Pad tokens still enter the context (see the module docstring).
    The default ``None`` scores every target. ``num_loops=None`` uses the model's maximum loop count.
    The model is run in eval mode and its previous mode is restored afterwards.
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
    """``perplexity`` plus ``mean_loops``, the mean expected PonderNet loop count over all evaluated positions.

    Both numbers come from the same forward pass per batch. ``mean_loops`` is weighted by batch size
    times sequence length, so larger batches count in proportion. ``ignore_id`` works as in ``perplexity``.
    """
    r = _run(model, batches, None, num_loops, ignore_id)
    return {
        "nll": r["nll"],
        "perplexity": math.exp(r["nll"]),
        "tokens": r["tokens"],
        "batches": r["batches"],
        "mean_loops": r["mean_loops"],
    }


def perplexity_by_loops(
    model: InkyModel,
    batches: Iterable[torch.Tensor],
    loop_counts: Sequence[int],
    max_batches: Optional[int] = None,
    ignore_id: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """Perplexity at each recurrent loop count in ``loop_counts``, over the same batches.

    Returns ``{loop_count: perplexity_result}``, where each result is the dict from ``perplexity``.
    ``batches`` is materialised once so it can be scanned for every loop count.
    """
    counts = [int(c) for c in loop_counts]
    if not counts:
        raise ValueError("loop_counts must not be empty")
    if any(c < 1 for c in counts):
        raise ValueError(f"loop counts must be at least 1, got {counts}")
    materialised = list(batches if max_batches is None else itertools.islice(batches, max_batches))
    return {c: perplexity(model, materialised, num_loops=c, ignore_id=ignore_id) for c in counts}


def early_exit_drift(
    model: InkyModel,
    batches: Iterable[torch.Tensor],
    halt_threshold: Optional[float] = None,
    num_loops: Optional[int] = None,
    max_batches: Optional[int] = None,
    ignore_id: Optional[int] = None,
) -> Dict[str, float]:
    """NLL of ``model.forward_early_exit`` compared with the full forward pass on the same tokens.

    ``forward_early_exit`` runs at batch size 1, so each row of each batch is evaluated on its own
    with both methods. ``drift = early_nll - full_nll`` (nats per token); ``perplexity_ratio`` is
    ``exp(drift)``. ``loop_fraction`` is the token-loop pairs computed by early exit divided by those
    of the full pass (1.0 when nothing exits early). With ``halt_threshold=None`` the drift should be
    numerically zero. Raises ValueError if the model has no ``forward_early_exit``.
    """
    if not hasattr(model, "forward_early_exit"):
        raise ValueError("model has no forward_early_exit; early-exit evaluation is unavailable")
    ignore_index = -100 if ignore_id is None else ignore_id
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    full_sum = 0.0
    early_sum = 0.0
    tokens = 0
    sequences = 0
    applied = 0
    full_applied = 0
    loop_sum = 0.0
    loop_positions = 0
    source = batches if max_batches is None else itertools.islice(batches, max_batches)
    try:
        with torch.no_grad():
            for ids in source:
                ids = ids.to(device)
                for row in ids.split(1, dim=0):
                    full = model(row, num_loops=num_loops, return_hidden=True)
                    full_nll, n = _nll_sum(model, full, row, ignore_index)
                    early = model.forward_early_exit(
                        row, halt_threshold=halt_threshold, num_loops=num_loops, return_hidden=True
                    )
                    early_nll, _ = _nll_sum(model, early, row, ignore_index)
                    full_sum += float(full_nll)
                    early_sum += float(early_nll)
                    tokens += n
                    sequences += 1
                    applied += int(early.loop_applications)
                    full_applied += int(early.full_loop_applications)
                    loop_sum += float(early.mean_loops) * row.shape[1]
                    loop_positions += row.shape[1]
    finally:
        model.train(was_training)
    if tokens == 0:
        raise ValueError("no tokens to evaluate: batches need at least one non-padding target token")
    full_nll = full_sum / tokens
    early_nll = early_sum / tokens
    drift = early_nll - full_nll
    return {
        "full_nll": full_nll,
        "early_nll": early_nll,
        "drift": drift,
        "perplexity_ratio": math.exp(drift),
        "loop_fraction": (applied / full_applied) if full_applied else None,
        "mean_loops_early": (loop_sum / loop_positions) if loop_positions else None,
        "tokens": tokens,
        "sequences": sequences,
    }
