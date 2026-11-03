"""Training loop for Inky: warmup-stable-decay learning rate, loop curriculum, periodic eval and checkpoints.

  - wsd_lr_scale:             linear warmup, flat plateau, linear decay to zero (a multiplier on the base LRs).
  - loop_curriculum:          recurrent loop count ramps linearly from min to max over the first part of training.
  - save/load_training_state: model weights, optimiser state, step, history, RNG state and train config.
                              Written atomically, so an interrupted save never corrupts a checkpoint.
  - packed_stream:            a window source for train() that repacks the corpus each epoch with random offsets.
  - train:                    the optimisation loop. Steps are 1-based in the history and checkpoint names.

Loss logging: ``loss`` is the full training objective. Its ``ce`` term is the per-token next-token NLL,
the same quantity ``evaluate`` reports as ``nll``, so only ``ce`` is comparable to eval numbers. The other
terms (``step_ce``, ``z_loss``, ``moe_aux``, ``ponder_kl``) are logged unweighted; ``loss`` multiplies
each by its config coefficient.

Document boundaries: when batches carry document ids (``(ids, doc_ids)`` pairs, or a window source that
returns them), tokens attend only within their own document (``segment_ids``), and the loss target at each
document-boundary transition is set to the ignore label -100. The model is never asked to predict the
first token of a document from the end of the previous one.
"""

import contextlib
import json
import math
import os
import random
import re
import shutil
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from .data import doc_boundary_mask, pack_sequences, pack_sequences_with_segments
from .evaluation import evaluate, perplexity_by_loops
from .optim import format_group_counts

# Ignore label for the loss (the model's convention: labels equal to -100 are not scored).
IGNORE_INDEX = -100

# Unweighted loss terms reported for every training step (see the module docstring).
LOSS_PART_NAMES = ("ce", "step_ce", "z_loss", "moe_aux", "ponder_kl")

# Mixed-precision settings accepted by ``train(amp_dtype=...)``. None means fp32.
AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _check_amp_dtype(amp_dtype: Optional[str]) -> None:
    if amp_dtype is not None and amp_dtype not in AMP_DTYPES:
        raise ValueError(f"amp_dtype must be None, 'bf16' or 'fp16', got {amp_dtype!r}")


def _autocast(device: torch.device, amp_dtype: Optional[str]):
    """Autocast context for the forward pass (a no-op for fp32). bf16 works on CPU and CUDA; fp16 is CUDA only."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=AMP_DTYPES[amp_dtype])


def wsd_lr_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_fraction: float = 0.2,
) -> float:
    """Warmup-stable-decay multiplier in [0, 1] for a 0-based ``step`` in ``[0, total_steps - 1]``.

    Linear warmup from 0 to 1 over ``warmup_steps`` (step 0 gives 0), flat at 1, then a linear decay
    over the last ``decay_fraction * total_steps`` steps. The decay reaches 0 just after the final step,
    so the final update still has a nonzero rate.
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
        remaining = max(0, total_steps - step)
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
RNG_FILE = "rng.pt"
TRAIN_CONFIG_FILE = "train_config.json"
BEST_FILE = "best.pt"
_STEP_DIR_RE = re.compile(r"step_(\d+)")
_SEED_MODULUS = 2**63 - 1


# ----------------------------------------------------------------------------- checkpoint I/O


def _unwrap(model):
    """The InkyModel under an optional wrapper (a wrapper exposes it as ``inner``)."""
    return getattr(model, "inner", model)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _replace_dir(tmp: Path, final: Path) -> None:
    """Move the finished directory ``tmp`` into place as ``final``. The old one is kept until the swap succeeds."""
    if not final.exists():
        os.replace(tmp, final)
        return
    backup = final.with_name(f".{final.name}.old")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(final, backup)
    try:
        os.replace(tmp, final)
    except BaseException:
        os.replace(backup, final)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def save_training_state(
    path: Union[str, Path],
    model,
    optimizer,
    step: int,
    history: List[Dict[str, Any]],
    rng_state: Optional[Dict[str, Any]] = None,
    train_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Write everything needed to resume into the directory ``path``.

    Contents: ``model/`` (weights and model config), ``optimizer.pt``, ``train_state.json`` (step and
    history), and, when given, ``rng.pt`` (RNG states) and ``train_config.json`` (settings as JSON).

    The directory is built under a temporary name and moved into place last, so a failed save leaves
    any previous checkpoint at ``path`` unchanged.
    """
    root = Path(path)
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = root.with_name(f".{root.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        _unwrap(model).save_pretrained(tmp / MODEL_DIR)
        _atomic_torch_save(optimizer.state_dict(), tmp / OPTIMIZER_FILE)
        if rng_state is not None:
            _atomic_torch_save(rng_state, tmp / RNG_FILE)
        if train_config is not None:
            _atomic_write_text(
                tmp / TRAIN_CONFIG_FILE, json.dumps(train_config, indent=2, sort_keys=True, default=str)
            )
        _atomic_write_text(tmp / TRAIN_STATE_FILE, json.dumps({"step": int(step), "history": history}))
        _replace_dir(tmp, root)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_train_state(path: Union[str, Path]) -> Dict[str, Any]:
    """The ``train_state.json`` of a checkpoint: ``{"step": int, "history": list}``."""
    return json.loads((Path(path) / TRAIN_STATE_FILE).read_text())


def load_training_state(path: Union[str, Path], model, optimizer) -> int:
    """Restore weights and optimiser state saved by ``save_training_state``. Returns the saved step."""
    root = Path(path)
    state = read_train_state(root)
    target = _unwrap(model)
    fresh = type(target).from_pretrained(root / MODEL_DIR)
    target.load_state_dict(fresh.state_dict())
    optimizer.load_state_dict(torch.load(root / OPTIMIZER_FILE, map_location="cpu", weights_only=False))
    return int(state["step"])


def load_rng_state(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """RNG states saved with a checkpoint, or None for a checkpoint written without them."""
    file = Path(path) / RNG_FILE
    if not file.exists():
        return None
    return torch.load(file, map_location="cpu", weights_only=False)


def _capture_rng(gen: torch.Generator) -> Dict[str, Any]:
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "generator": gen.get_state()}


def _restore_rng(state: Dict[str, Any], gen: torch.Generator) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    gen.set_state(state["generator"])


def _prune_checkpoints(root: Path, keep_last: int) -> None:
    """Delete all but the ``keep_last`` newest ``step_N`` directories (0 keeps all). Also clears stale temp dirs."""
    for stale in root.glob(".step_*.tmp"):
        shutil.rmtree(stale, ignore_errors=True)
    if keep_last <= 0:
        return
    found = []
    for entry in root.iterdir():
        match = _STEP_DIR_RE.fullmatch(entry.name)
        if match and entry.is_dir():
            found.append((int(match.group(1)), entry))
    found.sort()
    for _, entry in found[:-keep_last]:
        shutil.rmtree(entry, ignore_errors=True)


def _save_best(root: Path, model, step: int, eval_nll: float, eval_ppl: float) -> None:
    """Weights of the best eval checkpoint so far, as one file ``best.pt`` (written atomically)."""
    target = _unwrap(model)
    _atomic_torch_save(
        {
            "model": target.state_dict(),
            "config": target.config.to_dict(),
            "step": int(step),
            "eval_nll": float(eval_nll),
            "eval_perplexity": float(eval_ppl),
        },
        root / BEST_FILE,
    )


# ----------------------------------------------------------------------------- data sources


def packed_stream(
    make_docs: Callable[[], Any],
    seq_len: int,
    eos_id: int,
    *,
    with_segments: bool = False,
) -> Callable[[random.Random], Any]:
    """A window source for ``train``: packs the documents afresh each epoch with random window offsets.

    ``make_docs`` is called once per epoch and must return a fresh iterable of token-id lists (for
    example a function that reopens the JSONL files). ``train`` passes an epoch-seeded ``random.Random``
    into the source, which becomes the ``rng`` of ``pack_sequences``, so the window boundaries move every
    epoch. With ``with_segments=True`` the source returns ``(tokens, doc_ids)``.
    """

    def source(rng: random.Random):
        docs = make_docs()
        if with_segments:
            tokens, doc_ids, _positions = pack_sequences_with_segments(docs, seq_len, eos_id, rng=rng)
            return tokens, doc_ids
        return pack_sequences(docs, seq_len, eos_id, rng=rng)

    return source


def _as_pair(value) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Normalise one batch or window array to ``(ids, doc_ids_or_None)``. Both must be 2-D [B, S]."""
    if isinstance(value, torch.Tensor):
        ids, docs = value, None
    elif isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, torch.Tensor) for v in value):
        ids, docs = value
    else:
        raise ValueError(
            "a training batch must be a [B, S] tensor or an (ids, doc_ids) pair of such tensors, "
            f"got {type(value).__name__}"
        )
    if ids.dim() != 2:
        raise ValueError(f"training batches must be 2-D [B, S], got shape {tuple(ids.shape)}")
    if docs is not None and tuple(docs.shape) != tuple(ids.shape):
        raise ValueError(f"doc_ids shape {tuple(docs.shape)} differs from ids shape {tuple(ids.shape)}")
    return ids, docs


def _epoch_seed(seed: int, epoch: int) -> int:
    return (int(seed) * 1_000_003 + epoch) % _SEED_MODULUS


def _epoch_batches(
    source, kind: str, batch_size: Optional[int], seed: int
) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Endless stream of ``(ids, doc_ids_or_None)`` batches, reshuffled every epoch.

    Each epoch's order comes from a generator seeded by ``(seed, epoch)``, so the stream is reproducible
    and can be fast-forwarded to any step on resume. For a windows source the windows are rebuilt each
    epoch (a callable gets an epoch-seeded ``random.Random``), then shuffled and cut into batches.
    """
    epoch = 0
    while True:
        epoch_seed = _epoch_seed(seed, epoch)
        gen = torch.Generator().manual_seed(epoch_seed)
        if kind == "batches":
            for i in torch.randperm(len(source), generator=gen).tolist():
                yield _as_pair(source[i])
        else:
            windows = source(random.Random(epoch_seed)) if callable(source) else source
            ids, docs = _as_pair(windows)
            n = ids.shape[0]
            if n == 0:
                raise ValueError("training windows are empty: the corpus is too short for seq_len")
            order = torch.randperm(n, generator=gen)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                yield ids[idx], (docs[idx] if docs is not None else None)
        epoch += 1


def _normalise_source(train_batches, batch_size: Optional[int]) -> Tuple[Any, str]:
    """Return ``(source, kind)``: kind "batches" for a list of batches, "windows" for a tensor or callable."""
    if isinstance(train_batches, torch.Tensor) or callable(train_batches):
        if isinstance(train_batches, torch.Tensor) and train_batches.dim() != 2:
            raise ValueError(f"a window tensor must be [N, S], got shape {tuple(train_batches.shape)}")
        if batch_size is None or batch_size < 1:
            raise ValueError("batch_size (>= 1) is required when train_batches is a window tensor or a callable")
        return train_batches, "windows"
    source = train_batches if isinstance(train_batches, SequenceABC) else list(train_batches)
    if len(source) == 0:
        raise ValueError("train_batches is empty")
    return source, "batches"


# ----------------------------------------------------------------------------- training loop


def _global_grad_norm(params) -> float:
    grads = [p.grad.detach().float().norm(2) for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(grads), 2))


def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.4f}"


def _train_line(step: int, steps: int, rec: Dict[str, Any]) -> str:
    return (
        f"step {step}/{steps} loss {rec['loss']:.4f} "
        f"[ce {_fmt(rec['ce'])} (eval-comparable NLL) step_ce {_fmt(rec['step_ce'])} "
        f"z_loss {_fmt(rec['z_loss'])} moe_aux {_fmt(rec['moe_aux'])} ponder_kl {_fmt(rec['ponder_kl'])}] "
        f"lr_scale {rec['lr_scale']:.3f} num_loops {rec['num_loops']} grad_norm {_fmt(rec['grad_norm'])}"
    )


def train(
    model,
    optimizer,
    train_batches,
    steps: int,
    eval_batches: Optional[Iterable[torch.Tensor]] = None,
    eval_every: int = 0,
    warmup_steps: Optional[int] = None,
    decay_fraction: float = 0.2,
    max_grad_norm: Optional[float] = 1.0,
    use_loop_curriculum: bool = False,
    log_every: int = 10,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_every: int = 0,
    log_fn: Callable[[str], Any] = print,
    max_skipped_steps: int = 50,
    resume_from: Optional[Union[str, Path]] = None,
    seed: int = 0,
    batch_size: Optional[int] = None,
    keep_last: int = 3,
    grad_accum_steps: int = 1,
    amp_dtype: Optional[str] = None,
    eval_by_depth: bool = False,
) -> List[Dict[str, Any]]:
    """Run optimiser steps up to the total ``steps`` and return the history of the whole run.

    ``steps`` is the total run length: schedules (LR, curriculum) are always computed from it. When
    ``warmup_steps`` is None it defaults to 5% of ``steps`` (at least 1), so a fresh run never starts at
    full learning rate.

    ``grad_accum_steps`` (default 1) is the number of micro-batches per optimiser step. Each step draws that
    many batches from ``train_batches``, backpropagates each loss divided by ``grad_accum_steps``, and then
    steps once. With equal-size micro-batches and no document masking this equals one step on the
    concatenated batch. The effective batch is ``grad_accum_steps`` times the batch size. ``resume_from``
    fast-forwards by ``grad_accum_steps`` batches per completed step, so a resumed run sees the same data.

    ``amp_dtype`` (default None, fp32) runs the forward pass under ``torch.autocast``: ``"bf16"`` on CPU or
    CUDA, or ``"fp16"`` on CUDA only, which also uses a ``GradScaler``. The optimiser, the clip and the
    gradient check stay in fp32. Eval runs in fp32.

    ``eval_by_depth`` (default False) makes each eval also report perplexity at every loop count from
    ``min_recurrent_loops`` to ``max_recurrent_loops``. These are the depths training samples from. The
    max-depth value equals the ordinary ``eval_nll``, which is what the checkpoint selection uses.

    ``train_batches`` takes one of three forms:
      * a list (or other sequence) of batches, each a [B, S] tensor or an ``(ids, doc_ids)`` pair. Each
        epoch visits the batches in a new random order.
      * a window tensor [N, S] (for example from ``prepare_corpus``) with ``batch_size``. Each epoch shuffles
        the rows and cuts them into batches of ``batch_size``.
      * a callable ``source(rng: random.Random)`` returning a window tensor or a ``(tokens, doc_ids)`` pair,
        with ``batch_size``. It is rebuilt every epoch (see ``packed_stream``), and random window offsets
        come from ``rng``.
    Any other iterable is materialised once. ``seed`` drives the data order, the loop-depth draws and the
    step-CE subset. On a fresh start it also seeds torch's and Python's global RNGs. On resume they are
    restored from the checkpoint instead.

    With ``resume_from`` set to a directory written by ``save_training_state``, the weights, optimiser
    state, RNG states and history are restored, and the run continues from the step after the saved one.
    The data order is fast-forwarded, so the resumed run sees the same batches as an uninterrupted run.
    The returned history is the saved history followed by the new steps.

    Each training record is ``{"kind": "train", "step", "loss", "ce", "step_ce", "z_loss", "moe_aux",
    "ponder_kl", "lr_scale", "num_loops", "grad_norm", "skipped", "skip_reason"}``. ``loss`` is the full
    objective. ``ce`` is the per-token NLL, the only term comparable to eval. The other loss terms are
    unweighted. ``num_loops`` is None unless the loop curriculum is on. ``skip_reason`` is None,
    ``"nonfinite_loss"`` or ``"nonfinite_grad"``.

    A step is skipped when the loss is not finite, or when the global gradient norm is not finite. A
    skipped step does not run the optimiser, zeroes the gradients, is recorded with ``skipped=True``, and is
    not checkpointed. ``max_skipped_steps`` counts skipped steps in a row; one more than that raises
    RuntimeError.

    Eval records are ``{"kind": "eval", "step", "eval_nll", "eval_perplexity", "is_best"}``. The eval NLL
    is per token, like ``ce``. Eval runs every ``eval_every`` steps and at the last step. When
    ``checkpoint_dir`` is given:
      * ``step_N/`` is written every ``checkpoint_every`` steps (not on skipped steps) and always at the
        last step, as full training state. ``train_config.json`` next to it records the settings used.
      * only the newest ``keep_last`` ``step_N`` directories are kept (0 keeps all).
      * ``best.pt`` holds the weights of the best eval so far (lowest ``eval_nll``), rewritten on each
        improvement.
    The model's previous train/eval mode is restored afterwards.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if max_skipped_steps < 0:
        raise ValueError(f"max_skipped_steps must be >= 0, got {max_skipped_steps}")
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0 (0 keeps every checkpoint), got {keep_last}")
    if int(grad_accum_steps) != grad_accum_steps or grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be an integer >= 1, got {grad_accum_steps}")
    grad_accum_steps = int(grad_accum_steps)
    _check_amp_dtype(amp_dtype)
    if warmup_steps is None:
        warmup_steps = max(1, steps // 20)

    source, kind = _normalise_source(train_batches, batch_size)

    start_step = 1
    history: List[Dict[str, Any]] = []
    prior_rng = None
    if resume_from is not None:
        start_step = load_training_state(resume_from, model, optimizer) + 1
        if start_step > steps:
            raise ValueError(f"resume_from is at step {start_step - 1}, which is not before steps={steps}")
        history = list(read_train_state(resume_from).get("history", []))
        prior_rng = load_rng_state(resume_from)

    gen = torch.Generator().manual_seed(_epoch_seed(seed, -1))
    if prior_rng is not None:
        _restore_rng(prior_rng, gen)
        log_fn(f"resumed at step {start_step}: RNG state restored from {Path(resume_from)}")
    else:
        torch.manual_seed(seed)
        random.seed(seed)
        if resume_from is not None:
            log_fn("checkpoint has no RNG state; RNG reseeded from the train seed")

    batches = _epoch_batches(source, kind, batch_size, seed)
    for _ in range((start_step - 1) * grad_accum_steps):  # fast-forward past the micro-batches already consumed
        next(batches)

    eval_list = None
    if eval_batches is not None and eval_every > 0:
        eval_list = list(eval_batches)

    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    target = _unwrap(model)
    cfg = target.config
    device = next(model.parameters()).device
    if amp_dtype == "fp16" and device.type != "cuda":
        raise ValueError("amp_dtype='fp16' needs a CUDA model; use 'bf16' for mixed precision on CPU")
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == "fp16" else None
    depth_counts = list(range(cfg.min_recurrent_loops, cfg.max_recurrent_loops + 1))
    train_config = {
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_fraction": decay_fraction,
        "max_grad_norm": max_grad_norm,
        "use_loop_curriculum": use_loop_curriculum,
        "seed": seed,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "amp_dtype": amp_dtype,
        "eval_by_depth": eval_by_depth,
        "model_config": cfg.to_dict(),
    }

    if hasattr(optimizer, "group_counts"):
        log_fn("optimizer groups: " + format_group_counts(optimizer.group_counts()))
    log_fn(
        f"loss = ce + {cfg.z_loss_coef:g}*z_loss + moe_aux + {cfg.ponder_step_loss_coef:g}*step_ce "
        f"+ {cfg.ponder_kl_coef:g}*ponder_kl; ce is the eval-comparable term"
    )
    log_fn(f"training steps {start_step}..{steps}, warmup {warmup_steps}, seed {seed}")

    best_nll = min(
        (r["eval_nll"] for r in history if r.get("kind") == "eval" and math.isfinite(r["eval_nll"])),
        default=math.inf,
    )
    consecutive_skips = 0
    total_skipped = 0

    was_training = model.training
    model.train()
    try:
        for step in range(start_step, steps + 1):
            idx = step - 1
            is_last = step == steps
            lr_scale = wsd_lr_scale(idx, steps, warmup_steps, decay_fraction)
            optimizer.set_lr_scale(lr_scale)

            num_loops = None
            if use_loop_curriculum:
                num_loops = loop_curriculum(idx, steps, cfg.min_recurrent_loops, cfg.max_recurrent_loops)

            # Draw every micro-batch of this step first, so the data order does not depend on a skip.
            micro = [next(batches) for _ in range(grad_accum_steps)]
            losses: List[float] = []
            part_sums: Dict[str, float] = {}
            skip_reason = None
            for ids, docs in micro:
                ids = ids.to(device)
                labels = ids.clone()
                segment_ids = None
                if docs is not None:
                    segment_ids = docs.to(device)
                    boundary = doc_boundary_mask(segment_ids)
                    labels[:, 1:] = labels[:, 1:].masked_fill(boundary, IGNORE_INDEX)

                with _autocast(device, amp_dtype):
                    out = model(
                        input_ids=ids,
                        labels=labels,
                        num_loops=num_loops,
                        segment_ids=segment_ids,
                        generator=gen,
                    )
                loss_value = float(out.loss.detach())
                losses.append(loss_value)
                for k, v in (getattr(out, "loss_parts", None) or {}).items():
                    part_sums[k] = part_sums.get(k, 0.0) + float(v)
                if not math.isfinite(loss_value):
                    skip_reason = "nonfinite_loss"
                    break
                scaled = out.loss / grad_accum_steps
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            n_done = len(losses)
            loss_value = sum(losses) / n_done
            parts = {k: v / n_done for k, v in part_sums.items()}
            record: Dict[str, Any] = {
                "kind": "train",
                "step": step,
                "loss": loss_value,
                **{name: parts.get(name) for name in LOSS_PART_NAMES},
                "lr_scale": lr_scale,
                "num_loops": num_loops,
                "grad_norm": None,
                "skipped": False,
                "skip_reason": None,
            }
            grad_norm = None
            unscaled = False
            if skip_reason is None and scaler is not None:
                scaler.unscale_(optimizer.muon)
                scaler.unscale_(optimizer.adamw)
                unscaled = True
            if skip_reason is None:
                grad_norm = _global_grad_norm(optimizer.params)
                if not math.isfinite(grad_norm):
                    skip_reason = "nonfinite_grad"
            record["skip_reason"] = skip_reason

            if record["skip_reason"] is not None:
                # The optimiser does not run, so its moments and the weights stay as they were.
                optimizer.zero_grad()
                if unscaled:
                    scaler.update()  # lowers the loss scale after an fp16 overflow
                record.update(skipped=True, grad_norm=grad_norm)
                consecutive_skips += 1
                total_skipped += 1
                history.append(record)
                log_fn(
                    f"step {step}/{steps} skipped ({record['skip_reason']}), loss {loss_value}, "
                    f"grad_norm {grad_norm}; {consecutive_skips} in a row, {total_skipped} in total"
                )
                if consecutive_skips > max_skipped_steps:
                    raise RuntimeError(
                        f"{consecutive_skips} consecutive steps skipped for non-finite loss or gradients, "
                        f"more than max_skipped_steps={max_skipped_steps}; stopping at step {step}"
                    )
            else:
                if scaler is not None:
                    # The grads are already unscaled and finite. Clip here, as HybridOptimizer.step would.
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(optimizer.params, max_grad_norm)
                    scaler.step(optimizer.muon)
                    scaler.step(optimizer.adamw)
                    scaler.update()
                else:
                    optimizer.step(max_grad_norm=max_grad_norm)
                optimizer.zero_grad()
                model.update_moe_balance()
                consecutive_skips = 0
                record["grad_norm"] = grad_norm
                history.append(record)
                if log_every > 0 and (step % log_every == 0 or is_last):
                    log_fn(_train_line(step, steps, record))

            if eval_list is not None and (step % eval_every == 0 or is_last):
                ev = evaluate(model, eval_list, num_loops=None)
                is_best = math.isfinite(ev["nll"]) and ev["nll"] < best_nll
                if is_best:
                    best_nll = ev["nll"]
                entry: Dict[str, Any] = {
                    "kind": "eval",
                    "step": step,
                    "eval_nll": ev["nll"],
                    "eval_perplexity": ev["perplexity"],
                    "is_best": is_best,
                }
                history.append(entry)
                log_fn(
                    f"step {step} eval nll {ev['nll']:.4f} (per token, comparable to ce) "
                    f"perplexity {ev['perplexity']:.2f}" + (" new best" if is_best else "")
                )
                if eval_by_depth:
                    by_depth = perplexity_by_loops(model, eval_list, depth_counts)
                    entry["per_depth"] = {
                        str(c): {"nll": r["nll"], "perplexity": r["perplexity"]} for c, r in by_depth.items()
                    }
                    for c in depth_counts:
                        log_fn(
                            f"step {step} eval depth {c} loops: nll {by_depth[c]['nll']:.4f} "
                            f"perplexity {by_depth[c]['perplexity']:.2f}"
                        )
                model.train()
                if is_best and ckpt_root is not None:
                    _save_best(ckpt_root, model, step, ev["nll"], ev["perplexity"])

            periodic = checkpoint_every > 0 and step % checkpoint_every == 0 and not record["skipped"]
            if ckpt_root is not None and (periodic or is_last):
                path = ckpt_root / f"step_{step}"
                save_training_state(
                    path,
                    model,
                    optimizer,
                    step,
                    history,
                    rng_state=_capture_rng(gen),
                    train_config=train_config,
                )
                _prune_checkpoints(ckpt_root, keep_last)
                log_fn(f"saved checkpoint {path}")
    finally:
        model.train(was_training)

    log_fn(f"done: {steps} steps, {total_skipped} skipped in total")
    return history
