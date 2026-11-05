"""Text generation for ``InkyModel``: sampling with the prose controls, scratch reduction, and the helpers."""

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch

from .cache import InkyCache, InkyOutput
from .config import InkyConfig
from .data import DEFAULT_RESERVED_IDS
from .metrics import print_report, text_report
from .scratch import DEFAULT_MAX_NOTE_TOKENS, ScratchTracker
from .style import NgramBanIndex, banned_by_bad_words

# Default for ``eos_token_id`` in generate / self_speculative_generate: use ``config.eos_token_id``.
# An explicit None means "never stop at EOS"; an int means "stop at this id".
_USE_CONFIG_EOS = object()


class InkyGenerationMixin:
    """``generate`` for ``InkyModel`` (mixed in; it uses ``self(...)``, ``self.config`` and the cache types)."""

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: Union[int, None, object] = _USE_CONFIG_EOS,
        num_loops: Optional[int] = None,
        no_repeat_ngram_size: int = 4,
        bad_word_ids: Optional[Sequence[Sequence[int]]] = None,
        reduce_scratch_blocks: bool = False,
        report: bool = False,
        display: bool = False,
        stock_word_penalty: float = 0.0,
        soft_penalty_ids: Optional[Sequence[int]] = None,
        repeat_window: Optional[int] = None,
        count_prose_only: bool = False,
        scratch_max_note_tokens: Optional[int] = DEFAULT_MAX_NOTE_TOKENS,
        blocked_token_ids: Optional[Sequence[int]] = None,
    ):
        """Sample a continuation.

        ``blocked_token_ids`` are never emitted (set to -inf). None (the default) blocks the reserved
        special ids of the tokenizer, ``DEFAULT_RESERVED_IDS`` (49974 to 50002), minus the scratch ids,
        which generation uses through ``reduce_scratch_blocks``, and minus the configured EOS id, which
        must stay allowed (DATA-05). Pass an explicit list to block other ids, or ``[]`` to block nothing.
        The EOS id (``eos_token_id``, when given) and the configured ``config.eos_token_id`` are never
        blocked, even if they are in the list. Ids outside the vocabulary are ignored.
        ``eos_token_id`` stops generation once every row emits it: left at its default it is
        ``config.eos_token_id`` (DOC-05), an int overrides it, and an explicit ``None`` never stops at EOS.
        ``bad_word_ids`` blocks token sequences (see ``style.build_bad_word_ids``).
        ``no_repeat_ngram_size`` blocks any repeated n-gram (0 turns it off; 1 is rejected, because it
        would ban every token already seen).
        ``repeat_window`` (default None = the whole history) limits the repetition penalty and the
        no-repeat rule to the last ``repeat_window`` tokens, prompt included.
        ``stock_word_penalty`` (default 0) subtracts this value from the logit of every id in
        ``soft_penalty_ids`` (see ``style.soft_penalty_token_ids``). It is a soft preference, not a ban.
        ``reduce_scratch_blocks`` applies PENCIL-style reduction: when the model closes a scratch
        block, the notes are removed from the sequence. It needs batch size 1. Tokens in the prompt
        are never removed. Notes longer than ``scratch_max_note_tokens`` (None = no cap) are not
        treated as notes. Structurally invalid markers (a separator with no open block, a close with
        no separator) are masked out. After a close, the cache is rolled back to the block's open
        marker and only the text after it is recomputed, so no full re-prefill happens.
        ``count_prose_only`` (needs ``reduce_scratch_blocks`` to differ) counts only prose tokens
        against ``max_new_tokens``, so notes do not use up the budget. A note still open at the stop
        point is kept in the output.

        Each batch row keeps an incremental no-repeat n-gram index, updated with every new token, so
        the decode loop does not rebuild it from the whole sequence at each step.
        """
        self.eval()
        cfg = self.config
        if eos_token_id is _USE_CONFIG_EOS:
            eos_token_id = cfg.eos_token_id      # DOC-05: stop at the configured <|endoftext|> by default
        if reduce_scratch_blocks and input_ids.shape[0] != 1:
            raise ValueError("scratch reduction needs batch size 1")
        if no_repeat_ngram_size == 1:
            raise ValueError(
                "no_repeat_ngram_size=1 would ban every token already seen; use 0 to turn the rule off "
                "or a value >= 2"
            )
        if no_repeat_ngram_size < 0:
            raise ValueError(f"no_repeat_ngram_size must be >= 0, got {no_repeat_ngram_size}")
        if repeat_window is not None and repeat_window < 1:
            raise ValueError(f"repeat_window must be None or >= 1, got {repeat_window}")
        if scratch_max_note_tokens is not None and scratch_max_note_tokens < 0:
            raise ValueError(f"scratch_max_note_tokens must be None or >= 0, got {scratch_max_note_tokens}")
        if stock_word_penalty < 0:
            raise ValueError(f"stock_word_penalty must be >= 0, got {stock_word_penalty}")
        penalty_mask = None
        if stock_word_penalty > 0:
            ids_list = sorted({int(t) for t in (soft_penalty_ids or ())})
            if not ids_list:
                raise ValueError("stock_word_penalty > 0 needs soft_penalty_ids")
            if ids_list[0] < 0 or ids_list[-1] >= cfg.vocab_size:
                raise ValueError("soft_penalty_ids must be token ids inside the vocabulary")
            penalty_mask = torch.zeros(cfg.vocab_size, dtype=torch.float32, device=input_ids.device)
            penalty_mask[ids_list] = 1.0

        blocked = _blocked_ids(cfg, blocked_token_ids, eos_token_id)
        prompt_len = input_ids.shape[1]
        cache = InkyCache(cfg)
        seq = input_ids
        # Per-row token history (Python ints). Appended to once per new token; never re-listed.
        history: List[List[int]] = [row.tolist() for row in input_ids]
        ngram = (
            [_make_ngram_index(no_repeat_ngram_size, row, repeat_window) for row in history]
            if no_repeat_ngram_size else None
        )
        tail_len = max((len(s) - 1 for s in bad_word_ids), default=0) if bad_word_ids else 0
        scratch = (
            ScratchTracker(cfg.scratch_open_id, cfg.scratch_sep_id, cfg.scratch_close_id,
                           scratch_max_note_tokens, prefix=history[0])
            if reduce_scratch_blocks else None
        )
        # Before each open marker: the cache state and the logits that predicted the marker's position.
        # Keyed by the marker's index in seq. Used to roll back the cache when its block closes.
        snapshots: Dict[int, Tuple[dict, InkyOutput]] = {}
        out = self(seq, past=cache, num_loops=num_loops, last_only=True)
        steps = 0
        while True:
            if count_prose_only and scratch is not None:
                if scratch.prose_count(prompt_len) >= max_new_tokens:
                    break
            elif steps >= max_new_tokens:
                break
            steps += 1
            if cache.seq_len >= cfg.max_position_embeddings:
                break
            logits = _apply_repetition_penalty(out.logits[:, -1, :], seq, repetition_penalty, repeat_window)
            if penalty_mask is not None:
                logits = logits - stock_word_penalty * penalty_mask
            logits = _block_tokens(logits, ngram, history, tail_len, bad_word_ids)
            logits = _mask_ids(logits, blocked)
            if scratch is not None:
                logits = _mask_ids(logits, scratch.banned_markers())
            if do_sample:
                next_tok = _sample(logits, temperature, top_k, top_p)
            else:
                next_tok = logits.argmax(dim=-1)
            new_tokens = next_tok.tolist()
            if scratch is not None and new_tokens[0] == cfg.scratch_open_id:
                # cache.seq_len == seq.shape[1] here, so this is the state before the marker is fed.
                snapshots[seq.shape[1]] = (_cache_snapshot(cache), out)
            seq = torch.cat([seq, next_tok[:, None]], dim=1)
            for b, token in enumerate(new_tokens):
                history[b].append(token)
                if ngram is not None:
                    ngram[b].update(token)
            if eos_token_id is not None and bool((next_tok == eos_token_id).all()):
                break
            if scratch is not None:
                cut = scratch.feed(new_tokens[0])
                if cut is not None:
                    # Notes were removed from index `cut` onward. Everything before it is unchanged,
                    # so the cache is rolled back to the block's open marker and only the text after
                    # it is recomputed (the loop count is part of the cache and is kept).
                    snap, out_before = snapshots.pop(cut)
                    snapshots = {k: v for k, v in snapshots.items() if k in scratch.open_positions()}
                    _cache_restore(cache, snap)
                    seq = torch.tensor([scratch.out], dtype=seq.dtype, device=seq.device)
                    history[0] = list(scratch.out)
                    if ngram is not None:
                        ngram[0] = _make_ngram_index(no_repeat_ngram_size, scratch.out, repeat_window)
                    if seq.shape[1] > cut:
                        out = self(seq[:, cut:], past=cache, last_only=True)
                    else:
                        out = out_before
                    continue
                snapshots = {k: v for k, v in snapshots.items() if k in scratch.open_positions()}
            out = self(next_tok[:, None], past=cache, last_only=True)
        if not (report or display):
            return seq
        rep = text_report(seq[0, prompt_len:], bad_word_ids)
        if display:
            print_report(rep)
        return seq, rep


def _cache_snapshot(cache: InkyCache) -> dict:
    """Shallow copy of the cache state. The forward pass replaces entries and never edits them in place."""
    return {
        "pre": list(cache.pre),
        "post": list(cache.post),
        "core": list(cache.core),
        "prefix": list(cache.prefix),
        "seq_len": cache.seq_len,
    }


def _cache_restore(cache: InkyCache, snap: dict) -> None:
    cache.pre = list(snap["pre"])
    cache.post = list(snap["post"])
    cache.core = list(snap["core"])
    cache.prefix = list(snap["prefix"])
    cache.seq_len = snap["seq_len"]


def _blocked_ids(cfg: InkyConfig, explicit, eos_token_id: Optional[int]) -> List[int]:
    """Sorted token ids that generate must never emit (DATA-05).

    Default (``explicit`` None): the reserved special ids minus the scratch ids and minus the configured
    EOS id. An explicit list replaces the default. The EOS id (``eos_token_id`` and ``cfg.eos_token_id``)
    is always removed from the result, so the stop token stays allowed. Ids outside the vocabulary are
    dropped, since they cannot be indexed.
    """
    scratch = {cfg.scratch_open_id, cfg.scratch_sep_id, cfg.scratch_close_id}
    if explicit is None:
        ids = set(DEFAULT_RESERVED_IDS) - scratch - {cfg.eos_token_id}
    else:
        ids = {int(t) for t in explicit}
    ids.discard(cfg.eos_token_id)
    if eos_token_id is not None:
        ids.discard(int(eos_token_id))
    return sorted(t for t in ids if 0 <= t < cfg.vocab_size)


def _mask_ids(logits: torch.Tensor, ids) -> torch.Tensor:
    """Sets the given token ids to -inf (returns a copy; ``logits`` is not modified)."""
    if not ids:
        return logits
    logits = logits.clone()
    logits[:, sorted(ids)] = float("-inf")
    return logits


class _WindowedNgramIndex:
    """No-repeat index that counts only n-grams lying fully inside the last ``window`` tokens.

    Same answers as ``NgramBanIndex`` when the window covers the whole sequence. Grams that leave
    the window are removed as new tokens arrive, so the index stays O(window) in size.
    """

    def __init__(self, n: int, ids: Iterable[int] = (), window: Optional[int] = None):
        if window is not None and window < 1:
            raise ValueError(f"window must be None or >= 1, got {window}")
        self.n = n
        self.window = window
        self._seq: List[int] = []
        # (n-1)-token prefix -> {next token: number of windowed occurrences}
        self._next: Dict[Tuple[int, ...], Dict[int, int]] = {}
        for token in ids:
            self.update(token)

    def __len__(self) -> int:
        return len(self._seq)

    def _gram_at(self, start: int) -> Tuple[Tuple[int, ...], int]:
        n = self.n
        return tuple(self._seq[start:start + n - 1]), self._seq[start + n - 1]

    def _add(self, start: int) -> None:
        key, nxt = self._gram_at(start)
        counts = self._next.setdefault(key, {})
        counts[nxt] = counts.get(nxt, 0) + 1

    def _remove(self, start: int) -> None:
        key, nxt = self._gram_at(start)
        counts = self._next[key]
        counts[nxt] -= 1
        if counts[nxt] == 0:
            del counts[nxt]
        if not counts:
            del self._next[key]

    def update(self, token: int) -> None:
        self._seq.append(token)
        n, w, L = self.n, self.window, len(self._seq)
        if n < 2 or (w is not None and w < n):
            return
        if L >= n:
            self._add(L - n)                     # the gram that this token completes
        if w is not None and L - w - 1 >= 0:
            self._remove(L - w - 1)              # the gram that has just left the window

    def banned(self) -> Set[int]:
        n = self.n
        if n <= 0 or len(self._seq) < n:
            return set()
        if n == 1:
            return set(self._seq if self.window is None else self._seq[-self.window:])
        tail = tuple(self._seq[-(n - 1):])
        return set(self._next.get(tail, {}).keys())


def _make_ngram_index(n: int, ids: Iterable[int], window: Optional[int]):
    if window is None:
        return NgramBanIndex(n, ids)
    return _WindowedNgramIndex(n, ids, window)


def _block_tokens(
    logits: torch.Tensor,
    ngram: Optional[List[NgramBanIndex]],
    history: List[List[int]],
    tail_len: int,
    bad_word_ids: Optional[Sequence[Sequence[int]]],
) -> torch.Tensor:
    """Sets banned next tokens to -inf. ``ngram[b]`` is row b's incremental no-repeat index (None if off).

    Bad-word sequences only look at the last ``tail_len`` tokens of a row (the longest prefix they need),
    so the row history is sliced to that tail rather than scanned in full.
    """
    if ngram is None and not bad_word_ids:
        return logits
    logits = logits.clone()
    for b in range(logits.shape[0]):
        banned = ngram[b].banned() if ngram is not None else set()
        if bad_word_ids:
            tail = history[b][-tail_len:] if tail_len > 0 else []
            banned |= banned_by_bad_words(tail, bad_word_ids)
        if banned:
            logits[b, sorted(banned)] = float("-inf")
    return logits


def _apply_repetition_penalty(
    logits: torch.Tensor, seq: torch.Tensor, penalty: float, window: Optional[int] = None
) -> torch.Tensor:
    """Penalises every token seen in ``seq`` (or only the last ``window`` tokens when given)."""
    if penalty == 1.0:
        return logits
    ctx = seq if window is None else seq[:, -window:]
    picked = torch.gather(logits, 1, ctx)
    picked = torch.where(picked < 0, picked * penalty, picked / penalty)
    return logits.scatter(1, ctx, picked)


def _candidate_distribution(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kept candidates and their probabilities: ``(indices [B, K], probs [B, K])``.

    With ``top_k`` set, only the k largest logits are read (torch.topk), and top-p works inside
    those k values, so the full vocabulary is never sorted (GEN-05). Without ``top_k`` and with
    ``top_p < 1``, the full vocabulary is sorted as before. Ties at the k-th logit keep exactly k
    tokens here, where the old mask kept every tie.
    """
    logits = logits / max(temperature, 1e-5)
    V = logits.size(-1)
    if top_k and 0 < top_k < V:
        vals, idx = torch.topk(logits, top_k, dim=-1)          # sorted, largest first
    elif top_p < 1.0:
        vals, idx = torch.sort(logits, descending=True, dim=-1)
    else:
        vals = logits
        idx = torch.arange(V, device=logits.device).unsqueeze(0).expand(logits.shape[0], V)
    if top_p < 1.0:
        cum = torch.softmax(vals, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        vals = vals.masked_fill(remove, float("-inf"))
    return idx, torch.softmax(vals, dim=-1)


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    idx, probs = _candidate_distribution(logits, temperature, top_k, top_p)
    pick = torch.multinomial(probs, num_samples=1)
    return idx.gather(-1, pick).squeeze(-1)
