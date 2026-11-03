"""Scratch reduction for Inky, after PENCIL (arXiv:2503.14337).

The paper's rule is   C [CALL] T [SEP] A [RETURN]  =>  C A
Inky uses it for writing: the model drafts notes (T) between scratch_open and scratch_sep, writes
the text (A), and closes with scratch_close. Reduction removes the notes and the markers and keeps
the text, so the notes stop occupying the context window.

Safety rules (GEN-01):
  * Notes are capped at ``max_note_tokens`` tokens. A block whose notes grow past the cap is
    unpaired: its open marker stays as ordinary text and nothing is removed for it. So a stray
    open marker can never make a later separator delete a long run of prose.
  * ``ScratchTracker.banned_markers`` reports the markers that would be refused at this point
    (a separator with no open block, a close with no separator). Generation masks these, so the
    model is never shown the choice of a structurally invalid marker.
"""

from typing import Iterable, List, Optional, Sequence, Set

DEFAULT_MAX_NOTE_TOKENS = 256


class ScratchTracker:
    """Incremental scratch-block state over the sequence that remains after reduction.

    Feed tokens one at a time with ``feed``. ``out`` is always the reduced sequence so far. ``feed``
    returns the index at which the sequence was cut when a closed block's notes were removed (the
    block's open marker index in ``out``), or None when nothing was removed.
    """

    def __init__(
        self,
        open_id: int,
        sep_id: int,
        close_id: int,
        max_note_tokens: Optional[int] = DEFAULT_MAX_NOTE_TOKENS,
        prefix: Iterable[int] = (),
    ):
        if max_note_tokens is not None and max_note_tokens < 0:
            raise ValueError(f"max_note_tokens must be >= 0 or None, got {max_note_tokens}")
        self.open_id = open_id
        self.sep_id = sep_id
        self.close_id = close_id
        self.max_note_tokens = max_note_tokens
        self.out: List[int] = list(prefix)
        # One per open block: {"open": index in out, "sep": index in out or None}, outermost first.
        self._frames: List[dict] = []

    def feed(self, tok: int) -> Optional[int]:
        out, frames = self.out, self._frames
        cut: Optional[int] = None
        if tok == self.open_id:
            out.append(tok)
            frames.append({"open": len(out) - 1, "sep": None})
        elif tok == self.sep_id and frames and frames[-1]["sep"] is None:
            out.append(tok)
            frames[-1]["sep"] = len(out) - 1
        elif tok == self.close_id and frames and frames[-1]["sep"] is not None:
            frame = frames.pop()
            cut = frame["open"]
            # Drop the notes and their markers. Enclosing frames sit before this block, so their
            # indices are unchanged.
            del out[frame["open"]: frame["sep"] + 1]
        else:
            out.append(tok)
        self._drop_overlong()
        return cut

    def _drop_overlong(self) -> None:
        """Unpair open blocks whose notes exceed the cap. Their markers stay as text."""
        cap = self.max_note_tokens
        if cap is None:
            return
        frames, n = self._frames, len(self.out)
        while frames and frames[-1]["sep"] is None and n - frames[-1]["open"] - 1 > cap:
            frames.pop()

    def banned_markers(self) -> Set[int]:
        """Marker ids that would be refused as the next token."""
        top = self._frames[-1] if self._frames else None
        banned: Set[int] = set()
        if top is None or top["sep"] is not None:
            banned.add(self.sep_id)
        if top is None or top["sep"] is None:
            banned.add(self.close_id)
        return banned

    def open_positions(self) -> Set[int]:
        """Indices in ``out`` of the open markers of blocks that are still open."""
        return {frame["open"] for frame in self._frames}

    def prose_count(self, prompt_len: int) -> int:
        """Tokens after the prompt that are not notes (scratch markers and note text).

        Notes are the span from an open marker to its separator, or to the end of the sequence while
        the block has no separator yet. Answer text after a separator counts as prose.
        """
        n = len(self.out)
        covered = 0
        cur = prompt_len - 1
        for frame in self._frames:
            start = frame["open"]
            end = frame["sep"] if frame["sep"] is not None else n - 1
            start = max(start, cur + 1)
            if end >= start:
                covered += end - start + 1
                cur = max(cur, end)
        return (n - prompt_len) - covered


def reduce_scratch(
    tokens: Sequence[int],
    open_id: int,
    sep_id: int,
    close_id: int,
    start: int = 0,
    max_note_tokens: Optional[int] = DEFAULT_MAX_NOTE_TOKENS,
) -> List[int]:
    """Apply the reduction rule to every closed scratch block at or after ``start``.

    Tokens before ``start`` (the prompt) are never edited. Blocks nest: an inner block inside the
    notes is removed with the notes, and an inner block inside the text is reduced on its own.
    Unclosed blocks and stray markers are left in place. A block whose notes exceed
    ``max_note_tokens`` (None = no cap) is not paired, so its text is kept.
    """
    tracker = ScratchTracker(open_id, sep_id, close_id, max_note_tokens, prefix=tokens[:start])
    for tok in tokens[start:]:
        tracker.feed(tok)
    return tracker.out
