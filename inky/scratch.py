"""Scratch reduction for Inky, after PENCIL (arXiv:2503.14337).

The paper's rule is   C [CALL] T [SEP] A [RETURN]  =>  C A
Inky uses it for writing: the model drafts notes (T) between scratch_open and scratch_sep, writes
the text (A), and closes with scratch_close. Reduction removes the notes and the markers and keeps
the text, so the notes stop occupying the context window.
"""

from typing import List, Sequence


def reduce_scratch(
    tokens: Sequence[int],
    open_id: int,
    sep_id: int,
    close_id: int,
    start: int = 0,
) -> List[int]:
    """Apply the reduction rule to every closed scratch block at or after ``start``.

    Tokens before ``start`` (the prompt) are never edited. Blocks nest: an inner block inside the
    notes is removed with the notes, and an inner block inside the text is reduced on its own.
    Unclosed blocks and stray markers are left in place.
    """
    out: List[int] = list(tokens[:start])
    frames: List[dict] = []  # one per open block: {"open": index in out, "sep": index in out or None}
    for tok in tokens[start:]:
        if tok == open_id:
            out.append(tok)
            frames.append({"open": len(out) - 1, "sep": None})
        elif tok == sep_id and frames and frames[-1]["sep"] is None:
            out.append(tok)
            frames[-1]["sep"] = len(out) - 1
        elif tok == close_id and frames and frames[-1]["sep"] is not None:
            frame = frames.pop()
            # Drop the notes and their markers. Enclosing frames sit before this block, so their
            # indices are unchanged.
            del out[frame["open"]: frame["sep"] + 1]
        else:
            out.append(tok)
    return out
