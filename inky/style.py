"""Decoding controls for prose quality: stock phrases, repeated n-grams, and bad-word blocking.

These controls are for writing quality. Common AI-written prose is often marked by stock
transitions and words ("furthermore", "delve", "tapestry") and by repeated phrasing. The lists
below are a starting point, not a validated detector. Pass your own list to override them.

Default policy:
  * ``STOCK_PHRASES`` (multi-word phrases) are hard-banned via ``build_bad_word_ids``.
  * ``STOCK_WORDS`` (single words) are NOT hard-banned, because banning a common word everywhere
    deletes it from all prose. They are exposed as a soft penalty via ``soft_penalty_token_ids``,
    which the caller subtracts from logits. Pass ``include_single_words=True`` to restore the
    old hard ban.
"""

from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

# UNVALIDATED HEURISTIC: multi-word stock phrases seen in AI-written prose. Not measured against
# any corpus or detector; tune on your own data before relying on it.
STOCK_PHRASES: List[str] = [
    "dive into",
    "embark on",
    "navigate the complexities",
    "it's important to note",
    "it is important to note",
    "in conclusion",
    "game-changer",
]

# UNVALIDATED HEURISTIC: single words seen in AI-written prose. These are soft-penalised by
# default, not hard-banned, because they are ordinary English words in other contexts.
STOCK_WORDS: List[str] = [
    "delve",
    "tapestry",
    "pivotal",
    "realm",
    "testament",
    "underscore",
    "multifaceted",
    "intricate",
    "furthermore",
    "moreover",
]


def _variants(phrase: str) -> List[str]:
    return sorted({phrase, phrase.capitalize(), " " + phrase, " " + phrase.capitalize()})


def build_bad_word_ids(
    encode: Callable[[str], Sequence[int]],
    phrases: Iterable[str] = STOCK_PHRASES,
    include_single_words: bool = False,
    words: Iterable[str] = STOCK_WORDS,
) -> List[List[int]]:
    """Token-id sequences to block. ``encode`` maps text to ids without special tokens.

    Each phrase is tried as-is and with a capital first letter, both bare and after a space,
    since the same word is tokenised differently in each form. By default only ``phrases`` are
    used (multi-token hard bans). Set ``include_single_words=True`` to also hard-ban ``words``,
    which restores the old global single-word ban.
    """
    seqs: List[List[int]] = []
    sources = list(phrases)
    if include_single_words:
        sources += list(words)
    for phrase in sources:
        for variant in _variants(phrase):
            ids = list(encode(variant))
            if ids and ids not in seqs:
                seqs.append(ids)
    return seqs


def soft_penalty_token_ids(
    encode: Callable[[str], Sequence[int]],
    words: Iterable[str] = STOCK_WORDS,
) -> List[int]:
    """Unique token ids for single-token forms of ``words``, for a soft logit penalty.

    Only variants that encode to exactly one token are returned, since a multi-token word
    cannot be penalised by a single logit. The caller decides the penalty size; the default
    decoding path does not apply one.
    """
    out: List[int] = []
    for word in words:
        for variant in _variants(word):
            ids = list(encode(variant))
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


def banned_by_bad_words(ids: Sequence[int], bad_word_ids: Sequence[Sequence[int]]) -> Set[int]:
    """Next tokens that would finish a blocked sequence, given the context ``ids``.

    A one-token entry is always banned. A longer entry bans its last token only when the context
    already ends with the entry's earlier tokens.
    """
    banned: Set[int] = set()
    for seq in bad_word_ids:
        if len(seq) == 1:
            banned.add(seq[0])
            continue
        prefix = list(seq[:-1])
        if len(ids) >= len(prefix) and list(ids[len(ids) - len(prefix):]) == prefix:
            banned.add(seq[-1])
    return banned


class NgramBanIndex:
    """Incremental no-repeat-n-gram index.

    Maps each (n-1)-token prefix seen so far to the set of tokens that followed it. ``update``
    is O(n) per token and ``banned`` is O(size of the answer), so a decode loop does not rescan
    the whole sequence each step. Gives the same result as ``banned_by_no_repeat_ngram`` on the
    same token sequence.
    """

    def __init__(self, n: int, ids: Iterable[int] = ()):
        self.n = n
        self._seq: List[int] = []
        self._next: Dict[Tuple[int, ...], Set[int]] = {}
        for token in ids:
            self.update(token)

    def __len__(self) -> int:
        return len(self._seq)

    def update(self, token: int) -> None:
        """Append one generated token and index the n-gram it completes."""
        self._seq.append(token)
        n = self.n
        if n >= 2 and len(self._seq) >= n:
            key = tuple(self._seq[-n:-1])
            self._next.setdefault(key, set()).add(self._seq[-1])

    def banned(self) -> Set[int]:
        """Next tokens that would repeat an n-gram already present in the sequence."""
        n = self.n
        if n <= 0 or len(self._seq) < n:
            return set()
        if n == 1:
            return set(self._seq)
        tail = tuple(self._seq[-(n - 1):])
        return set(self._next.get(tail, ()))


def banned_by_no_repeat_ngram(ids: Sequence[int], n: int) -> Set[int]:
    """Next tokens that would repeat an n-gram already present in ``ids``.

    Stateless wrapper kept for compatibility. In a decode loop, keep a ``NgramBanIndex`` and call
    ``update`` per generated token instead, which avoids rebuilding the index each step.
    """
    return NgramBanIndex(n, ids).banned()
