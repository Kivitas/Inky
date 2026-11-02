"""Decoding controls for prose quality: stock phrases, repeated n-grams, and bad-word blocking.

These controls are for writing quality. Common AI-written prose is often marked by stock
transitions and words ("furthermore", "delve", "tapestry") and by repeated phrasing. The lists
below are a starting point, not a validated detector. Pass your own list to override them.
"""

from typing import Callable, Iterable, List, Sequence, Set

# Stock phrases that show up in AI-written prose. Sources for the pattern: see the README.
STOCK_PHRASES: List[str] = [
    "delve",
    "tapestry",
    "pivotal",
    "realm",
    "testament",
    "underscore",
    "multifaceted",
    "intricate",
    "game-changer",
    "dive into",
    "embark on",
    "navigate the complexities",
    "it's important to note",
    "it is important to note",
    "in conclusion",
    "furthermore",
    "moreover",
]


def build_bad_word_ids(encode: Callable[[str], Sequence[int]], phrases: Iterable[str] = STOCK_PHRASES) -> List[List[int]]:
    """Token-id sequences to block. ``encode`` maps text to ids without special tokens.

    Each phrase is tried as-is and with a capital first letter, both bare and after a space,
    since the same word is tokenised differently in each form.
    """
    seqs: List[List[int]] = []
    for phrase in phrases:
        variants = {phrase, phrase.capitalize(), " " + phrase, " " + phrase.capitalize()}
        for variant in sorted(variants):
            ids = list(encode(variant))
            if ids and ids not in seqs:
                seqs.append(ids)
    return seqs


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


def banned_by_no_repeat_ngram(ids: Sequence[int], n: int) -> Set[int]:
    """Next tokens that would repeat an n-gram already present in ``ids``."""
    if n <= 0 or len(ids) < n:
        return set()
    if n == 1:
        return set(ids)
    tail = tuple(ids[len(ids) - (n - 1):])
    banned: Set[int] = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == tail:
            banned.add(ids[i + n - 1])
    return banned
