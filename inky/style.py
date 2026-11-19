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

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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

# Register vocabularies for formal vs. informal human-like prose modes.
# Formal mode avoids casual slang, colloquial contractions, and chat interjections.
INFORMAL_WORDS: List[str] = [
    "gonna",
    "wanna",
    "gotta",
    "kinda",
    "sorta",
    "yeah",
    "nah",
    "yep",
    "nope",
    "lol",
    "omg",
    "btw",
    "imo",
    "tbh",
    "ain't",
    "y'all",
    "stuff",
    "guys",
    "awesome",
    "totally",
    "literally",
    "basically",
]

INFORMAL_PHRASES: List[str] = [
    "you know",
    "kind of",
    "sort of",
    "no way",
    "for real",
    "by the way",
    "to be honest",
]

# Informal mode avoids stiff, bureaucratic, archaic, or overly academic transitions.
FORMAL_WORDS: List[str] = [
    "heretofore",
    "herein",
    "thereof",
    "whereby",
    "notwithstanding",
    "henceforth",
    "aforementioned",
    "inasmuch",
    "consequently",
    "subsequently",
    "nevertheless",
    "nonetheless",
    "furthermore",
    "moreover",
    "thus",
    "hence",
    "shall",
    "whilst",
    "utilize",
    "commence",
]

FORMAL_PHRASES: List[str] = [
    "in accordance with",
    "with respect to",
    "prior to",
    "subsequent to",
    "it is imperative",
    "one must",
    "for the purpose of",
    "in the event that",
]

STYLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "formal": {
        "temperature": 0.7,
        "top_p": 0.90,
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.12,
        "no_repeat_ngram_size": 4,
        "stock_word_penalty": 1.5,
        "avoid_words": INFORMAL_WORDS,
        "avoid_phrases": INFORMAL_PHRASES,
    },
    "informal": {
        "temperature": 0.85,
        "top_p": 0.95,
        "top_k": 60,
        "min_p": 0.03,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 4,
        "stock_word_penalty": 2.0,
        "avoid_words": FORMAL_WORDS,
        "avoid_phrases": FORMAL_PHRASES,
    },
}


def get_style_profile(style_mode: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a copy of the decoding profile for ``style_mode`` ('formal' or 'informal').

    ``overrides`` replaces individual profile entries (any key of the profile, e.g. ``stock_word_penalty``
    or ``avoid_words``); unknown keys are rejected so a typo cannot silently do nothing.
    """
    if style_mode not in STYLE_PROFILES:
        raise ValueError(f"style_mode must be one of {tuple(STYLE_PROFILES)}, got {style_mode!r}")
    prof = dict(STYLE_PROFILES[style_mode])
    if overrides:
        unknown = set(overrides) - set(prof)
        if unknown:
            raise ValueError(f"unknown style profile keys {sorted(unknown)}; valid: {sorted(prof)}")
        prof.update(overrides)
    return prof


def build_style_controls(
    style_mode: str,
    encode: Optional[Callable[[str], Sequence[int]]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build sampling and vocabulary controls for ``style_mode`` ('formal' or 'informal').

    When ``encode`` is given, also computes ``bad_word_ids`` (stock phrases + out-of-register phrases)
    and ``soft_penalty_ids`` (stock words + out-of-register words).
    """
    prof = get_style_profile(style_mode, overrides)
    avoid_words = prof.pop("avoid_words")
    avoid_phrases = prof.pop("avoid_phrases")
    if encode is not None:
        prof["bad_word_ids"] = build_bad_word_ids(
            encode, phrases=list(STOCK_PHRASES) + list(avoid_phrases)
        )
        prof["soft_penalty_ids"] = soft_penalty_token_ids(
            encode, words=list(STOCK_WORDS) + list(avoid_words)
        )
    else:
        prof["bad_word_ids"] = None
        prof["soft_penalty_ids"] = None
    return prof


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
    seen: Set[Tuple[int, ...]] = set()
    sources = list(phrases)
    if include_single_words:
        sources += list(words)
    for phrase in sources:
        for variant in _variants(phrase):
            ids = list(encode(variant))
            key = tuple(ids)
            if ids and key not in seen:
                seen.add(key)
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
    seen: Set[int] = set()
    for word in words:
        for variant in _variants(word):
            ids = list(encode(variant))
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
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


class BadWordIndex:
    """Precompiled bad-word sequences: ``banned(tail)`` costs one dict lookup per distinct prefix length
    instead of a scan over every sequence (the style profiles produce hundreds of variants).

    Gives exactly the result of ``banned_by_bad_words(tail, bad_word_ids)``. ``tail_len`` is the number of
    trailing context tokens that matter.
    """

    def __init__(self, bad_word_ids: Sequence[Sequence[int]]):
        self.singles: Set[int] = set()
        self.by_prefix: Dict[Tuple[int, ...], Set[int]] = {}
        for seq in bad_word_ids:
            if len(seq) == 1:
                self.singles.add(int(seq[0]))
            elif len(seq) > 1:
                self.by_prefix.setdefault(tuple(int(t) for t in seq[:-1]), set()).add(int(seq[-1]))
        self.prefix_lens = sorted({len(k) for k in self.by_prefix})
        self.tail_len = self.prefix_lens[-1] if self.prefix_lens else 0

    def banned(self, ids: Sequence[int]) -> Set[int]:
        out = set(self.singles)
        n = len(ids)
        for k in self.prefix_lens:
            if k <= n:
                hit = self.by_prefix.get(tuple(ids[n - k:]))
                if hit:
                    out |= hit
        return out


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
