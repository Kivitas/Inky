"""Tests for inky/style.py: n-gram scan equivalence and the single-word policy.

Run with: python -m pytest tests/test_style.py -q -p no:cacheprovider
"""

import importlib.util
import random
from pathlib import Path

# Load style.py by path: importing the ``inky`` package runs its __init__, which pulls in other
# modules that are not part of this test's scope.
_STYLE_PATH = Path(__file__).resolve().parents[1] / "inky" / "style.py"
_spec = importlib.util.spec_from_file_location("inky_style_under_test", _STYLE_PATH)
style = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(style)

NgramBanIndex = style.NgramBanIndex
STOCK_PHRASES = style.STOCK_PHRASES
STOCK_WORDS = style.STOCK_WORDS
banned_by_no_repeat_ngram = style.banned_by_no_repeat_ngram
build_bad_word_ids = style.build_bad_word_ids
soft_penalty_token_ids = style.soft_penalty_token_ids


def old_banned_by_no_repeat_ngram(ids, n):
    """Reference copy of the original full-scan implementation."""
    if n <= 0 or len(ids) < n:
        return set()
    if n == 1:
        return set(ids)
    tail = tuple(ids[len(ids) - (n - 1):])
    banned = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == tail:
            banned.add(ids[i + n - 1])
    return banned


def test_stateless_matches_old_scan_on_random_sequences():
    rng = random.Random(0)
    for _ in range(300):
        vocab = rng.randint(2, 12)
        length = rng.randint(0, 60)
        ids = [rng.randrange(vocab) for _ in range(length)]
        for n in range(0, 6):
            assert banned_by_no_repeat_ngram(ids, n) == old_banned_by_no_repeat_ngram(ids, n), (ids, n)


def test_incremental_index_matches_old_scan_at_every_step():
    rng = random.Random(1)
    for _ in range(100):
        vocab = rng.randint(2, 10)
        n = rng.randint(1, 5)
        index = NgramBanIndex(n)
        seq = []
        for _ in range(rng.randint(0, 80)):
            token = rng.randrange(vocab)
            seq.append(token)
            index.update(token)
            assert index.banned() == old_banned_by_no_repeat_ngram(seq, n), (seq, n)


def test_index_constructed_from_ids_matches_incremental():
    ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    for n in range(1, 5):
        built = NgramBanIndex(n, ids)
        assert built.banned() == old_banned_by_no_repeat_ngram(ids, n)
        assert len(built) == len(ids)


def test_known_values_unchanged():
    assert banned_by_no_repeat_ngram([1, 2, 3, 1, 2], 3) == {3}
    assert banned_by_no_repeat_ngram([1, 2, 3], 4) == set()


def test_index_does_not_rescan_per_call():
    # Only the (n-1)-token suffix key is looked up: a token appearing only as a non-suffix
    # context must not be banned.
    index = NgramBanIndex(3, [7, 8, 9, 1, 2])
    assert index.banned() == set()  # suffix (1, 2) never seen before
    index.update(7)  # suffix (2, 7); no earlier (2, 7) occurrence
    assert index.banned() == set()


def test_default_hard_ban_has_no_single_words():
    single = {w for w in STOCK_PHRASES if " " not in w and "-" not in w}
    assert single == set(), "single words must not be in the hard-ban phrase list"
    for word in ("realm", "moreover", "delve"):
        assert word not in STOCK_PHRASES
        assert word in STOCK_WORDS


_TOY_TABLE: dict = {}


def _toy_encode(text):
    # Stable vocabulary across calls: one id per piece, so "realm" is always the same id.
    # Single words encode to one token; a space-separated phrase encodes to several.
    out = []
    for piece in text.replace(" ", " _").split(" "):
        if piece == "":
            continue
        out.append(_TOY_TABLE.setdefault(piece.lower(), len(_TOY_TABLE) + 1))
    return out


def test_default_bad_words_skip_single_word_hard_bans():
    seqs = build_bad_word_ids(_toy_encode)
    # Multi-token phrase entries exist; no one-token entry for a stock word.
    assert any(len(s) > 1 for s in seqs)
    word_ids = {_toy_encode(w)[0] for w in STOCK_WORDS}
    single_banned = {s[0] for s in seqs if len(s) == 1}
    assert not (single_banned & word_ids)


def test_include_single_words_restores_hard_ban():
    seqs = build_bad_word_ids(_toy_encode, include_single_words=True)
    single_banned = {s[0] for s in seqs if len(s) == 1}
    assert _toy_encode("realm")[0] in single_banned


def test_soft_penalty_returns_single_token_word_ids_only():
    ids = soft_penalty_token_ids(_toy_encode, STOCK_WORDS)
    assert len(ids) == len(set(ids))
    assert _toy_encode("realm")[0] in ids
    assert _toy_encode("moreover")[0] in ids


def test_soft_penalty_skips_multi_token_forms():
    def split_encode(text):
        return [1, 2] if text.strip().lower() == "realm" else [3]

    assert soft_penalty_token_ids(split_encode, ["realm"]) == []
