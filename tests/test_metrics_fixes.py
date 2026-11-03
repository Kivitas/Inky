"""Tests for the length-normalised metrics, text-level blocked phrases, baselines and reference comparison.
Run with: python -m pytest tests/test_metrics.py tests/test_metrics_fixes.py -q -p no:cacheprovider
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky.metrics import baseline_reports, compare_to_reference, text_report  # noqa: E402


def approx(a, b, tol=1e-6):
    return a is not None and abs(a - b) < tol


def test_summary_keeps_old_keys_and_adds_normalised_ones():
    r = text_report([5, 6, 7, 5, 6, 7, 8])
    for key in ("distinct_1", "distinct_2", "distinct_3", "repeated_4gram_rate", "type_token_ratio",
                "most_frequent_token_share", "blocked_phrase_hits"):
        assert key in r, key
    for key in ("mattr", "distinct_1_windowed", "distinct_2_windowed", "blocked_phrase_text_hits"):
        assert key in r, key


def test_mattr_is_length_invariant_where_raw_ttr_is_not():
    pattern = [1, 2, 3, 4, 5]
    short, long = text_report(pattern * 10, window=10), text_report(pattern * 40, window=10)
    assert approx(short["mattr"], 0.5) and approx(long["mattr"], 0.5)
    assert short["type_token_ratio"] != long["type_token_ratio"]   # raw TTR moves with length


def test_windowed_metrics_hand_computed():
    # [1,2]*5 with window 4: every window has 2 distinct tokens of 4, and 2 distinct bigrams of 3
    r = text_report([1, 2] * 5, window=4)
    assert approx(r["mattr"], 0.5)
    assert approx(r["distinct_1_windowed"], 0.5)
    assert approx(r["distinct_2_windowed"], 2 / 3)
    # [1,1,2,3] with window 2: windows [1,1], [1,2], [2,3] give distinct-1 of 0.5, 1, 1
    r = text_report([1, 1, 2, 3], window=2)
    assert approx(r["distinct_1_windowed"], 5 / 6)
    assert approx(r["distinct_2_windowed"], 1.0)


def test_windowed_metrics_are_none_when_text_is_shorter_than_window():
    r = text_report([1, 2, 3], window=10)
    assert r["mattr"] is None and r["distinct_1_windowed"] is None and r["distinct_2_windowed"] is None


def test_text_level_blocked_phrases_are_case_insensitive():
    vocab = {1: "The", 2: " cat", 3: " sat", 4: " dog"}
    decode = lambda ids: "".join(vocab[i] for i in ids)  # noqa: E731
    r = text_report([1, 2, 3, 1, 2], decode=decode, blocked_phrases=["THE CAT", "dog"])
    assert r["blocked_phrase_text_hits"] == 2     # "The cat" twice; "dog" absent
    assert r["blocked_phrase_hits"] is None       # no id-level list given


def test_text_check_sees_phrases_the_id_check_cannot():
    decode = lambda ids: "".join({1: "a", 2: "b"}[i] for i in ids)  # noqa: E731
    r = text_report([1, 2], bad_word_ids=[[5, 6]], decode=decode, blocked_phrases=["ab"])
    assert r["blocked_phrase_hits"] == 0
    assert r["blocked_phrase_text_hits"] == 1


def test_text_phrases_require_decode_and_none_without_phrases():
    with pytest.raises(ValueError):
        text_report([1, 2, 3], blocked_phrases=["x"])
    assert text_report([1, 2], decode=lambda ids: "x")["blocked_phrase_text_hits"] is None


def test_shuffled_baseline_keeps_unigram_counts_and_breaks_order():
    ref = [1, 2, 3, 4] * 10
    b = baseline_reports(ref, seed=0)["shuffled"]
    original = text_report(ref)
    assert b["tokens"] == original["tokens"]
    assert approx(b["distinct_1"], original["distinct_1"])
    assert approx(b["most_frequent_token_share"], original["most_frequent_token_share"])
    assert b["distinct_2"] > original["distinct_2"]   # the periodic bigrams are gone


def test_repeated_baseline_is_degenerate():
    b = baseline_reports([1, 2, 3, 4] * 10, length=30, window=10)["repeated"]
    assert b["tokens"] == 30
    assert approx(b["distinct_1"], 1 / 30)
    assert approx(b["mattr"], 1 / 10)
    assert approx(b["distinct_2_windowed"], 1 / 9)
    assert baseline_reports([1, 2, 3, 4] * 10, window=10)["repeated"]["tokens"] == 40


def test_compare_to_reference_identical_sequences_give_zero_deltas():
    seq = [1, 2, 3, 4, 1, 2, 5]
    deltas = compare_to_reference(seq, seq, window=4)
    assert deltas["distinct_1"] == 0.0 and deltas["mattr"] == 0.0
    assert all(v in (0.0, None) for v in deltas.values())
    assert "tokens" not in deltas and "window" not in deltas


def test_compare_to_reference_reports_direction_of_change():
    d = compare_to_reference([1, 2, 3, 4], [1, 1, 1, 1], window=2)
    assert approx(d["distinct_1"], 0.75)          # 1.0 - 0.25
    assert approx(d["type_token_ratio"], 0.75)
    assert approx(d["mattr"], 0.5)                # 1.0 - 0.5


def test_compare_to_reference_is_none_when_either_side_is_too_short():
    d = compare_to_reference([1, 2, 3, 4, 5], [1, 2], window=3)
    assert d["mattr"] is None


def test_repeated_4gram_warning_is_in_the_docstring():
    doc = text_report.__doc__
    assert "no_repeat_ngram_size" in doc and "0 by construction" in doc
