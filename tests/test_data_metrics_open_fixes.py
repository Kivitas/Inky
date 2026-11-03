"""Tests for the open data and metrics findings: DATA-02 (streaming dedupe), DATA-03 (short forms),
DATA-04 (BOM, null text, whitespace-only text, empty output), DATA-05 (reserved ids), MET-05 (label
validation) and MET-07 (split_heldout, expected to fail until evaluation.py is fixed).
Run with: python -B -m pytest tests/test_data_metrics_open_fixes.py -q -p no:cacheprovider
"""

import random
import sys
import tracemalloc
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky.data import (  # noqa: E402
    DEFAULT_RESERVED_IDS,
    FIRST_SPECIAL_ID,
    dedupe_documents,
    dedupe_stream,
    iter_jsonl,
    prepare_corpus,
    quality_filter,
)
from inky.metrics import binary_report, choice_report  # noqa: E402

PROSE = (
    "The river bends slowly past the old mill, and farmers walk their cattle along the muddy bank "
    "each morning before the sun rises over the hills. Children gather stones near the bridge while "
    "their parents trade stories about harvests, weather, and the price of grain in the market town."
)
OTHER = (
    "Winter arrived early that year, covering the orchard in a thin white layer. The keeper lit the "
    "stove in the small cabin and counted the firewood twice, wondering whether it would be enough "
    "until the thaw came down from the northern valleys and the roads opened again for travellers."
)


def _words_doc(seed: int, n_words: int) -> str:
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(10**9)}" for _ in range(n_words))


# DATA-02: streaming dedupe

def test_dedupe_stream_matches_dedupe_documents_on_mixed_input():
    docs = []
    for c in range(60):
        base = " ".join(f"c{c}t{j}" for j in range(60))
        docs.append(base)
        changed = base.split()
        changed[30] = "changed"
        docs.append(" ".join(changed))          # near-duplicate, Jaccard about 0.84
        docs.append(base.upper())               # exact duplicate after lowercasing
    docs += [PROSE, PROSE.upper(), OTHER, "   ", "", "!!!!", "   "]
    assert list(dedupe_stream(docs)) == dedupe_documents(docs)


def test_dedupe_stream_consumes_its_input_lazily():
    pulled = []

    def source():
        for i in range(5):
            pulled.append(i)
            yield f"doc {i} " + " ".join(f"u{i}w{j}" for j in range(30))

    gen = dedupe_stream(source())
    first = next(gen)
    assert first.startswith("doc 0")
    assert pulled == [0]                          # one document read to produce one output
    assert len(list(gen)) == 4 and pulled == [0, 1, 2, 3, 4]


def test_dedupe_stream_state_is_several_times_smaller_than_shingle_sets():
    docs = [_words_doc(i, 400) for i in range(40)]    # built before measuring, so not counted

    tracemalloc.start()
    try:
        dedupe_documents(docs)
        _, brute_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        kept = list(dedupe_stream(docs))
        _, stream_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(kept) == 40
    assert stream_peak * 4 < brute_peak, f"stream peak {stream_peak} vs shingle-set peak {brute_peak}"


def test_prepare_corpus_streams_and_reports_stats():
    docs = [PROSE, PROSE.upper(), OTHER, "short"]
    stats = {}
    packed = prepare_corpus(iter(docs), lambda s: [ord(c) for c in s], seq_len=16, eos_id=0, stats=stats)
    assert stats["input"] == 4 and stats["kept"] == 2
    assert packed.shape == ((len(PROSE) + 1 + len(OTHER) + 1) // 16, 16)


# DATA-03: short forms and layout

HAIKU = "Old pond\nfrog jumps into the water\nsplash of silence"
DIALOGUE = '"Where were you?"\n"Out walking."\n"Alone?"\n"With the dog, mostly."'
MICRO_FICTION = "She waited at the station until the last train left, then walked home and made tea for two."


def test_default_min_chars_keeps_haiku_dialogue_and_micro_fiction():
    for text in (HAIKU, DIALOGUE, MICRO_FICTION):
        assert quality_filter(text) is True


def test_explicit_min_chars_still_rejects_short_text():
    assert quality_filter(HAIKU, min_chars=200) is False
    assert quality_filter("Too short.") is False


def test_scene_breaks_and_refrain_in_a_chapter_are_kept():
    lines = []
    for i in range(10):
        lines.append(f"Paragraph {i}: " + _words_doc(i, 12))
        if i < 9:
            lines.append("* * *")
    text = "\n".join(lines)
    assert len(text.split("\n")) == 19 and sum(1 for x in text.split("\n") if x == "* * *") == 9
    assert quality_filter(text) is True


# DATA-04: bad input

def test_iter_jsonl_accepts_a_utf8_byte_order_mark(tmp_path):
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + '{"text": "first"}\n{"text": "second"}\n'.encode("utf-8"))
    assert list(iter_jsonl(path)) == ["first", "second"]


def test_iter_jsonl_skips_null_text_and_reports_non_string_with_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "ok"}\n{"text": null}\n{"text": 42}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 3.*int"):
        list(iter_jsonl(path))
    path.write_text('{"text": "ok"}\n{"text": null}\n', encoding="utf-8")
    assert list(iter_jsonl(path)) == ["ok"]


def test_quality_filter_rejects_whitespace_only_text():
    assert quality_filter(" " * 300) is False
    assert quality_filter("\n\t  \n" * 50) is False
    assert quality_filter(" " * 300, min_chars=0) is False


def test_prepare_corpus_warns_when_no_document_survives():
    with pytest.warns(UserWarning, match="kept no documents"):
        packed = prepare_corpus([], lambda s: [1] * len(s), seq_len=8, eos_id=0)
    assert packed.shape == (0, 8)
    with pytest.warns(UserWarning, match="kept no documents"):
        prepare_corpus(["   "] * 5, lambda s: [1] * len(s), seq_len=8, eos_id=0)


def test_prepare_corpus_rejects_non_string_documents():
    with pytest.raises(TypeError, match="index 1"):
        prepare_corpus([PROSE, None], lambda s: [1] * len(s), seq_len=8, eos_id=0)


# DATA-05: reserved ids in the data path

def _encode_with_markup(text: str):
    """Maps the literal marker <|im_start|> to a special id, the way a tokenizer that parses specials would."""
    if "<|im_start|>" in text:
        return [1, 2, FIRST_SPECIAL_ID, 3]
    return [ord(c) % 1000 for c in text]


# A body that shares no 5-grams with PROSE or OTHER, so dedupe does not remove the marked document first.
MARKED_BODY = (
    "Along the coast the fishermen mend their nets before dawn, counting lobster pots and arguing "
    "quietly about the tide, while gulls circle the harbour wall and the first boats slip out past the rocks."
)


def test_prepare_corpus_drops_documents_that_encode_to_reserved_ids():
    docs = [PROSE, "<|im_start|> " + MARKED_BODY, OTHER]
    stats = {}
    packed = prepare_corpus(docs, _encode_with_markup, seq_len=16, eos_id=0, stats=stats)
    assert stats["dropped_reserved"] == 1 and stats["kept"] == 2
    assert int(packed.max()) < FIRST_SPECIAL_ID


def test_scratch_ids_are_blocked_by_default_and_cannot_appear_in_packed_output():
    assert {50000, 50001, 50002} <= set(DEFAULT_RESERVED_IDS)

    def encode(text):
        return [50000 if ch == "[" else 50002 if ch == "]" else ord(ch) % 1000 for ch in text]

    docs = [PROSE, "a [note] " + MARKED_BODY, OTHER]
    stats = {}
    packed = prepare_corpus(docs, encode, seq_len=16, eos_id=0, stats=stats)
    assert stats["dropped_reserved"] == 1
    assert not set(packed.flatten().tolist()) & {50000, 50001, 50002}


def test_reserved_ids_check_can_be_disabled_explicitly():
    packed = prepare_corpus(
        ["<|im_start|> " + MARKED_BODY], _encode_with_markup, seq_len=4, eos_id=0, reserved_ids=frozenset()
    )
    assert packed.shape[0] == 1
    assert FIRST_SPECIAL_ID in packed.flatten().tolist()


# MET-05: label validation in metrics

def test_choice_gold_outside_the_options_raises_a_clear_error():
    probs = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    with pytest.raises(ValueError, match="item 0 is -1.*0 to 1"):
        choice_report(probs, gold=[-1, 1])
    with pytest.raises(ValueError, match="item 1 is 2"):
        choice_report(probs, gold=[0, 2])
    with pytest.raises(ValueError, match="item 0 is 1.5"):
        choice_report(probs, gold=[1.5, 0])


def test_choice_with_no_items_returns_none_fields_instead_of_an_index_error():
    report = choice_report(torch.zeros(0, 3), gold=[])
    assert report["items"] == 0
    assert report["top1_accuracy"] is None and report["mrr"] is None and report["nll"] is None
    assert report["macro_f1"] is None and report["micro_f1"] is None
    assert report["top2_accuracy"] is None


def test_choice_needs_at_least_one_option():
    with pytest.raises(ValueError, match="at least one option"):
        choice_report(torch.zeros(2, 0), gold=[0, 0])


def test_binary_labels_must_be_zero_or_one():
    probs = torch.tensor([0.9, 0.1, 0.6])
    with pytest.raises(ValueError, match="item 2 is 2"):
        binary_report(probs, labels=[1, 0, 2])
    with pytest.raises(ValueError, match="item 0 is -1"):
        binary_report(probs, labels=[-1, 0, 1])
    report = binary_report(probs, labels=[1, 0, 1])
    assert report["tp"] == 2 and report["fp"] == 0


def test_binary_with_no_items_returns_none_rates():
    report = binary_report(torch.zeros(0), labels=[])
    assert report["items"] == 0
    assert report["accuracy"] is None and report["precision"] is None and report["roc_auc"] is None


# MET-07: split_heldout never returns an empty train split.

def test_split_heldout_never_leaves_an_empty_train_split():
    from inky.evaluation import split_heldout

    with pytest.raises(ValueError, match="non-empty train split"):
        split_heldout([[1, 2, 3]], fraction=0.5, seed=0)
    with pytest.raises(ValueError):
        split_heldout([[1, 2], [3, 4]], fraction=0.9, seed=0)   # round(1.8) = 2 held out of 2
    train, held = split_heldout([[1, 2], [3, 4], [5, 6]], fraction=0.34, seed=0)     # round(1.02) = 1 held out
    assert len(train) >= 1 and len(held) == 1 and len(train) + len(held) == 3
