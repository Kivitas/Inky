"""Tests for inky/data.py using in-memory strings and tmp_path files. Run with: python -m pytest tests/test_data.py -q"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky.data import (  # noqa: E402
    clean_text,
    dedupe_documents,
    iter_jsonl,
    pack_sequences,
    prepare_corpus,
    quality_filter,
)

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


def test_clean_text_normalises_unicode_newlines_spaces_and_blank_lines():
    raw = "café \r\nline one\t\t  two  \r\n\r\n\r\n\r\nend\x07"
    assert clean_text(raw) == "café\nline one two\n\n\nend"


def test_clean_text_keeps_newline_and_tab_but_drops_other_control_characters():
    assert clean_text("x\x00y\nz") == "xy\nz"
    assert clean_text("a\tb") == "a b"


def test_quality_filter_rejects_short_text_and_respects_min_chars():
    assert quality_filter("Too short.") is False
    assert quality_filter("a" * 50, min_chars=20) is True
    assert quality_filter("a" * 50, min_chars=100) is False


def test_quality_filter_keeps_clean_prose_and_rejects_symbol_heavy_text():
    assert quality_filter(PROSE) is True
    assert quality_filter("@#$%^&*" * 40) is False


HINDI = (
    "नदी धीरे-धीरे पुराने मिल के पास से बहती है। सुबह होते ही किसान अपने मवेशियों को किनारे की ओर ले जाते हैं, "
    "और बच्चे पुल के पास पत्थर इकट्ठा करते हैं। माता-पिता बाज़ार में फसल, मौसम और अनाज के दामों पर बात करते हैं। "
    "सर्दियों में पहाड़ों से आने वाली हवा ठंडी होती है, इसलिए गाँव के लोग शाम को अलाव के पास बैठकर कहानियाँ सुनाते हैं। "
    "धीरे-धीरे रात गहरी होती है और पूरा गाँव शांत हो जाता है।"
)
THAI = (
    "แม่น้ำไหลอย่างช้าๆ ผ่านโรงสีเก่าริมทุ่งข้าว ทุกเช้าชาวนาจะพาวัวไปตามคันดินที่ชื้นแฉะ "
    "ก่อนดวงอาทิตย์จะขึ้นเหนือเนินเขา เด็ก ๆ เก็บก้อนหินใกล้สะพานขณะที่พ่อแม่คุยกันเรื่องฤดูกาล "
    "สภาพอากาศ และราคาข้าวในตลาดของอำเภอ ฤดูหนาวมาถึงเร็วกว่าปีก่อน ผู้เฝ้าบ้านจึงจุดเตาไฟ "
    "ในกระท่อมเล็ก ๆ และนับฟืนที่เหลืออยู่อีกครั้งหนึ่ง"
)
CHINESE = (
    "河水缓慢地流过老磨坊，农夫们每天清晨都会牵着牛沿着泥泞的河岸走过。孩子们在桥边捡石头，"
    "父母则谈论收成、天气以及集市上粮食的价格。冬天比往年来得更早，看守房屋的老人点燃小屋里的炉火，"
    "数了又数剩下的柴火，心里想着这些木柴能否撑到春天。村里的老人常讲《河边的故事》，"
    "说河神会在大雪之后醒来！你信吗？孩子们笑着点头：信，当然信。夜色渐渐深了，村子安静下来，"
    "只有远处的河水仍在不停地流动……第二天一早，磨坊的门又被推开，新的麦子运了进来，"
    "人们忙碌起来，开始了又一年的劳作。"
)


def test_quality_filter_keeps_non_latin_scripts():
    for text in (HINDI, THAI, CHINESE):
        assert len(text) >= 200
        assert quality_filter(text) is True


def test_quality_filter_rejects_text_that_is_mostly_emoji_or_symbols():
    emoji = "😀🎉🔥🌈✨" * 50                      # 250 characters, all So
    arrows = "→★♥✓©®™€£¥" * 30                    # 300 characters, all S*
    assert quality_filter(emoji) is False
    assert quality_filter(arrows) is False
    assert quality_filter("@#$%^&*" * 40) is False


def test_quality_filter_rejects_too_many_repeated_lines():
    distinct = [f"Distinct sentence number {i} about rivers, mills and farms." for i in range(5)]
    repeated = "This line repeats over and over again in the document."
    mostly_repeated = "\n".join(distinct + [repeated] * 5)     # 5 of 10 lines repeat
    assert quality_filter(mostly_repeated) is False
    lightly_repeated = "\n".join(distinct + [repeated] * 2)  # 2 of 7 lines repeat, under 0.30
    assert quality_filter(lightly_repeated) is True


def test_dedupe_removes_exact_duplicates_after_clean_and_lowercase_keeping_first():
    docs = [PROSE, OTHER, PROSE.upper(), "  " + OTHER]
    result = dedupe_documents(docs)
    assert result == [PROSE, OTHER]
    assert result[0] is PROSE


def test_dedupe_max_docs_guard_raises_above_limit():
    docs = [PROSE, OTHER, PROSE]
    with pytest.raises(ValueError, match="shard"):
        dedupe_documents(docs, max_docs=2)
    assert dedupe_documents(docs, max_docs=3) == [PROSE, OTHER]
    assert dedupe_documents(docs) == [PROSE, OTHER]


def test_dedupe_removes_near_duplicate_differing_by_one_word():
    # 60 words, so one changed word leaves 5-gram Jaccard at about 0.84, above the 0.8 threshold.
    base = " ".join(f"item{i}" for i in range(60))
    variant = base.replace("item30", "changed")
    unrelated = " ".join(f"other{i}" for i in range(60))
    assert dedupe_documents([base, variant]) == [base]
    assert dedupe_documents([base, unrelated]) == [base, unrelated]


def test_pack_sequences_shapes_and_eos_placement():
    packed = pack_sequences([[1, 2, 3], [4, 5]], seq_len=3, eos_id=0)
    assert packed.dtype == torch.long
    assert packed.shape == (2, 3)
    # stream: 1 2 3 0 4 5 0 -> windows [1 2 3] [0 4 5]; the trailing lone 0 is dropped
    assert packed.tolist() == [[1, 2, 3], [0, 4, 5]]


def test_pack_sequences_places_eos_after_every_document():
    packed = pack_sequences([[1, 2], [3]], seq_len=2, eos_id=9)
    assert packed.tolist() == [[1, 2], [9, 3]]


def test_pack_sequences_returns_zero_rows_when_no_full_window():
    assert pack_sequences([[1]], seq_len=5, eos_id=0).shape == (0, 5)
    assert pack_sequences([], seq_len=4, eos_id=0).shape == (0, 4)


def test_pack_sequences_rejects_seq_len_below_two():
    with pytest.raises(ValueError):
        pack_sequences([[1, 2]], seq_len=1, eos_id=0)


def test_iter_jsonl_skips_blank_lines_and_missing_fields(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"text": "first"}\n'
        "\n"
        '{"other": 1}\n'
        '{"text": "héllo"}\n'
        "   \n"
        '{"text": "third"}',
        encoding="utf-8",
    )
    assert list(iter_jsonl(path)) == ["first", "héllo", "third"]
    # a non-string value in the requested field is rejected with its line number (DATA-04)
    with pytest.raises(ValueError, match="line 3"):
        list(iter_jsonl(str(path), field="other"))


def test_prepare_corpus_filters_dedupes_encodes_and_packs():
    def encode(s):
        return [ord(c) for c in s]

    docs = [PROSE, PROSE.upper(), "short text", OTHER]
    packed = prepare_corpus(docs, encode, seq_len=16, eos_id=0)

    kept_tokens = len(PROSE) + 1 + len(OTHER) + 1
    assert packed.shape == (kept_tokens // 16, 16)
    flat = packed.flatten().tolist()
    assert flat[0] == ord("T")
    assert flat[len(PROSE)] == 0           # EOS after the first kept document
    assert flat[len(PROSE) + 1] == ord("W")  # second kept document starts with its first letter


def test_prepare_corpus_returns_empty_tensor_when_everything_is_filtered():
    packed = prepare_corpus(["short"], lambda s: [1] * len(s), seq_len=8, eos_id=0)
    assert packed.shape == (0, 8)



def test_thirty_word_documents_with_one_changed_word_are_both_kept():
    # PLAN.md: 30-word documents with one changed word are kept (word 5-gram Jaccard is about 0.68 < 0.8).
    base = " ".join(f"word{i}" for i in range(30))
    changed = base.replace("word15", "other15")
    assert len(dedupe_documents([base, changed])) == 2
